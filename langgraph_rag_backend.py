from __future__ import annotations

import os
import sqlite3
import tempfile
import re
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

# ==============================
# 1. MODELS
# ==============================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ==============================
# 2. Thread-Based Retriever Store
# ==============================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """Ingest a PDF and create a vector store for retrieval."""
    if not file_bytes:
        raise ValueError("No file received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        os.remove(temp_path)


# ==============================
# 3. Simple Calculator
# ==============================

def calculate(expression: str) -> str:
    """Safely evaluate simple math expressions."""
    try:
        # Remove any non-math characters for safety
        safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        result = eval(safe_expr)
        return f"Result: {result}"
    except:
        return "Error: Could not calculate. Use format like: 5 + 3 or 10 * 2"


# ==============================
# 4. State
# ==============================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==============================
# 5. Main Chat Node (No Tool Calling)
# ==============================

def chat_node(state: ChatState, config=None):
    """Main chat node that handles PDF RAG directly."""
    thread_id = config.get("configurable", {}).get("thread_id")
    user_message = state["messages"][-1].content
    
    # Check if PDF exists
    retriever = _get_retriever(thread_id)
    has_pdf = retriever is not None
    pdf_name = _THREAD_METADATA.get(thread_id, {}).get("filename", "")
    
    # Check if user is asking about calculations
    calc_patterns = [r'\d+\s*[\+\-\*/]\s*\d+', r'calculate', r'compute', r'what is \d+']
    is_calc_query = any(re.search(pattern, user_message.lower()) for pattern in calc_patterns)
    
    if is_calc_query:
        # Extract and calculate
        result = calculate(user_message)
        return {"messages": [AIMessage(content=result)]}
    
    # If PDF exists and user is asking about it, do RAG
    if has_pdf:
        # Retrieve relevant chunks
        docs = retriever.invoke(user_message)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Create enhanced prompt with context
        system = SystemMessage(
            content=f"""You are a helpful assistant answering questions about a PDF document.

PDF Filename: {pdf_name}

IMPORTANT: Use ONLY the information from the context below to answer. If the answer is not in the context, say "I don't see that information in the PDF."

CONTEXT FROM PDF:
{context}

Answer the user's question based on this context."""
        )
        
        messages = [system, HumanMessage(content=user_message)]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    else:
        # No PDF - general chat
        system = SystemMessage(
            content="""You are a helpful assistant. 

If the user asks about a PDF or document, tell them to upload one first using the sidebar.

For math questions, I can help calculate simple expressions like "5 + 3" or "10 * 2"."""
        )
        
        messages = [system] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}


# ==============================
# 6. Graph Setup
# ==============================

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ==============================
# 7. Helpers
# ==============================

def retrieve_all_threads():
    """Get all conversation threads."""
    threads = set()
    for cp in checkpointer.list(None):
        threads.add(cp.config["configurable"]["thread_id"])
    return list(threads)


def thread_document_metadata(thread_id: str):
    """Get metadata about uploaded document for a thread."""
    return _THREAD_METADATA.get(str(thread_id), {})