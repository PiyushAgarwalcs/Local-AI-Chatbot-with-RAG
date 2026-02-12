import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

st.set_page_config(page_title="Multi Utility Chatbot", layout="wide")

# ==============================
# Session Init
# ==============================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "history" not in st.session_state:
    st.session_state.history = []

if "threads" not in st.session_state:
    st.session_state.threads = retrieve_all_threads()

thread_id = st.session_state.thread_id

# ==============================
# Sidebar
# ==============================

st.sidebar.title("Threads")

if st.sidebar.button("New Chat"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.rerun()

for t in st.session_state.threads:
    if st.sidebar.button(t):
        st.session_state.thread_id = t
        st.session_state.history = []
        st.rerun()

st.sidebar.divider()

uploaded = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded:
    summary = ingest_pdf(uploaded.getvalue(), thread_id, uploaded.name)
    st.sidebar.success(f"Indexed {summary['chunks']} chunks")

doc_meta = thread_document_metadata(thread_id)
if doc_meta:
    st.sidebar.info(f"Using {doc_meta['filename']}")

# ==============================
# Chat UI
# ==============================

st.title("Multi Utility Chatbot (Ollama Local)")

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Add to history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Show assistant response
    with st.chat_message("assistant"):
        CONFIG = {"configurable": {"thread_id": thread_id}}

        def stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):
                    yield chunk.content

        response = st.write_stream(stream())

    # Add assistant response to history
    st.session_state.history.append({"role": "assistant", "content": response})
    
    # Rerun to show the updated history properly
    st.rerun()