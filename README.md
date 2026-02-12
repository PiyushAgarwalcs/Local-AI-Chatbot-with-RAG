# 🤖 Multi-Utility AI Chatbot (100% Local & Free)

A fully local, privacy-focused AI chatbot built with **Ollama**, **LangGraph**, and **Streamlit**. No API keys, no cloud services, no data leaving your machine.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)

---
![Uploading Screenshot 2026-02-12 154619.png…]()

## ✨ Features

- 📄 **PDF Question Answering (RAG)**: Upload PDFs and ask questions about their content
- 💬 **Multi-Thread Conversations**: Maintain separate conversation threads with memory
- 🧮 **Calculator**: Perform basic arithmetic operations
- 🗄️ **Persistent Memory**: SQLite-based conversation history across sessions
- 🔒 **100% Local & Private**: All processing happens on your machine
- 💰 **Zero Cost**: No API fees, completely free to run

---

## 🎯 Use Cases

- Analyze resumes and job descriptions
- Extract information from research papers
- Ask questions about legal documents
- Quick calculations during conversations
- Private document analysis without cloud uploads

---

## 🖥️ System Requirements

### Minimum
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **OS**: Windows 11
- **Python**: 3.10 or higher
  
---

## 🚀 Installation

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai)

**Pull required models:**
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**Start Ollama server:**
```bash
ollama serve
```
Keep this terminal running.

---

### 2. Setup Python Environment

**Clone or download this repository:**
```bash
git clone <your-repo-url>
cd chatbot-project
```

**Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Basic Chat
1. Type your message in the input box
2. Press Enter to send
3. Wait for the AI response

### PDF Question Answering
1. Click **"Upload PDF"** in the sidebar
2. Select a PDF file from your computer
3. Wait for indexing to complete
4. Ask questions about the PDF content

**Example Questions:**
- "What are the key skills mentioned in this resume?"
- "Summarize the main findings of this report"
- "What is the person's work experience?"

### Managing Conversations
- **New Chat**: Click "New Chat" to start a fresh conversation
- **Switch Threads**: Click on thread IDs in sidebar to switch between conversations
- **Persistent History**: All conversations are saved and restored automatically

---

## 🏗️ Project Structure

```
chatbot-project/
├── langgraph_rag_backend.py   # Core logic: LLM, RAG, graph setup
├── streamlit_app.py            # User interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── chatbot.db                  # SQLite database (auto-created)
```

---

## ⚙️ Configuration

### Change the LLM Model

Edit `langgraph_rag_backend.py`:

```python
llm = ChatOllama(
    model="llama3.2:3b",  # Change this
    temperature=0.3,
)
```

**Alternative models:**
- `tinyllama` - Faster, less accurate (~1GB RAM)
- `llama3.2:1b` - Lightweight (~1.5GB RAM)
- `llama3.1:8b` - More powerful (~6GB RAM)

### Adjust Chunk Size (for PDF processing)

Edit `langgraph_rag_backend.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,      # Smaller = faster, less context
    chunk_overlap=50,    # Overlap between chunks
)
```

---

## 🔧 Troubleshooting

### Issue: "Model not found" error
**Solution:** Pull the model first
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Issue: App is very slow/laggy
**Solutions:**
1. Close other heavy applications (browsers, IDEs)
2. Use a lighter model (`llama3.2:1b` or `tinyllama`)
3. Reduce chunk_size to 300 in the backend
4. Restart Ollama: `ollama serve`

### Issue: PDF questions not working well
**Solutions:**
1. Upload smaller PDFs (< 50 pages)
2. Ask specific questions instead of broad ones
3. Try rephrasing your question
4. Use a more powerful model like `llama3.1:8b`

### Issue: "Connection refused" error
**Solution:** Make sure Ollama is running
```bash
ollama serve
```

### Issue: Database locked error
**Solution:** Delete the database and restart
```bash
del chatbot.db  # Windows
rm chatbot.db   # macOS/Linux
```

---

### Expected Response Times
- **Simple chat**: 5-10 seconds
- **PDF upload**: 15-25 seconds (depending on size)
- **PDF question**: 15-20 seconds
- **Calculation**: 1-2 seconds

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **LLM Orchestration**: LangGraph (state machine for AI workflows)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Ollama embeddings (nomic-embed-text)
- **LLM**: Ollama (llama3.2:3b)
- **Database**: SQLite (conversation persistence)
- **Document Processing**: PyPDF, LangChain

---

## 🔒 Privacy & Security

- ✅ **100% Local Processing**: No data sent to external servers
- ✅ **No Telemetry**: No usage tracking or analytics
- ✅ **No API Keys Required**: Fully offline operation
- ✅ **Local Storage**: All conversations stored on your machine
- ⚠️ **Note**: SQLite database is unencrypted. For sensitive data, implement encryption.

---

## 🚧 Known Limitations

1. **RAG Accuracy**: Local models may not match GPT-4 quality for complex PDF queries
2. **Performance**: Slower than cloud-based solutions on low-end hardware
3. **Context Window**: Can cause issues with very long conversations
4. **Tool Calling**: Local models have limited function calling capabilities 
5. **Languages**: Best performance with English content

---

## 🔮 Future Improvements

- [ ] Add web search capability
- [ ] Support for more document formats (Word, Excel, PowerPoint)
- [ ] Better conversation export (JSON, Markdown)
- [ ] Custom system prompts per thread
- [ ] Model switching from UI
- [ ] Multi-document RAG (query across multiple PDFs)
- [ ] Voice input/output
- [ ] Dark mode UI

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM runtime
- **LangChain/LangGraph** - LLM orchestration framework
- **Streamlit** - Web interface framework
- **Meta AI** - Llama models
- **Nomic AI** - Embedding models

---

## 📧 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [piyushagarwal2003k@gmail.com]

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Install Ollama and pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Start Ollama
ollama serve

# 3. Setup Python
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 4. Run
streamlit run streamlit_app.py
```

**That's it!** 🎉

---

**Built with ❤️ using 100% local, open-source technologies**
