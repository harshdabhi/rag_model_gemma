# 🔍 RAG Chatbot with LangChain, Gemma, and Streamlit

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using:
- **LangChain** for chaining LLMs and tools
- **FAISS** for fast vector similarity search
- **Ollama + Gemma** for local LLM inference
- **nomic-embed-text** for embeddings
- **Streamlit** for the user interface

🚫 No cloud APIs or proprietary services required — it all runs **locally**!

---

## 📦 Features

- 📄 Upload multiple PDFs
- 🧠 Chunk, embed, and store with FAISS
- 🔍 Search relevant context using vector similarity
- 💬 Generate answers from open-source LLM (Gemma) with full grounding
- 🖥️ Fully interactive UI with Streamlit
- 🔐 Local-first and privacy-friendly

---

## 🛠️ Tech Stack

| Tool            | Purpose                                  |
|-----------------|------------------------------------------|
| LangChain       | RAG logic & orchestration                |
| FAISS           | Vector store for similarity search       |
| Ollama          | Run open-source LLMs locally             |
| Gemma 3B        | LLM for answering questions              |
| nomic-embed-text| Embedding model for document chunks      |
| PyPDFDirectoryLoader | Load all PDFs from a folder         |
| Streamlit       | Web app front-end                        |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt

```

### 2. Install Ollama and models

Install ollama on system and open terminal and type command

```bash

ollama run gemma:2b
ollama run nomic-embed-text
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

