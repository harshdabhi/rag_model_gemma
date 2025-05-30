import os
import shutil
from io import BytesIO

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Constants
TEMP_DIR = "temp_uploaded_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)

# Streamlit setup
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 PDF-based RAG Chatbot")

# Upload PDFs
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

# Save files to temp dir and load with PyPDFDirectoryLoader
if uploaded_files:
    with st.spinner("Saving PDFs..."):
        for file in uploaded_files:
            with open(os.path.join(TEMP_DIR, file.name), "wb") as f:
                f.write(file.read())

    # Load documents with PyPDFDirectoryLoader
    with st.spinner("Loading documents..."):
        loader = PyPDFDirectoryLoader(TEMP_DIR)
        docs = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_documents(docs)

        # Store in session
        st.session_state.documents = documents
        st.success(f"Loaded and split {len(documents)} chunks from PDFs.")

# If documents are ready, run embedding + QA
if "documents" in st.session_state and st.session_state.documents:
    documents = st.session_state.documents

    # Embedding
    with st.spinner("Creating vector store..."):
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://host.docker.internal:11434"
        )
        db = FAISS.from_documents(documents, embeddings)

    # Prompt and LLM
    prompt = ChatPromptTemplate.from_template("""
    Use the following context to answer the question as accurately as possible.
    <context>
    {context}
    </context>
    Question: {input}
    """)

    llm = Ollama(
        model="Gemma3:1b",
        base_url="http://host.docker.internal:11434",
        temperature=0.9
    )

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Ask a question
    user_query = st.text_input("Ask a question based on your uploaded PDFs:")

    if user_query:
        with st.spinner("Generating answer..."):
            result = retrieval_chain.invoke({"input": user_query})
            st.markdown("### 💬 Answer")
            st.write(result["answer"])

# Optional: Reset
if st.button("🔁 Reset uploaded files"):
    shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    st.session_state.documents = []
    st.success("Temporary files and session state cleared.")
