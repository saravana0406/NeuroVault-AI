import os
import tempfile

import streamlit as st

from ui.streamlit_ui import render_ui

from ingestion.pdf_loader import load_pdf
from ingestion.text_chunker import split_documents
from ingestion.embedding_model import get_embedding_model

from database.chroma_store import (
    create_vector_store,
    load_vector_store,
)

from agents.llm_agent import get_llm
from agents.memory_agent import get_memory

from graph.workflow import build_graph


# -----------------------------
# SESSION STATE INITIALIZATION
# FIX: vectordb and memory must persist across Streamlit reruns,
#      otherwise the uploaded PDF is lost and memory resets every query.
# -----------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "memory" not in st.session_state:
    st.session_state.memory = get_memory()


# -----------------------------
# STREAMLIT UI
# -----------------------------
uploaded_file, query = render_ui()


# -----------------------------
# FILE UPLOAD PROCESSING
# -----------------------------
if uploaded_file:

    # Allow ONLY PDF
    if uploaded_file.type != "application/pdf":
        st.error("Only PDF files are supported currently.")
        st.stop()

    try:
        # Create Temp Directory
        temp_dir = tempfile.mkdtemp()

        # Save Uploaded File
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        docs = load_pdf(file_path)

        if not docs:
            st.error("Could not extract text from the PDF. It may be scanned or image-based.")
            st.stop()

        # Split Text
        chunks = split_documents(docs)

        # Embedding Model
        embeddings = get_embedding_model()

        # Create Vector DB and persist in session state
        # FIX: was not stored in session_state — lost on every rerun
        st.session_state.vectordb = create_vector_store(chunks, embeddings)

        st.success(f"✅ PDF processed successfully — {len(chunks)} chunks indexed.")

    except Exception as e:
        st.error(f"Error Processing PDF: {str(e)}")
        st.stop()


# -----------------------------
# QUERY PROCESSING
# -----------------------------
if query:

    # FIX: guard against querying before any PDF is uploaded
    if st.session_state.vectordb is None:
        try:
            embeddings = get_embedding_model()
            st.session_state.vectordb = load_vector_store(embeddings)
        except Exception:
            st.warning("⚠️ Please upload a PDF document before asking a question.")
            st.stop()

    try:
        # Load LLM
        llm = get_llm()

        # Build LangGraph Workflow
        app = build_graph()

        # Invoke Graph — pass memory from session state so it persists
        result = app.invoke({
            "query": query,
            "vectordb": st.session_state.vectordb,
            "llm": llm,
            "memory": st.session_state.memory,
            "docs": [],
            "answer": "",
        })

        # Display Answer
        st.subheader("💬 Answer")
        st.write(result["answer"])

    except Exception as e:
        st.error(f"Error Generating Response: {str(e)}")
