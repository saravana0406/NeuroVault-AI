import streamlit as st


def render_ui():
    """Render the NeuroVault AI sidebar and main query input."""

    st.set_page_config(
        page_title="NeuroVault AI",
        page_icon="🧠",
        layout="wide",
    )

    # Sidebar — PDF upload
    with st.sidebar:
        st.title("🧠 NeuroVault AI")
        st.markdown("**Enterprise RAG Knowledge Assistant**")
        st.divider()

        uploaded_file = st.file_uploader(
            "Upload Enterprise Document (PDF)",
            type=["pdf"],
            help="Upload a PDF to build your knowledge base.",
        )

        st.divider()
        st.caption("Powered by LangGraph · ChromaDB · Groq LLaMA3")

    # Main area — query input
    st.header("💬 Ask Your Document")

    query = st.chat_input("Ask a question about your uploaded document...")

    return uploaded_file, query
