from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str):
    """Load a PDF and return a list of LangChain Document objects (one per page)."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs
