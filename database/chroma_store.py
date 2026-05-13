import os

from langchain_community.vectorstores import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "neurovault_docs"


def create_vector_store(chunks, embeddings):
    """
    Create a new ChromaDB vector store from document chunks.
    FIX: Explicit persist_directory and collection_name ensure
         load_vector_store() reads from the same location.
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectordb


def load_vector_store(embeddings):
    """
    Load an existing ChromaDB vector store.
    FIX: Uses the same CHROMA_DIR and COLLECTION_NAME constants —
         previously a mismatch between create and load caused retrieval
         to silently query an empty/different collection.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "No vector store found. Please upload a PDF document first."
        )

    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectordb
