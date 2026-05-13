from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the sentence-transformers embedding model.
    all-MiniLM-L6-v2 is fast, lightweight, and good for semantic search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings
