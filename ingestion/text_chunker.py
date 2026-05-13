from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    """
    Split documents into smaller chunks for better semantic retrieval.
    chunk_overlap=150 ensures context isn't lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks
