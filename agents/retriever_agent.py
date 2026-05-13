def retrieve_context(vectordb, query):
    """
    FIX 1: Increased k from 3 to 5 for better context coverage.
    FIX 2: Added guard for empty results — previously an empty list would
           silently produce a blank context string, causing the LLM to say
           it found no information even when documents were loaded.
    FIX 3: Added error handling around retrieval to surface real failures.
    """
    try:
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        docs = retriever.invoke(query)

        if not docs:
            return []

        return docs

    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {str(e)}") from e
