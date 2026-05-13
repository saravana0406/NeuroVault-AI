from prompts.system_prompt import SYSTEM_PROMPT


def generate_response(llm, query, docs, memory):
    """
    FIX 1: chat_history was dumped as a raw dict string like
           "{'chat_history': [HumanMessage(...), AIMessage(...)]}"
           which the LLM cannot parse. Now messages are formatted
           as readable "Human: ... / Assistant: ..." lines.

    FIX 2: Added explicit fallback when no docs are retrieved, instead of
           sending an empty context that confuses the model.

    FIX 3: Wrapped LLM invocation in try/except to surface API errors clearly.
    """

    # Build context from retrieved documents
    if docs:
        context = "\n\n".join(
            [f"[Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )
    else:
        return (
            "I could not find relevant information in the uploaded documents "
            "to answer your question. Please make sure the document has been "
            "uploaded and try rephrasing your query."
        )

    # FIX: Format chat history as readable text, not a raw Python dict
    raw_history = memory.load_memory_variables({})
    messages = raw_history.get("chat_history", [])

    if messages:
        formatted_history = ""
        for msg in messages:
            role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            formatted_history += f"{role}: {msg.content}\n"
    else:
        formatted_history = "No previous conversation."

    prompt = f"""{SYSTEM_PROMPT}

---

Previous Conversation:
{formatted_history}

---

Relevant Document Context:
{context}

---

User Question:
{query}

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer = response.content

        # Save to memory after successful response
        memory.save_context(
            {"input": query},
            {"output": answer},
        )

        return answer

    except Exception as e:
        raise RuntimeError(f"LLM invocation failed: {str(e)}") from e
