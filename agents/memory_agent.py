from langchain.memory import ConversationBufferMemory


def get_memory():
    """
    FIX: Memory was a module-level singleton, meaning it was shared across
    all users/sessions and never reset. Now it returns a new instance,
    managed by Streamlit session_state in app.py for proper per-session isolation.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return memory
