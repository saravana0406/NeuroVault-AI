from typing import Any, List

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from agents.retriever_agent import retrieve_context
from agents.response_agent import generate_response


# FIX: State must declare ALL keys passed in app.invoke().
#      Missing keys cause KeyError inside graph nodes silently swallowing results.
class GraphState(TypedDict):
    query: str
    vectordb: Any
    llm: Any
    memory: Any
    docs: List[Document]
    answer: str


# ─── Node: Retrieve ───────────────────────────────────────────────────────────

def retriever_node(state: GraphState) -> GraphState:
    """Retrieve semantically similar document chunks for the query."""
    docs = retrieve_context(state["vectordb"], state["query"])
    return {**state, "docs": docs}


# ─── Node: Generate ───────────────────────────────────────────────────────────

def response_node(state: GraphState) -> GraphState:
    """Generate a grounded answer from retrieved docs + conversation memory."""
    answer = generate_response(
        llm=state["llm"],
        query=state["query"],
        docs=state["docs"],
        memory=state["memory"],
    )
    return {**state, "answer": answer}


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", response_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", END)

    return graph.compile()
