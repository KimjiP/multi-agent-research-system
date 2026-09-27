from langgraph.graph import END, StateGraph

from src.edges import review_decision, should_continue_research
from src.nodes.analyst import analyst_node
from src.nodes.researcher import researcher_node
from src.nodes.reviewer import reviewer_node
from src.nodes.writer import writer_node
from src.state import ResearchState


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("researcher")

    workflow.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "continue": "researcher",
            "proceed": "analyst",
        },
    )

    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        review_decision,
        {
            "revise": "writer",
            "end": END,
        },
    )

    return workflow.compile()
