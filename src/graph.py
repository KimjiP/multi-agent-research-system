from langgraph.graph import END, StateGraph

from src.config import RESEARCHER
from src.edges import review_decision, should_continue_research
from src.nodes.analyst import analyst_node
from src.nodes.researcher import researcher_node
from src.nodes.reviewer import reviewer_node
from src.nodes.tool_researcher import tool_researcher_node
from src.nodes.writer import writer_node
from src.state import ResearchState

RESEARCHERS = {"fixed": researcher_node, "tools": tool_researcher_node}


def build_graph(researcher: str = RESEARCHER):
    """The V2 graph, with the Researcher named in config.RESEARCHER unless another is given."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("researcher", RESEARCHERS[researcher])
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("researcher")

    if researcher == "fixed":
        workflow.add_conditional_edges(
            "researcher",
            should_continue_research,
            {
                "continue": "researcher",
                "proceed": "analyst",
            },
        )
    else:
        # Claude decides inside the node when the research is done, so the graph has no research loop
        workflow.add_edge("researcher", "analyst")

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
