from src.config import MAX_SEARCH_ITERATIONS, MIN_SOURCES
from src.state import ResearchState


def should_continue_research(state: ResearchState) -> str:
    """Deterministic check: loop Researcher or proceed to Analyst."""
    if state["iteration_count"] >= MAX_SEARCH_ITERATIONS:
        return "proceed"
    if len(state["gathered_sources"]) >= MIN_SOURCES:
        return "proceed"
    return "continue"


def review_decision(state: ResearchState) -> str:
    """Deterministic check: send the report back to the Writer, or end.

    The run ends when the report is approved, when it is still not approved
    after MAX_REVIEWS rounds, or when the review itself failed. Only an
    approved report ends with approved=True.
    """
    return "revise" if state["review_status"] == "revision_requested" else "end"
