from src.config import MAX_REVIEWS, MAX_SEARCH_ITERATIONS, MIN_SOURCES
from src.state import ResearchState


def should_continue_research(state: ResearchState) -> str:
    """Deterministic check: loop Researcher or proceed to Analyst."""
    if state["iteration_count"] >= MAX_SEARCH_ITERATIONS:
        return "proceed"
    if len(state["gathered_sources"]) >= MIN_SOURCES:
        return "proceed"
    return "continue"


def review_decision(state: ResearchState) -> str:
    """Deterministic check: approve report or send back to Writer."""
    if state["approved"]:
        return "approved"
    if state["review_count"] >= MAX_REVIEWS:
        return "approved"  # force approve after max revisions
    return "revise"
