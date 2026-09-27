"""Tests for the deterministic graph edges."""

import pytest
from src.config import MAX_SEARCH_ITERATIONS, MIN_SOURCES
from src.edges import review_decision, should_continue_research


def test_research_continues_until_enough_sources():
    assert should_continue_research({"iteration_count": 1, "gathered_sources": []}) == "continue"
    assert should_continue_research({"iteration_count": 1, "gathered_sources": [0] * MIN_SOURCES}) == "proceed"


def test_research_stops_at_max_iterations():
    state = {"iteration_count": MAX_SEARCH_ITERATIONS, "gathered_sources": []}
    assert should_continue_research(state) == "proceed"


@pytest.mark.parametrize(
    "status, expected",
    [
        ("revision_requested", "revise"),
        ("approved", "end"),
        ("not_approved", "end"),
        ("review_failed", "end"),
    ],
)
def test_review_decision(status, expected):
    assert review_decision({"review_status": status}) == expected
