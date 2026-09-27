"""Tests for the Reviewer's deterministic link check and its approval rules."""

import pytest
import src.nodes.reviewer as reviewer
from src.config import MAX_REVIEWS
from src.llm import LLMError
from src.state import CheckResult, SourceDoc, initial_state

SOURCE = SourceDoc(url="https://eur-lex.europa.eu/a", title="AI Act", snippet="s",
                   source_type="official_eu", published_date=None, search_query="q")


def _state(report: str, review_count: int = 0) -> dict:
    state = initial_state("q")
    state.update(report=report, gathered_sources=[SOURCE], review_count=review_count)
    return state


def _review(**failed: str) -> reviewer.Review:
    checks = {
        name: CheckResult(passed=name not in failed, notes=failed.get(name, "ok"))
        for name in reviewer.LLM_CHECKS
    }
    return reviewer.Review(**checks, revision_instructions="fix it" if failed else "")


@pytest.fixture
def llm_review(monkeypatch):
    """Replace the LLM call with a fixed Review (or an exception)."""
    def install(result):
        def fake(*args, **kwargs):
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(reviewer, "generate_structured", fake)
    return install


def test_link_check_accepts_gathered_sources():
    assert reviewer.check_source_links("See [AI Act](https://eur-lex.europa.eu/a).", [SOURCE]).passed


def test_link_check_rejects_unknown_url():
    check = reviewer.check_source_links("See [Press](https://ec.europa.eu/fake).", [SOURCE])
    assert not check.passed
    assert "https://ec.europa.eu/fake" in check.notes


def test_all_checks_pass_approves(llm_review):
    llm_review(_review())
    out = reviewer.reviewer_node(_state("[AI Act](https://eur-lex.europa.eu/a)"))
    assert out["approved"] and out["review_status"] == "approved"
    assert out["review_feedback"] is None


def test_failed_check_requests_revision(llm_review):
    llm_review(_review(hallucination_check="2026 date not in claims"))
    out = reviewer.reviewer_node(_state("report"))
    assert not out["approved"]
    assert out["review_status"] == "revision_requested"
    assert out["review_feedback"] == "fix it"


def test_not_approved_after_max_reviews(llm_review):
    llm_review(_review(citation_coverage="uncited sentence"))
    out = reviewer.reviewer_node(_state("report", review_count=MAX_REVIEWS - 1))
    assert out["review_status"] == "not_approved"
    assert not out["approved"]


def test_unknown_link_blocks_approval_even_if_llm_passes(llm_review):
    llm_review(_review())
    out = reviewer.reviewer_node(_state("See [Press](https://ec.europa.eu/fake)."))
    assert not out["approved"]
    assert not out["review_checklist"]["source_links"].passed
    assert "cite only the listed sources" in out["review_feedback"]


def test_review_failure_fails_closed(llm_review):
    # The original reviewer approved the report when its own output could not be parsed.
    llm_review(LLMError("no structured output"))
    out = reviewer.reviewer_node(_state("report"))
    assert out["review_status"] == "review_failed"
    assert not out["approved"]
    assert out["error"].startswith("Review failed")
