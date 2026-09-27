"""Tests for turning the Analyst's structured output into claims."""

from datetime import date

from src.nodes.analyst import Analysis, ExtractedClaim, ExtractedContradiction, build_findings
from src.state import SourceDoc

TODAY = date(2026, 9, 28)

SOURCES = [
    SourceDoc(url="https://eur-lex.europa.eu/a", title="AI Act", snippet="s",
              source_type="official_eu", published_date=None, search_query="q"),
    SourceDoc(url="https://news.com/b", title="News", snippet="s",
              source_type="news_blog", published_date="2026-09-01", search_query="q"),
]


def _analysis(claims, contradictions=()) -> Analysis:
    return Analysis(claims=list(claims), contradictions=list(contradictions), open_issues=["gap"])


def test_source_numbers_map_to_urls_and_confidence_is_computed():
    claims, _, open_issues = build_findings(
        _analysis([ExtractedClaim(text="c", source_numbers=[1, 2], category="timeline")]),
        SOURCES, TODAY,
    )
    assert claims[0].supporting_sources == ["https://eur-lex.europa.eu/a", "https://news.com/b"]
    assert claims[0].claim_id == "claim_001"
    assert claims[0].confidence.score == round(1.0 * 0.8 * 1.2, 2)
    assert open_issues == ["gap"]


def test_claims_citing_no_valid_source_are_dropped():
    claims, _, _ = build_findings(
        _analysis([
            ExtractedClaim(text="unsupported", source_numbers=[0, 7], category="interpretation"),
            ExtractedClaim(text="supported", source_numbers=[2, 2], category="enforcement"),
        ]),
        SOURCES, TODAY,
    )
    assert [c.text for c in claims] == ["supported"]
    assert claims[0].supporting_sources == ["https://news.com/b"]


def test_contradiction_with_invalid_source_number():
    _, contradictions, _ = build_findings(
        _analysis([], [ExtractedContradiction(claim_a="a", claim_b="b", source_a=1, source_b=9, description="d")]),
        SOURCES, TODAY,
    )
    assert contradictions[0].source_a == "https://eur-lex.europa.eu/a"
    assert contradictions[0].source_b == ""
