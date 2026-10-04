"""Tests for turning a search result into a source (no API calls)."""

from src.config import SNIPPET_CHARS
from src.nodes.researcher import source_from_result


def test_search_result_becomes_a_source():
    result = {
        "url": "https://www.example-law.com/2026/09/15/ai-act",
        "title": "AI Act update",
        "content": "x" * (SNIPPET_CHARS + 100),
    }
    source = source_from_result(result, "AI Act dates")
    assert source.title == "AI Act update" and source.search_query == "AI Act dates"
    assert len(source.snippet) == SNIPPET_CHARS
    assert source.source_type == "legal_analysis"  # the host contains "law"
    assert source.published_date == "2026-09-15"  # from the URL path


def test_missing_fields_give_empty_text():
    source = source_from_result({"url": "https://news.com/a"}, "q")
    assert (source.title, source.snippet, source.published_date) == ("", "", None)
    assert source.source_type == "news_blog"
