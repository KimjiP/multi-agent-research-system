"""Tests for source classification and publication dates."""

import pytest
from src.sources import classify_source, published_date


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://eur-lex.europa.eu/eli/reg/2024/1689/oj", "official_eu"),
        ("https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai", "eu_guidance"),
        ("https://www.europarl.europa.eu/news/en/press-room/x", "eu_guidance"),
        ("https://www.nkom.no/kunstig-intelligens", "national_authority"),
        ("https://www.whitehouse.gov/presidential-actions/x", "national_authority"),
        ("https://www.gibsondunn.com/eu-ai-act-omnibus/", "legal_analysis"),
        ("https://artificialintelligenceact.eu/article/6/", "legal_analysis"),
        ("https://www.cms.law/en/int/publication/x", "legal_analysis"),
        ("https://www2.deloitte.com/x", "industry"),
        ("https://www.reuters.com/technology/x", "news_blog"),
    ],
)
def test_classify_source(url, expected):
    assert classify_source(url) == expected


def test_us_gov_is_not_eu_guidance():
    # The original classifier labelled every .gov site as EU guidance.
    assert classify_source("https://www.nist.gov/itl/ai-risk-management-framework") != "eu_guidance"


def test_suffix_match_respects_label_boundaries():
    assert classify_source("https://notgov.com/article") == "news_blog"
    assert classify_source("https://europa.eu.example.com/x") == "news_blog"


class TestPublishedDate:
    def test_iso_date_from_search_result(self):
        assert published_date({"url": "https://x.com/a", "published_date": "2026-07-24T10:00:00Z"}) == "2026-07-24"

    def test_rfc2822_date_from_search_result(self):
        result = {"url": "https://x.com/a", "published_date": "Fri, 24 Jul 2026 10:00:00 GMT"}
        assert published_date(result) == "2026-07-24"

    def test_date_in_url_path(self):
        assert published_date({"url": "https://www.insideglobaltech.com/2026/05/28/eu-ai-act/"}) == "2026-05-28"
        assert published_date({"url": "https://x.eu/news/2025-11-19-digital-omnibus"}) == "2025-11-19"

    def test_year_and_month_only(self):
        assert published_date({"url": "https://x.com/blog/2026/03/ai-act-update"}) == "2026-03-01"

    def test_eli_path_is_not_a_date(self):
        assert published_date({"url": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj"}) is None

    def test_no_date(self):
        assert published_date({"url": "https://example.com/ai-act"}) is None
