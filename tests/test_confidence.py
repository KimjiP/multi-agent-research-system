"""Tests for claim confidence computed in code."""

from datetime import date, timedelta

import pytest
from src.confidence import recency_multiplier, score_claim
from src.state import SourceDoc

TODAY = date(2026, 9, 28)


def _source(url: str, source_type: str, days_old: int | None = None) -> SourceDoc:
    published = (TODAY - timedelta(days=days_old)).isoformat() if days_old is not None else None
    return SourceDoc(
        url=url, title="t", snippet="s", source_type=source_type,
        published_date=published, search_query="q",
    )


def test_undated_official_source():
    conf = score_claim([_source("https://eur-lex.europa.eu/a", "official_eu")], TODAY)
    assert (conf.authority, conf.recency, conf.corroboration) == (1.0, 0.8, 1.0)
    assert conf.score == 0.8
    assert conf.undated


def test_recent_news_source():
    conf = score_claim([_source("https://news.com/a", "news_blog", days_old=10)], TODAY)
    assert conf.score == 0.15
    assert not conf.undated


def test_strongest_single_source_sets_authority_and_recency():
    sources = [
        _source("https://eur-lex.europa.eu/a", "official_eu"),  # 1.0 × 0.8
        _source("https://lawfirm.com/a", "legal_analysis", days_old=5),  # 0.5 × 1.0
    ]
    conf = score_claim(sources, TODAY)
    assert (conf.authority, conf.recency) == (1.0, 0.8)
    assert conf.corroboration == 1.2


def test_corroboration_counts_distinct_domains():
    same_site = [_source("https://news.com/a", "news_blog", 1), _source("https://news.com/b", "news_blog", 1)]
    assert score_claim(same_site, TODAY).corroboration == 1.0
    three_sites = same_site[:1] + [
        _source("https://other.com/a", "news_blog", 1),
        _source("https://third.org/a", "news_blog", 1),
    ]
    assert score_claim(three_sites, TODAY).corroboration == 1.4


def test_score_capped_at_one():
    sources = [
        _source("https://eur-lex.europa.eu/a", "official_eu", 1),
        _source("https://ec.europa.eu/a", "eu_guidance", 1),
    ]
    assert score_claim(sources, TODAY).score == 1.0


@pytest.mark.parametrize("days, expected", [(0, 1.0), (90, 1.0), (100, 0.8), (200, 0.5), (400, 0.2)])
def test_recency_buckets(days, expected):
    assert recency_multiplier((TODAY - timedelta(days=days)).isoformat(), TODAY) == expected


def test_recency_of_undated_source_is_none():
    assert recency_multiplier(None, TODAY) is None


def test_claim_without_sources_rejected():
    with pytest.raises(ValueError):
        score_claim([], TODAY)


@pytest.mark.parametrize("score, label", [(0.82, "high"), (0.7, "high"), (0.69, "medium"), (0.4, "medium"), (0.39, "low")])
def test_confidence_label(score, label):
    from src.confidence import confidence_label

    assert confidence_label(score) == label
