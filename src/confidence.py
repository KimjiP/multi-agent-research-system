"""Claim confidence, computed in code from the sources that support a claim.

    confidence = authority × recency × corroboration   (capped at 1.0)

- authority and recency come from the strongest single source: the one with the
  highest authority × recency. A primary legal text outranks commentary, and a
  recent source outranks an old one.
- corroboration counts distinct domains, so three pages from one site do not
  count as three independent sources.

The Analyst only decides which sources support a claim. The number is computed
here, so the same sources always give the same score.
"""

from datetime import date

from src.config import (
    CORROBORATION,
    CORROBORATION_MAX,
    RECENCY_MULTIPLIERS,
    SOURCE_AUTHORITY_WEIGHTS,
    UNDATED_MULTIPLIER,
)
from src.sources import host_of
from src.state import Confidence, SourceDoc


def confidence_label(score: float) -> str:
    """The label the report uses: high from 0.7, medium from 0.4, low below."""
    if score >= 0.7:
        return "high"
    return "medium" if score >= 0.4 else "low"


def recency_multiplier(published: str | None, today: date) -> float | None:
    """Multiplier for a source's age, or None if the source has no date."""
    if not published:
        return None
    age_days = (today - date.fromisoformat(published)).days
    for max_days, multiplier in RECENCY_MULTIPLIERS:
        if max_days is None or age_days <= max_days:
            return multiplier
    return RECENCY_MULTIPLIERS[-1][1]


def score_claim(sources: list[SourceDoc], today: date) -> Confidence:
    """Confidence for a claim supported by `sources` (at least one)."""
    if not sources:
        raise ValueError("A claim needs at least one supporting source")

    def evidence(source: SourceDoc) -> tuple[float, float, bool]:
        recency = recency_multiplier(source.published_date, today)
        authority = SOURCE_AUTHORITY_WEIGHTS.get(source.source_type, SOURCE_AUTHORITY_WEIGHTS["news_blog"])
        return authority, recency if recency is not None else UNDATED_MULTIPLIER, recency is None

    authority, recency, undated = max((evidence(s) for s in sources), key=lambda e: e[0] * e[1])
    domains = len({host_of(s.url) for s in sources})
    corroboration = CORROBORATION.get(domains, CORROBORATION_MAX)
    return Confidence(
        score=round(min(authority * recency * corroboration, 1.0), 2),
        authority=authority,
        recency=recency,
        corroboration=corroboration,
        undated=undated,
    )
