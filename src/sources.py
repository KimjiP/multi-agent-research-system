"""Deterministic source metadata: what kind of source a URL is, and when it was published."""

import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from src.config import DOMAIN_TO_SOURCE_TYPE

# A date in the URL path, e.g. /2026/05/28/ or /2025-11-19-omnibus
_URL_DATE = re.compile(r"/(20\d{2})[/-](0?[1-9]|1[0-2])(?:[/-](0?[1-9]|[12]\d|3[01]))?(?=[/-]|$)")


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def classify_source(url: str) -> str:
    """Source type from the URL's host, using config.DOMAIN_TO_SOURCE_TYPE."""
    host = host_of(url)
    for suffix, source_type in DOMAIN_TO_SOURCE_TYPE:
        if host == suffix or host.endswith("." + suffix):
            return source_type
    if re.search(r"law|legal", host):
        return "legal_analysis"
    return "news_blog"


def published_date(result: dict) -> str | None:
    """ISO publication date from a search result, or from a date in its URL path.

    Tavily only returns `published_date` for news searches, so most results
    have none. A date in the URL path (common on news and law-firm sites) is
    used when present; otherwise the source is undated.
    """
    parsed = _parse_date(result.get("published_date"))
    if parsed:
        return parsed.isoformat()
    m = _URL_DATE.search(urlparse(result.get("url", "")).path)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3] or 1)).isoformat()
        except ValueError:
            return None
    return None


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError):
        return None
