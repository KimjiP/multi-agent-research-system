import json
import logging
import os
from datetime import date

from pydantic import BaseModel
from tavily import TavilyClient

from src.config import (
    EFFORT,
    OFFICIAL_DOMAINS,
    QUERIES_PER_ITERATION,
    RESULTS_PER_QUERY,
    SNIPPET_CHARS,
)
from src.llm import LLMError, generate_structured
from src.sources import classify_source, published_date
from src.state import ResearchState, SourceDoc

logger = logging.getLogger(__name__)

_tavily: TavilyClient | None = None

QUERY_SYSTEM = (
    "You write web search queries for regulatory research. Today's date is {today}. "
    "Prefer queries that surface primary sources, such as legal texts and official "
    "guidance, and recent developments."
)


class SearchQueries(BaseModel):
    queries: list[str]


def _get_tavily() -> TavilyClient:
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily


def generate_queries(
    question: str,
    previous_queries: list[str],
    sources: list[SourceDoc],
) -> list[str]:
    """Use the LLM to write diverse search queries; follow-ups target gaps."""
    if not previous_queries:
        prompt = (
            f"Write {QUERIES_PER_ITERATION} diverse web search queries to research this "
            f"question:\n\n{question}\n\nVary them by specificity, angle and focus."
        )
    else:
        source_summary = "\n".join(f"- {s.title} ({s.source_type})" for s in sources[:10])
        prompt = (
            f"Write {QUERIES_PER_ITERATION} follow-up web search queries that fill gaps in "
            f"the research on:\n\n{question}\n\n"
            f"Queries already run: {json.dumps(previous_queries)}\n\n"
            f"Sources found so far:\n{source_summary}\n\n"
            "Target missing perspectives and primary sources."
        )
    system = QUERY_SYSTEM.format(today=date.today().isoformat())
    try:
        result = generate_structured(system, prompt, SearchQueries, effort=EFFORT["queries"])
    except LLMError as e:
        logger.warning(f"Query generation failed ({e}); searching for the question itself")
        return [question]
    queries = [q.strip() for q in result.queries if q.strip()][:QUERIES_PER_ITERATION]
    return queries or [question]


def search_sources(
    query: str,
    seen_urls: set[str],
    include_domains: list[str] | None = None,
) -> list[SourceDoc]:
    """Run one web search and return results not seen before, adding them to seen_urls."""
    try:
        results = _get_tavily().search(
            query=query,
            max_results=RESULTS_PER_QUERY,
            search_depth="advanced",
            include_domains=include_domains or [],
            include_raw_content=False,
            include_answer=False,
        )
    except Exception as e:
        logger.warning(f"Tavily search failed for query '{query}': {e}")
        return []

    sources: list[SourceDoc] = []
    for r in results.get("results", []):
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append(
            SourceDoc(
                url=url,
                title=r.get("title", ""),
                snippet=r.get("content", "")[:SNIPPET_CHARS],
                source_type=classify_source(url),
                published_date=published_date(r),
                search_query=query,
            )
        )
    return sources


def researcher_node(state: ResearchState) -> dict:
    """Tool-calling node: generate queries, search via Tavily, collect sources."""
    iteration = state.get("iteration_count", 0)
    previous_queries = state.get("search_queries", [])
    existing_sources = state.get("gathered_sources", [])
    seen_urls = {s.url for s in existing_sources}

    logger.info(f"Researcher iteration {iteration}: generating queries")
    queries = generate_queries(state["question"], previous_queries, existing_sources)
    # Official sources first, then the open web
    new_sources = [
        s for query in queries
        for s in search_sources(query, seen_urls, include_domains=OFFICIAL_DOMAINS)
    ]
    new_sources += [s for query in queries for s in search_sources(query, seen_urls)]
    all_sources = existing_sources + new_sources

    logger.info(
        f"Researcher iteration {iteration}: "
        f"found {len(new_sources)} new sources, {len(all_sources)} total"
    )

    return {
        "search_queries": previous_queries + queries,
        "gathered_sources": all_sources,
        "iteration_count": iteration + 1,
        "research_complete": False,
        "current_phase": "researching",
    }
