import json
import logging
import os
from urllib.parse import urlparse

from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from src.config import (
    DOMAIN_TO_SOURCE_TYPE,
    MODEL_NAME,
    QUERIES_PER_ITERATION,
    TEMPERATURE,
)
from src.state import ResearchState, SourceDoc

logger = logging.getLogger(__name__)

llm = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def _classify_source_type(url: str) -> str:
    """Classify source type based on URL domain."""
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    for known_domain, source_type in DOMAIN_TO_SOURCE_TYPE.items():
        if known_domain in domain:
            return source_type
    if domain.endswith(".gov") or domain.endswith(".europa.eu"):
        return "eu_guidance"
    if any(kw in domain for kw in ["law", "legal", "compliance"]):
        return "legal_analysis"
    return "news_blog"


def _generate_queries(
    question: str,
    iteration: int,
    previous_queries: list[str],
    sources: list[SourceDoc],
) -> list[str]:
    """Use LLM to generate diverse search queries."""
    if iteration == 0:
        prompt = (
            f"Generate exactly {QUERIES_PER_ITERATION} diverse web search queries "
            f"to research this question:\n\n\"{question}\"\n\n"
            "Vary the queries by specificity, angle, and focus. "
            "Return ONLY a JSON array of strings, no other text."
        )
    else:
        source_summary = "\n".join(
            f"- {s.title} ({s.source_type})" for s in sources[:10]
        )
        prompt = (
            f"Generate exactly {QUERIES_PER_ITERATION} follow-up search queries "
            f"to fill gaps in research on:\n\n\"{question}\"\n\n"
            f"Previous queries already tried: {json.dumps(previous_queries)}\n\n"
            f"Sources found so far:\n{source_summary}\n\n"
            "Generate NEW queries that target gaps or missing perspectives. "
            "Return ONLY a JSON array of strings, no other text."
        )

    response = llm.invoke(prompt)
    text = response.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [str(q) for q in queries[:QUERIES_PER_ITERATION]]
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM query response as JSON, using fallback")
    return [question]


def researcher_node(state: ResearchState) -> dict:
    """L3 tool-calling node: generate queries, search via Tavily, collect sources."""
    iteration = state.get("iteration_count", 0)
    previous_queries = state.get("search_queries", [])
    existing_sources = state.get("gathered_sources", [])
    existing_urls = {s.url for s in existing_sources}

    logger.info(f"Researcher iteration {iteration}: generating queries")

    queries = _generate_queries(
        state["question"], iteration, previous_queries, existing_sources
    )
    all_queries = previous_queries + queries

    new_sources: list[SourceDoc] = []
    for query in queries:
        try:
            results = tavily_client.search(
                query=query,
                max_results=5,
                search_depth="advanced",
                include_raw_content=False,
                include_answer=False,
            )
            for r in results.get("results", []):
                url = r.get("url", "")
                if url in existing_urls:
                    continue
                existing_urls.add(url)
                new_sources.append(
                    SourceDoc(
                        url=url,
                        title=r.get("title", ""),
                        snippet=r.get("content", "")[:500],
                        source_type=_classify_source_type(url),
                        published_date=r.get("published_date"),
                        search_query=query,
                    )
                )
        except Exception as e:
            logger.warning(f"Tavily search failed for query '{query}': {e}")

    all_sources = existing_sources + new_sources

    logger.info(
        f"Researcher iteration {iteration}: "
        f"found {len(new_sources)} new sources, {len(all_sources)} total"
    )

    return {
        "search_queries": all_queries,
        "gathered_sources": all_sources,
        "iteration_count": iteration + 1,
        "research_complete": False,
        "current_phase": "researching",
    }
