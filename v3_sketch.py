"""
Project A2 - Version 3 (L5 Sketch): Multi-Agent Orchestration Demo
====================================================================
Demonstrates dynamic researcher spawning and parallel execution.
NOT production-grade — built to show tradeoffs vs V2.

Key differences from V2:
- Supervisor decomposes the question into subtopics
- Multiple Researcher instances run in parallel (one per subtopic)
- Topic-aware Writer adapts report structure based on question type
- Higher cost/latency, but better coverage for complex questions
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient

from src.config import (
    DOMAIN_TO_SOURCE_TYPE,
    MODEL_NAME,
    SOURCE_AUTHORITY_WEIGHTS,
    RECENCY_MULTIPLIERS,
    TEMPERATURE,
)
from src.state import SourceDoc

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# =============================================================================
# Supervisor: Decompose question into subtopics
# =============================================================================

def supervisor_decompose(question: str) -> dict:
    """Decompose a research question into subtopics and classify the question type."""
    prompt = f"""You are a research supervisor. Given a research question, decompose it into 2-4 independent subtopics that can be researched in parallel.

Also classify the question type as one of: "compliance_analysis", "cross_jurisdictional_comparison", "technical_deep_dive", "policy_overview".

Research Question: "{question}"

Respond with ONLY valid JSON, no markdown code fences:
{{
  "question_type": "one of the types above",
  "subtopics": [
    {{
      "title": "Short subtopic title",
      "search_queries": ["query 1", "query 2"]
    }}
  ]
}}"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


# =============================================================================
# Parallel Researcher: One per subtopic
# =============================================================================

from urllib.parse import urlparse


def _classify_source_type(url: str) -> str:
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    for known_domain, source_type in DOMAIN_TO_SOURCE_TYPE.items():
        if known_domain in domain:
            return source_type
    if domain.endswith(".gov") or domain.endswith(".europa.eu"):
        return "eu_guidance"
    if any(kw in domain for kw in ["law", "legal", "compliance"]):
        return "legal_analysis"
    return "news_blog"


def research_subtopic(subtopic: dict) -> list[SourceDoc]:
    """Research a single subtopic using Tavily search."""
    title = subtopic["title"]
    queries = subtopic["search_queries"]
    sources: list[SourceDoc] = []
    seen_urls: set[str] = set()

    logger.info(f"  Researching subtopic: {title}")

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
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append(
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
            logger.warning(f"  Search failed for '{query}': {e}")

    logger.info(f"  Subtopic '{title}': found {len(sources)} sources")
    return sources


async def research_all_subtopics(subtopics: list[dict]) -> list[SourceDoc]:
    """Run all subtopic researchers in parallel."""
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, research_subtopic, subtopic)
        for subtopic in subtopics
    ]
    results = await asyncio.gather(*tasks)

    # Merge and deduplicate
    all_sources: list[SourceDoc] = []
    seen_urls: set[str] = set()
    for source_list in results:
        for s in source_list:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                all_sources.append(s)

    return all_sources


# =============================================================================
# Analyst (reused from V2 logic)
# =============================================================================

def analyze_sources(question: str, sources: list[SourceDoc]) -> dict:
    """Extract claims, contradictions, open issues from sources."""
    formatted_sources = "\n\n".join(
        f"Source {i}:\n  URL: {s.url}\n  Title: {s.title}\n  Type: {s.source_type}\n"
        f"  Published: {s.published_date or 'Unknown'}\n  Content: {s.snippet}"
        for i, s in enumerate(sources, 1)
    )

    prompt = f"""You are a regulatory research analyst. Extract factual claims, assess confidence, and identify contradictions.

Research Question: {question}

Sources ({len(sources)} total):

{formatted_sources}

Source Authority Weights: {json.dumps(SOURCE_AUTHORITY_WEIGHTS, indent=2)}
Recency Multipliers: {json.dumps(RECENCY_MULTIPLIERS, indent=2)}
Confidence Formula: authority × recency × corroboration (1.0 single, 1.2 two sources, 1.4 three+, cap at 1.0 final)

DOMAIN-SPECIFIC ALERTS:
- US Executive Order 14110 on AI was REVOKED on January 20, 2025. Flag as STALE.
- EU AI Act Digital Omnibus (proposed Nov 2025): PROPOSED, NOT ADOPTED.
- GPAI model obligations: applicable since August 2, 2025.
- High-risk AI obligations: applicable from August 2, 2026.

Respond with ONLY valid JSON:
{{
  "claims": [{{"claim_id": "claim_001", "text": "...", "supporting_sources": ["url"], "confidence": 0.85, "category": "legal_requirement|timeline|enforcement|interpretation"}}],
  "contradictions": [{{"claim_a": "...", "claim_b": "...", "source_a": "url", "source_b": "url", "description": "..."}}],
  "open_issues": ["..."]
}}"""

    response = llm.invoke(prompt)
    text = response.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


# =============================================================================
# Topic-Aware Writer
# =============================================================================

WRITER_PROMPTS = {
    "compliance_analysis": "Focus on actionable compliance steps, obligations, deadlines, and penalties. Structure findings by compliance requirement.",
    "cross_jurisdictional_comparison": "Structure as a side-by-side comparison. Highlight similarities, differences, and gaps between jurisdictions. Use comparison tables where appropriate.",
    "technical_deep_dive": "Focus on specific procedures, technical requirements, and implementation details. Include step-by-step processes where applicable.",
    "policy_overview": "Provide a broad overview of the policy landscape. Highlight key stakeholders, timelines, and emerging trends.",
}


def write_report(question: str, question_type: str, claims: list[dict], contradictions: list[dict], open_issues: list[str], sources: list[SourceDoc]) -> str:
    """Generate a report with topic-aware structure."""
    topic_guidance = WRITER_PROMPTS.get(question_type, WRITER_PROMPTS["policy_overview"])

    formatted_claims = "\n".join(
        f"- [{c['claim_id']}] ({c['category']}, confidence={c['confidence']:.2f}): {c['text']}\n"
        f"  Sources: {', '.join(c['supporting_sources'])}"
        for c in claims
    )

    formatted_sources = "\n".join(
        f"- [{s.title}]({s.url}) (type: {s.source_type}, published: {s.published_date or 'Unknown'})"
        for s in sources
    )

    prompt = f"""You are a research report writer. Produce a structured, cited report.

TOPIC-SPECIFIC GUIDANCE: {topic_guidance}

Research Question: {question}
Question Type: {question_type}

Extracted Claims:
{formatted_claims}

Contradictions:
{json.dumps(contradictions, indent=2) if contradictions else "None detected."}

Open Issues:
{chr(10).join(f"- {issue}" for issue in open_issues)}

Available Sources:
{formatted_sources}

Write a report with: Executive Summary, Key Findings (by theme, with [Source Title](url) citations and confidence levels), Contradictions & Uncertainties, Knowledge Gaps, Sources list, and Methodology Note.
Every factual statement must cite its source. Do not invent claims or sources."""

    response = llm.invoke(prompt)
    return response.content


# =============================================================================
# Main Pipeline
# =============================================================================

def run_v3_pipeline(question: str) -> dict:
    """Run the full V3 pipeline: Supervisor → Parallel Research → Analyst → Writer."""
    start_time = datetime.now()

    # 1. Supervisor decomposes
    logger.info("V3 Supervisor: decomposing question")
    decomposition = supervisor_decompose(question)
    question_type = decomposition["question_type"]
    subtopics = decomposition["subtopics"]
    logger.info(f"V3 Supervisor: type={question_type}, {len(subtopics)} subtopics")
    for st in subtopics:
        logger.info(f"  - {st['title']}: {len(st['search_queries'])} queries")

    # 2. Parallel research
    logger.info("V3 Researchers: running in parallel")
    all_sources = asyncio.run(research_all_subtopics(subtopics))
    logger.info(f"V3 Researchers: {len(all_sources)} total sources")

    # 3. Analyst
    logger.info("V3 Analyst: extracting claims")
    analysis = analyze_sources(question, all_sources)
    claims = analysis.get("claims", [])
    contradictions = analysis.get("contradictions", [])
    open_issues = analysis.get("open_issues", [])
    logger.info(f"V3 Analyst: {len(claims)} claims, {len(contradictions)} contradictions")

    # 4. Topic-aware writer
    logger.info(f"V3 Writer: generating report (type={question_type})")
    report = write_report(question, question_type, claims, contradictions, open_issues, all_sources)
    logger.info(f"V3 Writer: {len(report.split())} words")

    duration = (datetime.now() - start_time).total_seconds()

    return {
        "question": question,
        "question_type": question_type,
        "subtopics": subtopics,
        "search_queries": [q for st in subtopics for q in st["search_queries"]],
        "gathered_sources": [s.model_dump() for s in all_sources],
        "extracted_claims": claims,
        "contradictions": contradictions,
        "open_issues": open_issues,
        "report": report,
        "duration_seconds": duration,
        "timestamp": start_time.isoformat(),
    }


# =============================================================================
# Benchmark
# =============================================================================

BENCHMARK_QUESTIONS = [
    "What are the current compliance requirements for a company deploying a high-risk AI hiring tool under the EU AI Act, and what enforcement actions have been taken so far?",
    "How do the EU AI Act's requirements for foundation model providers compare to current US federal AI policy approaches?",
    "What conformity assessment procedures are required for high-risk AI systems under the EU AI Act, and which notified bodies have been designated so far?",
]

if __name__ == "__main__":
    all_results = []

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"V3 BENCHMARK Q{i}: {question[:80]}...")
        print(f"{'='*60}\n")

        result = run_v3_pipeline(question)
        all_results.append(result)

        print(f"\n--- Q{i} Complete ---")
        print(f"Type: {result['question_type']}")
        print(f"Subtopics: {len(result['subtopics'])}")
        print(f"Duration: {result['duration_seconds']:.1f}s")
        print(f"Sources: {len(result['gathered_sources'])}")
        print(f"Claims: {len(result['extracted_claims'])}")
        print(f"Report: {len(result['report'].split())} words")

    output_file = f"v3_sketch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")
