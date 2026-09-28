"""
Version 3 (Sketch): Multi-Agent Orchestration Demo
===================================================
Demonstrates dynamic decomposition and parallel research.
NOT production-grade: built to show the trade-offs against V2.

Differences from V2:
- A supervisor splits the question into subtopics, each with its own search queries
- One researcher per subtopic, run in parallel
- Topic-aware writer: the report structure depends on the question type
- No review loop, on purpose, to show what the quality gate is worth
Shared with V2: web search, source classification, the Analyst and confidence scoring.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.benchmarks import BENCHMARK_QUESTIONS, mentions_current_annex_iii_date
from src.config import EFFORT, MODEL_NAME, OFFICIAL_DOMAINS
from src.llm import generate_structured, generate_text, usage
from src.nodes.analyst import analyze
from src.nodes.researcher import search_sources
from src.nodes.writer import (
    WRITER_SYSTEM_PROMPT,
    format_claims,
    format_contradictions,
    format_source_list,
)
from src.state import SourceDoc

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")

QuestionType = Literal[
    "compliance_analysis",
    "cross_jurisdictional_comparison",
    "technical_deep_dive",
    "policy_overview",
]

SUPERVISOR_SYSTEM_PROMPT = (
    "You are a research supervisor. Today's date is {today}. Split a research question into "
    "2-4 independent subtopics that can be researched in parallel, each with 2-3 web search "
    "queries, and classify the question type."
)

WRITER_PROMPTS = {
    "compliance_analysis": "Focus on actionable compliance steps, obligations, deadlines, and penalties. Structure findings by compliance requirement.",
    "cross_jurisdictional_comparison": "Structure as a side-by-side comparison. Highlight similarities, differences, and gaps between jurisdictions. Use comparison tables where appropriate.",
    "technical_deep_dive": "Focus on specific procedures, technical requirements, and implementation details. Include step-by-step processes where applicable.",
    "policy_overview": "Provide a broad overview of the policy landscape. Highlight key stakeholders, timelines, and emerging trends.",
}


class Subtopic(BaseModel):
    title: str
    search_queries: list[str]


class Decomposition(BaseModel):
    question_type: QuestionType
    subtopics: list[Subtopic]


# =============================================================================
# Supervisor: decompose the question into subtopics
# =============================================================================


def supervisor_decompose(question: str) -> Decomposition:
    return generate_structured(
        SUPERVISOR_SYSTEM_PROMPT.format(today=date.today().isoformat()),
        f'Research Question: "{question}"',
        Decomposition,
        effort=EFFORT["supervisor"],
    )


# =============================================================================
# Parallel researchers: one per subtopic
# =============================================================================


def research_subtopic(subtopic: Subtopic) -> list[SourceDoc]:
    """One researcher: the same searches as a V2 research round, for one subtopic."""
    seen_urls: set[str] = set()
    sources = [
        s for query in subtopic.search_queries
        for s in search_sources(query, seen_urls, include_domains=OFFICIAL_DOMAINS)
    ]
    sources += [s for query in subtopic.search_queries for s in search_sources(query, seen_urls)]
    logger.info(f"  Subtopic '{subtopic.title}': found {len(sources)} sources")
    return sources


def research_all_subtopics(subtopics: list[Subtopic]) -> list[SourceDoc]:
    """Run one researcher per subtopic in parallel, then merge and deduplicate by URL."""
    with ThreadPoolExecutor(max_workers=max(len(subtopics), 1)) as pool:
        per_subtopic = list(pool.map(research_subtopic, subtopics))
    merged: dict[str, SourceDoc] = {}
    for sources in per_subtopic:
        for s in sources:
            merged.setdefault(s.url, s)
    return list(merged.values())


# =============================================================================
# Topic-aware writer: the V2 writer plus structure guidance for the question type
# =============================================================================


def write_report(question, question_type, claims, contradictions, open_issues, sources) -> str:
    system = (
        f"{WRITER_SYSTEM_PROMPT}\n\n## Topic-specific guidance\n{WRITER_PROMPTS[question_type]}"
    )
    user_prompt = (
        f"Research Question: {question}\nQuestion Type: {question_type}\n\n"
        f"## Extracted Claims\n\n{format_claims(claims)}\n\n"
        f"## Contradictions\n\n{format_contradictions(contradictions)}\n\n"
        f"## Open Issues\n\n" + "\n".join(f"- {issue}" for issue in open_issues) + "\n\n"
        f"## Available Sources\n\n{format_source_list(sources)}\n\n"
        "Write the research report following the structure in your instructions."
    )
    return generate_text(system, user_prompt, effort=EFFORT["writer"])


# =============================================================================
# Main pipeline
# =============================================================================


def run_v3_pipeline(question: str) -> dict:
    """Run the full V3 pipeline: Supervisor → Parallel Research → Analyst → Writer."""
    usage.reset()
    start = time.time()

    logger.info("V3 Supervisor: decomposing question")
    decomposition = supervisor_decompose(question)
    logger.info(
        f"V3 Supervisor: type={decomposition.question_type}, "
        f"{len(decomposition.subtopics)} subtopics"
    )

    logger.info("V3 Researchers: running in parallel")
    sources = research_all_subtopics(decomposition.subtopics)
    logger.info(f"V3 Researchers: {len(sources)} total sources")

    logger.info("V3 Analyst: extracting claims")
    claims, contradictions, open_issues = analyze(question, sources)
    logger.info(f"V3 Analyst: {len(claims)} claims, {len(contradictions)} contradictions")

    logger.info(f"V3 Writer: generating report (type={decomposition.question_type})")
    report = write_report(
        question, decomposition.question_type, claims, contradictions, open_issues, sources
    )
    logger.info(f"V3 Writer: {len(report.split())} words")

    return {
        "question": question,
        "question_type": decomposition.question_type,
        "subtopics": [s.model_dump() for s in decomposition.subtopics],
        "search_queries": [q for s in decomposition.subtopics for q in s.search_queries],
        "gathered_sources": [s.model_dump() for s in sources],
        "extracted_claims": [c.model_dump() for c in claims],
        "contradictions": [c.model_dump() for c in contradictions],
        "open_issues": open_issues,
        "report": report,
        "model": MODEL_NAME,
        "duration_seconds": round(time.time() - start, 1),
        "llm_calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(usage.cost_usd, 4),
        "mentions_current_annex_iii_date": mentions_current_annex_iii_date(report),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    all_results = []

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n{'=' * 60}\nV3 BENCHMARK Q{i}: {question[:80]}...\n{'=' * 60}\n")
        result = run_v3_pipeline(question)
        all_results.append(result)
        print(
            f"\n--- Q{i}: type {result['question_type']}, {len(result['subtopics'])} subtopics, "
            f"{result['duration_seconds']}s, ${result['cost_usd']}, "
            f"{len(result['gathered_sources'])} sources, {len(result['extracted_claims'])} claims"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")
