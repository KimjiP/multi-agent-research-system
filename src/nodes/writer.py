import json
import logging

from langchain_anthropic import ChatAnthropic

from src.config import MODEL_NAME, TEMPERATURE
from src.state import Claim, Contradiction, ResearchState, SourceDoc

logger = logging.getLogger(__name__)

llm = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=4096)

WRITER_SYSTEM_PROMPT = """You are a research report writer specializing in AI governance and regulatory compliance.

Your job is to produce a clear, well-structured research report from pre-analyzed claims and sources. Every factual statement MUST cite its source using markdown links.

## Report Structure

Use this exact structure:

# [Report Title derived from the research question]

## Executive Summary
2-3 sentence overview of key findings.

## Key Findings
Organize by theme/category (use the claim categories provided).
Every factual statement must cite its source: [Source Title](url)
Include confidence level for key claims: (high confidence), (medium confidence), or (low confidence)
- high confidence: >= 0.7
- medium confidence: 0.4 to 0.69
- low confidence: < 0.4

## Contradictions & Uncertainties
List any contradictions between sources.
Flag claims where confidence is below 0.4.

## Knowledge Gaps
What questions remain unanswered?
What would require further research?

## Sources
Numbered list of all sources cited, with URLs and publication dates.

## Methodology Note
"This report was produced by an automated research pipeline using web search, source quality assessment, and confidence-weighted claim synthesis."

## Rules
- Do NOT invent claims or sources that are not in the provided data.
- Every factual statement must link to a source from the provided list.
- Present high-confidence claims prominently, low-confidence claims with appropriate caveats.
- Be explicit about what is uncertain or contested."""


def _format_claims(claims: list[Claim]) -> str:
    parts = []
    for c in claims:
        parts.append(
            f"- [{c.claim_id}] ({c.category}, confidence={c.confidence:.2f}): {c.text}\n"
            f"  Sources: {', '.join(c.supporting_sources)}"
        )
    return "\n".join(parts)


def _format_contradictions(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "None detected."
    parts = []
    for ct in contradictions:
        parts.append(
            f"- {ct.description}\n"
            f"  Source A: {ct.source_a}\n"
            f"  Source B: {ct.source_b}"
        )
    return "\n".join(parts)


def _format_source_list(sources: list[SourceDoc]) -> str:
    parts = []
    for s in sources:
        parts.append(f"- [{s.title}]({s.url}) (type: {s.source_type}, published: {s.published_date or 'Unknown'})")
    return "\n".join(parts)


def writer_node(state: ResearchState) -> dict:
    """Produce a formatted research report from analyst output."""
    question = state["question"]
    claims = state["extracted_claims"]
    contradictions = state["contradictions"]
    open_issues = state["open_issues"]
    sources = state["gathered_sources"]
    review_feedback = state.get("review_feedback")
    existing_report = state.get("report", "")

    logger.info(f"Writer generating report from {len(claims)} claims")

    if review_feedback and existing_report:
        # Revision mode
        user_prompt = (
            f"Research Question: {question}\n\n"
            f"## Previous Report\n\n{existing_report}\n\n"
            f"## Reviewer Feedback\n\n{review_feedback}\n\n"
            f"## Available Claims\n\n{_format_claims(claims)}\n\n"
            f"## Available Sources\n\n{_format_source_list(sources)}\n\n"
            "Revise the report to address the reviewer's feedback. "
            "Keep what works, fix what was flagged."
        )
        logger.info("Writer in revision mode")
    else:
        # Initial generation
        user_prompt = (
            f"Research Question: {question}\n\n"
            f"## Extracted Claims\n\n{_format_claims(claims)}\n\n"
            f"## Contradictions\n\n{_format_contradictions(contradictions)}\n\n"
            f"## Open Issues\n\n" + "\n".join(f"- {issue}" for issue in open_issues) + "\n\n"
            f"## Available Sources\n\n{_format_source_list(sources)}\n\n"
            "Write the research report following the structure in your instructions."
        )

    response = llm.invoke(
        [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    report = response.content

    logger.info(f"Writer produced report: {len(report.split())} words")

    return {
        "report": report,
        "current_phase": "writing",
    }
