import logging

from src.confidence import confidence_label
from src.config import EFFORT
from src.llm import generate_text
from src.state import Claim, Contradiction, ResearchState, SourceDoc

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a research report writer specializing in AI governance and regulatory compliance.

Your job is to produce a clear, well-structured research report from pre-analyzed claims and sources. Every factual statement MUST cite its source as a markdown link: [Source Title](url), using only the sources listed.

## Report Structure

Use this exact structure:

# [Report Title derived from the research question]

## Executive Summary
2-3 sentence overview of key findings.

## Key Findings
Organize by theme/category (use the claim categories provided).
Every factual statement must cite its source: [Source Title](url)
Give the confidence level of key claims exactly as labelled in the claims list: (high confidence), (medium confidence) or (low confidence). Do not merge claims with different labels into one sentence; give each its own sentence and label. Cite each statement to a source of the claim it comes from.

## Contradictions & Uncertainties
List any contradictions between sources.
Flag claims where confidence is below 0.4.

## Knowledge Gaps
What questions remain unanswered?
What would require further research?

## Sources
Numbered list of all sources cited, with URLs and publication dates.

## Methodology Note
"This report was produced by an automated research pipeline: web search, source classification, confidence scores computed from source authority, recency and corroboration, and an automated review against the extracted claims."

## Rules
- Keep the report between 500 and 800 words. Cite one to three sources per statement, the strongest ones, rather than every source available.
- Do NOT invent claims or sources that are not in the provided data.
- Every factual statement must link to a source from the provided list.
- Present high-confidence claims prominently, low-confidence claims with appropriate caveats.
- Be explicit about what is uncertain, contested, proposed rather than adopted, or outdated."""


def format_claims(claims: list[Claim]) -> str:
    parts = []
    for c in claims:
        conf = c.confidence
        parts.append(
            f"- [{c.claim_id}] ({c.category}, {confidence_label(conf.score)} confidence, "
            f"{conf.score:.2f} = authority {conf.authority} × recency {conf.recency}"
            f"{' (undated)' if conf.undated else ''} × corroboration {conf.corroboration}): {c.text}\n"
            f"  Sources: {', '.join(c.supporting_sources)}"
        )
    return "\n".join(parts)


def format_contradictions(contradictions: list[Contradiction]) -> str:
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


def format_source_list(sources: list[SourceDoc]) -> str:
    return "\n".join(
        f"- [{s.title}]({s.url}) (type: {s.source_type}, published: {s.published_date or 'no date'})"
        for s in sources
    )


def writer_node(state: ResearchState) -> dict:
    """Produce a formatted research report from analyst output."""
    question = state["question"]
    claims = state["extracted_claims"]
    review_feedback = state.get("review_feedback")
    existing_report = state.get("report", "")

    logger.info(f"Writer generating report from {len(claims)} claims")

    findings = (
        f"## Extracted Claims\n\n{format_claims(claims)}\n\n"
        f"## Contradictions\n\n{format_contradictions(state['contradictions'])}\n\n"
        f"## Open Issues\n\n" + "\n".join(f"- {issue}" for issue in state["open_issues"]) + "\n\n"
        f"## Available Sources\n\n{format_source_list(state['gathered_sources'])}"
    )

    if review_feedback and existing_report:
        user_prompt = (
            f"Research Question: {question}\n\n"
            f"## Previous Report\n\n{existing_report}\n\n"
            f"## Reviewer Feedback\n\n{review_feedback}\n\n"
            f"{findings}\n\n"
            "Revise the report to address the reviewer's feedback. "
            "Keep what works, fix what was flagged."
        )
        logger.info("Writer in revision mode")
    else:
        user_prompt = (
            f"Research Question: {question}\n\n{findings}\n\n"
            "Write the research report following the structure in your instructions."
        )

    report = generate_text(WRITER_SYSTEM_PROMPT, user_prompt, effort=EFFORT["writer"])
    logger.info(f"Writer produced report: {len(report.split())} words")

    return {
        "report": report,
        "current_phase": "writing",
    }
