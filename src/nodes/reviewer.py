import logging
import re

from pydantic import BaseModel

from src.config import EFFORT, MAX_REVIEWS
from src.llm import LLMError, generate_structured
from src.nodes.writer import format_claims, format_contradictions, format_source_list
from src.state import CheckResult, Claim, Contradiction, ResearchState, SourceDoc

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a research report reviewer specializing in AI governance and regulatory compliance.

Your job is to evaluate a research report against the structured claims extracted from its sources, which are the ground truth. You are a strict quality gate.

## Review Checklist

1. citation_coverage: Does every factual statement in the report have a citation (markdown link)?
2. hallucination_check: Does every factual statement match an extracted claim? Flag any fact, number or date that the claims do not contain, including numbers or dates that differ from the claims.
3. contradiction_disclosure: Are contradictions and uncertainties between sources disclosed?
4. confidence_assessment: Do key claims carry confidence levels consistent with their scores?
5. executive_summary_accuracy: Does the executive summary accurately reflect the findings in the body?
6. knowledge_gaps: Are knowledge gaps and open issues clearly stated?

For each item, give passed and short notes that name the specific problem, quoting the sentence at fault. If any item fails, give clear, actionable revision instructions; otherwise leave them empty.
Be strict but fair: minor formatting issues are not failures."""

LLM_CHECKS = [
    "citation_coverage",
    "hallucination_check",
    "contradiction_disclosure",
    "confidence_assessment",
    "executive_summary_accuracy",
    "knowledge_gaps",
]

_MARKDOWN_LINK = re.compile(r"\]\((https?://[^)\s]+)\)")


class Review(BaseModel):
    citation_coverage: CheckResult
    hallucination_check: CheckResult
    contradiction_disclosure: CheckResult
    confidence_assessment: CheckResult
    executive_summary_accuracy: CheckResult
    knowledge_gaps: CheckResult
    revision_instructions: str


def check_source_links(report: str, sources: list[SourceDoc]) -> CheckResult:
    """Deterministic check: every link in the report points to a gathered source."""
    known = {s.url for s in sources}
    unknown = sorted({url for url in _MARKDOWN_LINK.findall(report) if url not in known})
    if unknown:
        return CheckResult(
            passed=False,
            notes=f"Links to URLs that are not among the gathered sources: {', '.join(unknown[:5])}",
        )
    return CheckResult(passed=True, notes="Every link points to a gathered source.")


def review_report(
    question: str,
    report: str,
    claims: list[Claim],
    contradictions: list[Contradiction],
    open_issues: list[str],
    sources: list[SourceDoc],
) -> tuple[dict[str, CheckResult], str]:
    """Run the link check and the LLM checklist. Returns the checklist and revision instructions.

    Raises LLMError if the LLM review cannot be completed.
    """
    checklist = {"source_links": check_source_links(report, sources)}

    user_prompt = (
        f"Research Question: {question}\n\n"
        f"## Report to Review\n\n{report}\n\n"
        f"## Extracted Claims (ground truth)\n\n{format_claims(claims)}\n\n"
        f"## Known Contradictions\n\n{format_contradictions(contradictions)}\n\n"
        f"## Open Issues\n\n" + "\n".join(f"- {issue}" for issue in open_issues) + "\n\n"
        f"## Sources\n\n{format_source_list(sources)}"
    )
    review = generate_structured(
        REVIEWER_SYSTEM_PROMPT, user_prompt, Review, effort=EFFORT["reviewer"]
    )
    for name in LLM_CHECKS:
        checklist[name] = getattr(review, name)

    instructions = review.revision_instructions.strip()
    if not checklist["source_links"].passed:
        instructions = (
            f"{instructions}\n" if instructions else ""
        ) + f"Remove or replace these links; cite only the listed sources. {checklist['source_links'].notes}"
    return checklist, instructions


def reviewer_node(state: ResearchState) -> dict:
    """Evaluate the report. Approve only if every check passes; fail closed if the review fails."""
    review_count = state.get("review_count", 0) + 1
    logger.info(f"Reviewer evaluating report (review #{review_count})")

    try:
        checklist, instructions = review_report(
            state["question"],
            state["report"],
            state["extracted_claims"],
            state["contradictions"],
            state["open_issues"],
            state["gathered_sources"],
        )
    except LLMError as e:
        logger.error(f"Review failed ({e}); report not approved")
        return {
            "review_feedback": None,
            "review_checklist": {},
            "review_status": "review_failed",
            "approved": False,
            "review_count": review_count,
            "current_phase": "reviewing",
            "error": f"Review failed: {e}",
        }

    approved = all(check.passed for check in checklist.values())
    if approved:
        status = "approved"
    elif review_count >= MAX_REVIEWS:
        status = "not_approved"
    else:
        status = "revision_requested"

    passed = sum(check.passed for check in checklist.values())
    logger.info(f"Reviewer: {passed}/{len(checklist)} checks passed, {status}")
    for name, check in checklist.items():
        if not check.passed:
            logger.info(f"  failed {name}: {check.notes}")

    return {
        "review_feedback": None if approved else instructions,
        "review_checklist": checklist,
        "review_status": status,
        "approved": approved,
        "review_count": review_count,
        "current_phase": "reviewing",
    }
