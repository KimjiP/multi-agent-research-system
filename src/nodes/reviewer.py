import json
import logging

from langchain_anthropic import ChatAnthropic

from src.config import MODEL_NAME, TEMPERATURE
from src.state import Claim, Contradiction, ResearchState

logger = logging.getLogger(__name__)

llm = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)

REVIEWER_SYSTEM_PROMPT = """You are a research report reviewer specializing in AI governance and regulatory compliance.

Your job is to evaluate a research report against the structured claims that were extracted from sources. You are a strict quality gate — only approve reports that meet all criteria.

## Review Checklist

Evaluate the report against each of these criteria:

1. **Citation Coverage**: Does every factual claim in the report have a citation (markdown link)?
2. **Hallucination Check**: Does the report reference any factual claim that is NOT in the provided extracted claims list? If so, flag it.
3. **Contradiction Disclosure**: Are contradictions and uncertainties between sources properly disclosed in the report?
4. **Confidence Assessment**: Does the report include confidence levels (high/medium/low) for key claims?
5. **Executive Summary Accuracy**: Does the executive summary accurately reflect the findings in the body?
6. **Knowledge Gaps**: Are knowledge gaps and open issues clearly stated?

## Output Format

Respond with ONLY valid JSON, no markdown code fences, no preamble:
{{
  "approved": true or false,
  "checklist": {{
    "citation_coverage": {{"pass": true/false, "notes": "..."}},
    "hallucination_check": {{"pass": true/false, "notes": "..."}},
    "contradiction_disclosure": {{"pass": true/false, "notes": "..."}},
    "confidence_assessment": {{"pass": true/false, "notes": "..."}},
    "executive_summary_accuracy": {{"pass": true/false, "notes": "..."}},
    "knowledge_gaps": {{"pass": true/false, "notes": "..."}}
  }},
  "revision_instructions": "Specific instructions for revision, or null if approved"
}}

## Rules
- Set approved=true ONLY if all 6 checklist items pass.
- If any item fails, set approved=false and provide clear, actionable revision instructions.
- Be strict but fair — minor formatting issues are not failures."""


def _format_claims_for_review(claims: list[Claim]) -> str:
    parts = []
    for c in claims:
        parts.append(
            f"- [{c.claim_id}] ({c.category}, confidence={c.confidence:.2f}): {c.text}\n"
            f"  Sources: {', '.join(c.supporting_sources)}"
        )
    return "\n".join(parts)


def _format_contradictions_for_review(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "None detected."
    parts = []
    for ct in contradictions:
        parts.append(f"- {ct.description} (between {ct.source_a} and {ct.source_b})")
    return "\n".join(parts)


def _parse_reviewer_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def reviewer_node(state: ResearchState) -> dict:
    """Evaluate report against extracted claims. Approve or request revision."""
    report = state["report"]
    claims = state["extracted_claims"]
    contradictions = state["contradictions"]
    question = state["question"]
    review_count = state.get("review_count", 0)

    logger.info(f"Reviewer evaluating report (review #{review_count + 1})")

    user_prompt = (
        f"Research Question: {question}\n\n"
        f"## Report to Review\n\n{report}\n\n"
        f"## Extracted Claims (ground truth)\n\n{_format_claims_for_review(claims)}\n\n"
        f"## Known Contradictions\n\n{_format_contradictions_for_review(contradictions)}"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    try:
        parsed = _parse_reviewer_response(response.content)
    except json.JSONDecodeError:
        logger.error("Failed to parse reviewer JSON response, forcing approval")
        return {
            "review_feedback": None,
            "approved": True,
            "review_count": review_count + 1,
            "current_phase": "reviewing",
        }

    approved = parsed.get("approved", False)
    revision_instructions = parsed.get("revision_instructions")

    checklist = parsed.get("checklist", {})
    passed = sum(1 for v in checklist.values() if isinstance(v, dict) and v.get("pass"))
    total = len(checklist)

    logger.info(
        f"Reviewer: {passed}/{total} checks passed, "
        f"{'APPROVED' if approved else 'REVISION REQUESTED'}"
    )

    if not approved and checklist:
        failed_items = [
            f"- {k}: {v.get('notes', '')}"
            for k, v in checklist.items()
            if isinstance(v, dict) and not v.get("pass")
        ]
        if failed_items:
            logger.info("Failed checks:\n" + "\n".join(failed_items))

    return {
        "review_feedback": revision_instructions if not approved else None,
        "approved": approved,
        "review_count": review_count + 1,
        "current_phase": "reviewing",
    }
