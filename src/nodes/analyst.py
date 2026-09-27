import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel

from src.confidence import score_claim
from src.config import EFFORT
from src.llm import generate_structured
from src.state import Claim, Contradiction, ResearchState, SourceDoc

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a regulatory research analyst specializing in AI governance and compliance. Today's date is {today}.

Given a research question and numbered sources, you:
1. Extract the factual claims the sources make that are relevant to the question.
2. Detect contradictions between sources.
3. List open issues: what the sources leave unanswered.

Rules:
- Support every claim with the numbers of the sources that state it. Do not state anything no source supports.
- Keep what a law requires separate from interpretation, and adopted law separate from proposals.
- When sources disagree about whether something was adopted, or about a date, record a contradiction and say in the claim which source says what.
- Use the publication dates, where given, to tell current information from outdated information.
- Do not score confidence. It is computed from the sources afterwards."""


class ExtractedClaim(BaseModel):
    text: str
    source_numbers: list[int]
    category: Literal["legal_requirement", "timeline", "enforcement", "interpretation"]


class ExtractedContradiction(BaseModel):
    claim_a: str
    claim_b: str
    source_a: int
    source_b: int
    description: str


class Analysis(BaseModel):
    claims: list[ExtractedClaim]
    contradictions: list[ExtractedContradiction]
    open_issues: list[str]


def _format_sources(sources: list[SourceDoc]) -> str:
    """Number the sources for the prompt; the Analyst cites them by number."""
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(
            f"[{i}] {s.title}\n"
            f"    URL: {s.url}\n"
            f"    Type: {s.source_type} | Published: {s.published_date or 'no date'}\n"
            f"    Content: {s.snippet}"
        )
    return "\n\n".join(parts)


def build_findings(
    analysis: Analysis,
    sources: list[SourceDoc],
    today: date,
) -> tuple[list[Claim], list[Contradiction], list[str]]:
    """Map source numbers to sources, drop claims without a valid source, score confidence."""

    def source(number: int) -> SourceDoc | None:
        return sources[number - 1] if 1 <= number <= len(sources) else None

    claims: list[Claim] = []
    dropped = 0
    for extracted in analysis.claims:
        supporting = list({s.url: s for n in extracted.source_numbers if (s := source(n))}.values())
        if not supporting:
            dropped += 1
            continue
        claims.append(
            Claim(
                claim_id=f"claim_{len(claims) + 1:03d}",
                text=extracted.text,
                supporting_sources=[s.url for s in supporting],
                category=extracted.category,
                confidence=score_claim(supporting, today),
            )
        )
    if dropped:
        logger.warning(f"Dropped {dropped} claims that cited no valid source number")

    contradictions = [
        Contradiction(
            claim_a=c.claim_a,
            claim_b=c.claim_b,
            source_a=s_a.url if (s_a := source(c.source_a)) else "",
            source_b=s_b.url if (s_b := source(c.source_b)) else "",
            description=c.description,
        )
        for c in analysis.contradictions
    ]
    return claims, contradictions, analysis.open_issues


def analyze(
    question: str,
    sources: list[SourceDoc],
    today: date | None = None,
) -> tuple[list[Claim], list[Contradiction], list[str]]:
    """Extract claims, contradictions and open issues from the sources."""
    today = today or date.today()
    user_prompt = (
        f"Research Question: {question}\n\n"
        f"Sources ({len(sources)} total):\n\n"
        f"{_format_sources(sources)}"
    )
    analysis = generate_structured(
        ANALYST_SYSTEM_PROMPT.format(today=today.isoformat()),
        user_prompt,
        Analysis,
        effort=EFFORT["analyst"],
    )
    return build_findings(analysis, sources, today)


def analyst_node(state: ResearchState) -> dict:
    """Extract claims, compute their confidence, detect contradictions."""
    sources = state["gathered_sources"]
    logger.info(f"Analyst processing {len(sources)} sources")

    claims, contradictions, open_issues = analyze(state["question"], sources)

    logger.info(
        f"Analyst extracted {len(claims)} claims, "
        f"found {len(contradictions)} contradictions, "
        f"{len(open_issues)} open issues"
    )
    return {
        "extracted_claims": claims,
        "contradictions": contradictions,
        "open_issues": open_issues,
        "current_phase": "analyzing",
    }
