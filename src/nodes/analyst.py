import json
import logging

from langchain_anthropic import ChatAnthropic

from src.config import (
    MODEL_NAME,
    RECENCY_MULTIPLIERS,
    SOURCE_AUTHORITY_WEIGHTS,
    TEMPERATURE,
)
from src.state import Claim, Contradiction, ResearchState, SourceDoc

logger = logging.getLogger(__name__)

llm = ChatAnthropic(model=MODEL_NAME, temperature=TEMPERATURE)

ANALYST_SYSTEM_PROMPT = """You are a regulatory research analyst specializing in AI governance and compliance.

Given a research question and a set of sources, you must:
1. Extract factual claims from the sources
2. Assess confidence for each claim using the formula below
3. Detect contradictions between sources
4. Identify open issues and knowledge gaps

## Confidence Scoring

confidence = authority_weight × recency_multiplier × corroboration_factor

### Source Authority Weights:
{authority_weights}

### Recency Multipliers (based on source publication date):
{recency_multipliers}

### Corroboration Factor:
- 1.0 if single source supports the claim
- 1.2 if 2 sources agree
- 1.4 if 3+ sources agree (cap at 1.4)

Cap final confidence at 1.0.

## Rules:
- If a claim is about what the law actually requires, primary EU legal text (official_eu) should dominate. Secondary sources can explain but should not override.
- Be precise about what is established law vs. interpretation vs. proposed changes.

## DOMAIN-SPECIFIC ALERTS:
- US Executive Order 14110 on AI was REVOKED on January 20, 2025. Any source discussing it as active policy is STALE — flag this.
- The EU AI Act Digital Omnibus (proposed Nov 2025) may extend high-risk system deadlines — flag as PROPOSED, NOT ADOPTED.
- GPAI model obligations: applicable since August 2, 2025.
- High-risk AI system obligations: applicable from August 2, 2026 (original timeline).

## Output Format

Respond with ONLY valid JSON, no markdown code fences, no preamble:
{{
  "claims": [
    {{
      "claim_id": "claim_001",
      "text": "The factual claim in plain language",
      "supporting_sources": ["https://source-url.com"],
      "confidence": 0.85,
      "category": "legal_requirement | timeline | enforcement | interpretation"
    }}
  ],
  "contradictions": [
    {{
      "claim_a": "claim_id or description",
      "claim_b": "claim_id or description",
      "source_a": "https://source-a-url.com",
      "source_b": "https://source-b-url.com",
      "description": "What the contradiction is"
    }}
  ],
  "open_issues": [
    "Description of a gap or unresolved question"
  ]
}}"""


def _format_sources(sources: list[SourceDoc]) -> str:
    """Format sources for the analyst prompt."""
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(
            f"Source {i}:\n"
            f"  URL: {s.url}\n"
            f"  Title: {s.title}\n"
            f"  Type: {s.source_type}\n"
            f"  Published: {s.published_date or 'Unknown'}\n"
            f"  Content: {s.snippet}"
        )
    return "\n\n".join(parts)


def _parse_analyst_response(text: str) -> dict:
    """Parse the LLM's JSON response, handling common formatting issues."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def analyst_node(state: ResearchState) -> dict:
    """Extract claims, assess confidence, detect contradictions."""
    question = state["question"]
    sources = state["gathered_sources"]

    logger.info(f"Analyst processing {len(sources)} sources")

    system_prompt = ANALYST_SYSTEM_PROMPT.format(
        authority_weights=json.dumps(SOURCE_AUTHORITY_WEIGHTS, indent=2),
        recency_multipliers=json.dumps(RECENCY_MULTIPLIERS, indent=2),
    )

    user_prompt = (
        f"Research Question: {question}\n\n"
        f"Sources ({len(sources)} total):\n\n"
        f"{_format_sources(sources)}"
    )

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    try:
        parsed = _parse_analyst_response(response.content)
    except json.JSONDecodeError:
        logger.error("Failed to parse analyst JSON response")
        return {
            "extracted_claims": [],
            "contradictions": [],
            "open_issues": ["ERROR: Analyst failed to produce valid JSON output"],
            "current_phase": "analyzing",
        }

    claims = [Claim(**c) for c in parsed.get("claims", [])]
    contradictions = [Contradiction(**c) for c in parsed.get("contradictions", [])]
    open_issues = parsed.get("open_issues", [])

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
