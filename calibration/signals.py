"""Two other uncertainty signals for each claim, from the agent's own model (see ANALYSIS_PLAN.md).

    uv run python -m calibration.signals --pilot   # 10 random claims, then a cost estimate
    uv run python -m calibration.signals           # every claim that is not done yet

For each claim, the model sees the claim and the sources it cites, in the format the
Analyst saw them (`_format_sources` in src/nodes/analyst.py):
1. Stated confidence: asked once for the probability, 0 to 100, that the claim is correct.
2. Agreement: asked REPEATS times, in separate calls, whether the sources fully support
   the claim. The signal is the share of yes answers.

Answers go to calibration/signals.jsonl, one line per claim, so a stopped run continues
where it left off. The prompts are fixed: changing them means deleting the file and
starting over.
"""

import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from calibration.extract import RUNS
from src.config import EFFORT, MODEL_NAME
from src.llm import LLMError, generate_structured, usage
from src.nodes.analyst import _format_sources
from src.state import SourceDoc

SIGNALS_JSONL = Path(__file__).resolve().parent / "signals.jsonl"
REPEATS = 5
PILOT_SIZE = 10
SEED = 20261004
TOTAL_CALLS_PER_CLAIM = 1 + REPEATS

SYSTEM_PROMPT = "You check claims from research reports on AI regulation. Today's date is 2026-09-27."
CONFIDENCE_PROMPT = (
    "Claim: {claim}\n\n"
    "Sources cited for the claim ({n} total):\n\n{sources}\n\n"
    "What is the probability, from 0 to 100, that the claim is correct?"
)
SUPPORT_PROMPT = (
    "Claim: {claim}\n\n"
    "Sources cited for the claim ({n} total):\n\n{sources}\n\n"
    "Do these sources fully support the claim? Answer yes or no."
)


class StatedConfidence(BaseModel):
    # The API cannot enforce a numeric range, so the range is checked after parsing
    probability: int = Field(description="Probability from 0 to 100 that the claim is correct")


class SupportCheck(BaseModel):
    fully_supported: bool = Field(description="True if the sources fully support the claim")


def load_claims() -> dict[str, tuple[str, list[SourceDoc]]]:
    """Claim text and cited sources for every claim, by the same key as claims.csv."""
    claims = {}
    for run, path in RUNS.items():
        for question, record in enumerate(json.loads(path.read_text()), 1):
            sources = {s["url"]: SourceDoc(**s) for s in record["gathered_sources"]}
            for claim in record["extracted_claims"]:
                key = f"{run}-Q{question}-{claim['claim_id']}"
                claims[key] = (claim["text"], [sources[u] for u in claim["supporting_sources"]])
    return claims


def check_claim(text: str, sources: list[SourceDoc]) -> dict:
    """One stated-confidence call and REPEATS support calls for a claim, with their token use."""
    fields = {"claim": text, "n": len(sources), "sources": _format_sources(sources)}
    start = (usage.calls, usage.input_tokens, usage.output_tokens)
    stated = generate_structured(SYSTEM_PROMPT, CONFIDENCE_PROMPT.format(**fields), StatedConfidence, EFFORT["analyst"])
    if not 0 <= stated.probability <= 100:
        raise LLMError(f"probability out of range: {stated.probability}")
    answers = [
        generate_structured(SYSTEM_PROMPT, SUPPORT_PROMPT.format(**fields), SupportCheck, EFFORT["analyst"]).fully_supported
        for _ in range(REPEATS)
    ]
    return {
        "stated_probability": stated.probability,
        "support_answers": answers,
        "support_share": sum(answers) / REPEATS,
        "calls": usage.calls - start[0],
        "input_tokens": usage.input_tokens - start[1],
        "output_tokens": usage.output_tokens - start[2],
    }


def done_keys() -> set[str]:
    if not SIGNALS_JSONL.exists():
        return set()
    lines = [json.loads(line) for line in SIGNALS_JSONL.read_text().splitlines() if line.strip()]
    return {line["key"] for line in lines if "error" not in line}


def run(keys: list[str], claims: dict) -> None:
    done = done_keys()
    todo = [k for k in keys if k not in done]
    print(f"{len(todo)} claims to check ({len(done)} already done), model {MODEL_NAME}, effort {EFFORT['analyst']}")
    usage.reset()
    with SIGNALS_JSONL.open("a") as out:
        for i, key in enumerate(todo, 1):
            text, sources = claims[key]
            try:
                line = {"key": key, "model": MODEL_NAME, **check_claim(text, sources)}
            except LLMError as e:
                line = {"key": key, "model": MODEL_NAME, "error": str(e)}
            out.write(json.dumps(line) + "\n")
            out.flush()
            print(f"  {i}/{len(todo)} {key}: {line.get('stated_probability', 'error')} "
                  f"{line.get('support_answers', line.get('error'))}")
    if usage.calls:
        per_call_in, per_call_out = usage.input_tokens / usage.calls, usage.output_tokens / usage.calls
        print(f"This run: {usage.calls} calls, {usage.input_tokens} input and {usage.output_tokens} output tokens, "
              f"${usage.cost_usd:.2f} (per call: {per_call_in:.0f} in, {per_call_out:.0f} out)")
        remaining = (len(claims) - len(done_keys())) * TOTAL_CALLS_PER_CLAIM
        print(f"Estimate for the remaining {remaining} calls: ${usage.cost_usd / usage.calls * remaining:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help=f"check {PILOT_SIZE} random claims only")
    args = parser.parse_args()
    claims = load_claims()
    keys = sorted(claims)
    if args.pilot:
        keys = sorted(random.Random(SEED).sample(keys, PILOT_SIZE))
    run(keys, claims)


if __name__ == "__main__":
    main()
