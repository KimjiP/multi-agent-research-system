"""
Version 1 (Baseline): Single API Call Research Agent
=====================================================
Purpose: Establish a baseline showing what a single LLM call can and cannot do.
This output is the comparison point for the V2 pipeline and the V3 sketch.
"""

import json
import time
from datetime import date, datetime
from pathlib import Path

from src.benchmarks import BENCHMARK_QUESTIONS, mentions_current_annex_iii_date
from src.config import EFFORT, MODEL_NAME
from src.llm import generate_text, usage

RESULTS_DIR = Path("results")

SYSTEM_PROMPT = """You are a regulatory research analyst specializing in AI governance and compliance. Today's date is {today}.

Given a research question, produce a structured research report with:
1. An executive summary (2-3 sentences)
2. Key findings organized by theme
3. Specific citations for every factual claim (include source name, author if known, publication date)
4. A confidence assessment noting what you're certain about vs. uncertain about
5. Identified gaps - what information would require further research

Be precise about what you know vs. what you're inferring. If you cannot verify a specific fact, say so explicitly rather than guessing."""


def run_v1_baseline(question: str) -> dict:
    """Run a single API call and capture the output + metadata."""
    usage.reset()
    start = time.time()
    output_text = generate_text(
        SYSTEM_PROMPT.format(today=date.today().isoformat()),
        f"Research question: {question}",
        effort=EFFORT["baseline"],
    )
    return {
        "question": question,
        "response": output_text,
        "model": MODEL_NAME,
        "duration_seconds": round(time.time() - start, 1),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(usage.cost_usd, 4),
        "mentions_current_annex_iii_date": mentions_current_annex_iii_date(output_text),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    all_results = []
    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n{'=' * 60}\nV1 QUESTION {i}: {question[:80]}...\n{'=' * 60}\n")
        result = run_v1_baseline(question)
        all_results.append(result)
        print(result["response"])
        print(
            f"\n--- Duration {result['duration_seconds']}s, "
            f"tokens {result['input_tokens']} in / {result['output_tokens']} out, "
            f"${result['cost_usd']}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
