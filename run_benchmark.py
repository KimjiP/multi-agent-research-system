"""Run the benchmark questions through the V2 pipeline and save results."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.benchmarks import BENCHMARK_QUESTIONS, mentions_current_annex_iii_date
from src.config import MODEL_NAME
from src.graph import build_graph
from src.llm import usage
from src.state import initial_state

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def serialize_state(state: dict) -> dict:
    """Convert Pydantic models in state to dicts for JSON serialization."""
    return {
        "question": state["question"],
        "search_queries": state["search_queries"],
        "gathered_sources": [s.model_dump() for s in state["gathered_sources"]],
        "iteration_count": state["iteration_count"],
        "extracted_claims": [c.model_dump() for c in state["extracted_claims"]],
        "contradictions": [c.model_dump() for c in state["contradictions"]],
        "open_issues": state["open_issues"],
        "report": state["report"],
        "review_feedback": state["review_feedback"],
        "review_checklist": {k: v.model_dump() for k, v in state["review_checklist"].items()},
        "review_status": state["review_status"],
        "approved": state["approved"],
        "review_count": state["review_count"],
        "error": state.get("error"),
    }


def run_v2(question: str, graph=None) -> dict:
    """Run one question through the V2 graph and return the serialized result with run stats."""
    graph = graph or build_graph()
    usage.reset()
    start = time.time()
    final_state = graph.invoke(initial_state(question))
    result = serialize_state(final_state)
    result.update(
        {
            "model": MODEL_NAME,
            "duration_seconds": round(time.time() - start, 1),
            "llm_calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(usage.cost_usd, 4),
            "mentions_current_annex_iii_date": mentions_current_annex_iii_date(result["report"]),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return result


def run_benchmark():
    graph = build_graph()
    all_results = []

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n{'=' * 60}\nBENCHMARK Q{i}: {question[:80]}...\n{'=' * 60}\n")
        result = run_v2(question, graph)
        all_results.append(result)
        print(
            f"\n--- Q{i}: {result['duration_seconds']}s, ${result['cost_usd']}, "
            f"{len(result['gathered_sources'])} sources, {len(result['extracted_claims'])} claims, "
            f"{result['review_count']} review round(s), status {result['review_status']}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    run_benchmark()
