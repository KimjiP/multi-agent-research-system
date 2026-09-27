"""Run all 3 benchmark questions through the V2 pipeline and save results."""

import json
import logging
from datetime import datetime

from src.graph import build_graph
from src.state import SourceDoc, Claim, Contradiction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_QUESTIONS = [
    "What are the current compliance requirements for a company deploying a high-risk AI hiring tool under the EU AI Act, and what enforcement actions have been taken so far?",
    "How do the EU AI Act's requirements for foundation model providers compare to current US federal AI policy approaches?",
    "What conformity assessment procedures are required for high-risk AI systems under the EU AI Act, and which notified bodies have been designated so far?",
]


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
        "approved": state["approved"],
        "review_count": state["review_count"],
        "current_phase": state["current_phase"],
    }


def run_benchmark():
    graph = build_graph()
    all_results = []

    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"BENCHMARK Q{i}: {question[:80]}...")
        print(f"{'='*60}\n")

        start_time = datetime.now()

        initial_state = {
            "question": question,
            "search_queries": [],
            "gathered_sources": [],
            "iteration_count": 0,
            "research_complete": False,
            "extracted_claims": [],
            "contradictions": [],
            "open_issues": [],
            "report": "",
            "review_feedback": None,
            "approved": False,
            "review_count": 0,
            "current_phase": "",
            "error": None,
        }

        final_state = graph.invoke(initial_state)

        duration = (datetime.now() - start_time).total_seconds()

        result = serialize_state(final_state)
        result["duration_seconds"] = duration
        result["timestamp"] = start_time.isoformat()
        all_results.append(result)

        print(f"\n--- Q{i} Complete ---")
        print(f"Duration: {duration:.1f}s")
        print(f"Sources: {len(final_state['gathered_sources'])}")
        print(f"Claims: {len(final_state['extracted_claims'])}")
        print(f"Contradictions: {len(final_state['contradictions'])}")
        print(f"Report: {len(final_state['report'].split())} words")
        print(f"Review rounds: {final_state['review_count']}")
        print(f"Approved: {final_state['approved']}")

    output_file = f"v2_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")


if __name__ == "__main__":
    run_benchmark()
