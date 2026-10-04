"""Run the V2 pipeline from the Analyst on a fixed list of sources.

    uv run python -m injection.runner      # smoke test: one clean run each for Q4 and Q3

The Researcher is skipped: the sources come from a saved V2 run, with or without
planted sources (injection/planted.py). The Analyst, Writer and Reviewer are the
node functions of src/graph.py, joined by the same edges, so a report goes back
to the Writer by the same rule (src.edges.review_decision). Every run sees the
same real sources, so differences between runs come only from the planted
sources and the model's own variation.
"""

import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from src.benchmarks import BENCHMARK_QUESTIONS, mentions_current_annex_iii_date
from src.config import MODEL_NAME
from src.edges import review_decision
from src.llm import usage
from src.nodes.analyst import analyst_node
from src.nodes.reviewer import reviewer_node
from src.nodes.writer import writer_node
from src.state import ResearchState, SourceDoc, initial_state

ROOT = Path(__file__).resolve().parent
SAVED_RUN = ROOT.parent / "results" / "v2_20260927_235716.json"
# The two test questions, by their index in BENCHMARK_QUESTIONS and in the saved run
QUESTIONS = {"q4": 3, "q3": 2}


def load_saved(qid: str, path: Path = SAVED_RUN) -> tuple[str, list[SourceDoc], list[str]]:
    """The question, gathered sources and search queries of one question in the saved run."""
    result = json.loads(path.read_text())[QUESTIONS[qid]]
    if result["question"] != BENCHMARK_QUESTIONS[QUESTIONS[qid]]:
        raise ValueError(f"{path.name} does not hold {qid} at index {QUESTIONS[qid]}")
    return result["question"], [SourceDoc(**s) for s in result["gathered_sources"]], result["search_queries"]


def build_fixed_source_graph():
    """The V2 graph of src/graph.py without the Researcher."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("analyst")

    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        review_decision,
        {
            "revise": "writer",
            "end": END,
        },
    )

    return workflow.compile()


def serialize_state(state: dict) -> dict:
    """The final state as JSON, with the same keys as run_benchmark.py."""
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


def run_fixed(question: str, sources: list[SourceDoc], search_queries: list[str], graph=None) -> dict:
    """Run the Analyst, Writer and Reviewer on `sources`.

    Returns the final state, every draft of the report and every review round, so
    a run shows whether a check caught something that a revision then removed.
    """
    graph = graph or build_fixed_source_graph()
    state = initial_state(question)
    state.update(search_queries=list(search_queries), gathered_sources=list(sources), iteration_count=1)

    usage.reset()
    start = time.time()
    final, drafts, reviews = state, [], []
    for mode, chunk in graph.stream(state, stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
        elif "writer" in chunk:
            drafts.append(chunk["writer"]["report"])
        elif "reviewer" in chunk:
            update = chunk["reviewer"]
            reviews.append(
                {
                    "status": update["review_status"],
                    "approved": update["approved"],
                    "checklist": {k: v.model_dump() for k, v in update["review_checklist"].items()},
                    "feedback": update["review_feedback"],
                    "error": update.get("error"),
                }
            )

    result = serialize_state(final)
    result.update(
        {
            "drafts": drafts,
            "reviews": reviews,
            "model": MODEL_NAME,
            "run_date": date.today().isoformat(),
            "duration_seconds": round(time.time() - start, 1),
            "llm_calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(usage.cost_usd, 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return result


def main() -> None:
    """Smoke test: one clean run per question, next to the saved run it starts from."""
    logging.basicConfig(level=logging.WARNING)
    saved = json.loads(SAVED_RUN.read_text())
    out_dir = ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = build_fixed_source_graph()

    for qid, index in QUESTIONS.items():
        question, sources, queries = load_saved(qid)
        print(f"{qid}: {question[:70]}... ({len(sources)} saved sources)")
        result = run_fixed(question, sources, queries, graph)
        (out_dir / f"{qid}_clean.json").write_text(json.dumps(result, indent=2))
        for label, r in (("saved run", saved[index]), ("this run", result)):
            print(
                f"  {label:<10} {len(r['extracted_claims'])} claims, {len(r['report'].split())} words, "
                f"{r['review_count']} review round(s), {r['review_status']}, "
                f"current Annex III date: {mentions_current_annex_iii_date(r['report'])}, ${r.get('cost_usd')}"
            )
    print(f"\nSaved in {out_dir}")


if __name__ == "__main__":
    main()
