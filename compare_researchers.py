"""Compare the fixed Researcher with the Researcher with tools on the benchmark questions.

    uv run python compare_researchers.py             # the runs (API and Tavily); runs already saved are skipped
    uv run python compare_researchers.py --summary   # summary.json from the saved runs, no API calls

Each run is the full V2 pipeline (src/graph.py) with one of the two Researchers,
saved on its own in results/researcher_comparison/, so a stopped comparison goes
on where it stopped. For each run number and question, the fixed Researcher runs
first and the Researcher with tools right after it, so both search the web at
about the same time. Every number in a write-up comes from summary.json.
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from run_benchmark import serialize_state
from src.benchmarks import BENCHMARK_QUESTIONS, mentions_current_annex_iii_date
from src.config import EFFORT, MAX_TOOL_SEARCHES, MAX_TOOL_TURNS, MODEL_NAME, OFFICIAL_DOMAINS
from src.graph import build_graph
from src.llm import usage
from src.sources import host_of
from src.state import initial_state

OUT_DIR = Path("results") / "researcher_comparison"
RESEARCHERS = ["fixed", "tools"]
RUNS = 3
CREDITS_PER_SEARCH = 2  # Tavily charges 2 credits for an advanced search
Q4 = "q4"  # the question with the currency check


def on_official_site(url: str) -> bool:
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def run_once(question: str, researcher: str, graph) -> dict:
    """The full pipeline on one question; the Researcher's own cost and time are kept apart."""
    usage.reset()
    start = time.time()
    state = initial_state(question)
    research_cost = research_seconds = 0.0
    for update in graph.stream(state, stream_mode="updates"):
        for node, output in update.items():
            state.update(output)
            if node == "researcher":  # the fixed Researcher can run up to 3 rounds
                research_cost, research_seconds = usage.cost_usd, time.time() - start

    result = serialize_state(state)
    sources = result["gathered_sources"]
    tool_research = state.get("tool_research")
    result.update(
        {
            "researcher": researcher,
            "tool_research": tool_research,
            # The fixed Researcher runs every query twice: official sites, then the open web
            "searches": tool_research["searches"] if tool_research else 2 * len(result["search_queries"]),
            "official_sources": sum(on_official_site(s["url"]) for s in sources),
            "model": MODEL_NAME,
            "settings": {
                "max_tool_searches": MAX_TOOL_SEARCHES,
                "max_tool_turns": MAX_TOOL_TURNS,
                "researcher_effort": EFFORT["researcher"],
            },
            "duration_seconds": round(time.time() - start, 1),
            "research_seconds": round(research_seconds, 1),
            "llm_calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": round(usage.cost_usd, 4),
            "research_cost_usd": round(research_cost, 4),
            "mentions_current_annex_iii_date": mentions_current_annex_iii_date(result["report"]),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return result


def run_comparison(out_dir: Path, runs: int, questions: list[int], researchers: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs = {r: build_graph(r) for r in researchers}
    for k in range(1, runs + 1):
        for i in questions:
            for researcher in researchers:
                path = out_dir / f"{researcher}_q{i}_run{k}.json"
                if path.exists():
                    continue
                print(f"\n=== run {k}, Q{i}, {researcher} Researcher ===", flush=True)
                result = run_once(BENCHMARK_QUESTIONS[i - 1], researcher, graphs[researcher])
                path.write_text(json.dumps(result | {"question_id": f"q{i}", "run": k}, indent=2))
                print(
                    f"--- {len(result['gathered_sources'])} sources ({result['official_sources']} official), "
                    f"{result['searches']} searches, {result['review_status']}, "
                    f"${result['cost_usd']}, {result['duration_seconds']}s",
                    flush=True,
                )


def describe(runs: list[dict]) -> dict:
    """The measures for a group of runs: a list per run, and the mean or count."""
    shares = [r["official_sources"] / len(r["gathered_sources"]) if r["gathered_sources"] else 0.0 for r in runs]
    group = {
        "runs": len(runs),
        "sources": [len(r["gathered_sources"]) for r in runs],
        "official_share": [round(s, 4) for s in shares],
        "searches": [r["searches"] for r in runs],
        "approved": sum(r["approved"] for r in runs),
        "cost_usd": [r["cost_usd"] for r in runs],
        "research_cost_usd": [r["research_cost_usd"] for r in runs],
        "duration_seconds": [r["duration_seconds"] for r in runs],
        "research_seconds": [r["research_seconds"] for r in runs],
    }
    group["mean"] = {
        key: round(mean(group[key]), 4)
        for key in ("sources", "official_share", "searches", "cost_usd", "research_cost_usd", "duration_seconds", "research_seconds")
    }
    group["tavily_credits"] = sum(group["searches"]) * CREDITS_PER_SEARCH
    group["total_cost_usd"] = round(sum(group["cost_usd"]), 4)
    if runs[0]["question_id"] == Q4:
        group["gives_current_annex_iii_date"] = sum(r["mentions_current_annex_iii_date"] for r in runs)
    tool_runs = [r["tool_research"] for r in runs if r["tool_research"]]
    if tool_runs:
        group["ended_by"] = dict(Counter(t["ended_by"] for t in tool_runs))
        group["told_to_continue"] = sum(t["told_to_continue"] for t in tool_runs)
        group["turns"] = [t["turns"] for t in tool_runs]
        calls = [c for t in tool_runs for c in t["calls"] if c["tool"] != "finish_research"]
        refused = [c for c in calls if c["result"].startswith("Search limit reached")]
        group["searches_by_tool"] = dict(Counter(c["tool"] for c in calls if c not in refused))
        group["searches_refused_at_limit"] = len(refused)
    return group


def summarize(out_dir: Path) -> dict:
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_q*_run*.json"))]
    by_question = {}
    for qid in sorted({r["question_id"] for r in runs}):
        by_question[qid] = {
            researcher: describe(group)
            for researcher in RESEARCHERS
            if (group := [r for r in runs if r["question_id"] == qid and r["researcher"] == researcher])
        }
    overall = {
        researcher: describe(group)
        for researcher in RESEARCHERS
        if (group := [r for r in runs if r["researcher"] == researcher])
    }
    summary = {"runs": len(runs), "by_question": by_question, "overall": overall}
    pairs = [q for q in by_question.values() if len(q) == 2]
    if pairs:
        # In how many questions the Researcher with tools has the higher, or the lower, mean
        keys = ("sources", "official_share", "searches", "cost_usd", "duration_seconds")
        summary["questions_compared"] = len(pairs)
        summary["questions_where_tools_is_higher"] = {
            key: sum(q["tools"]["mean"][key] > q["fixed"]["mean"][key] for q in pairs) for key in keys
        }
        summary["questions_where_tools_is_lower"] = {
            key: sum(q["tools"]["mean"][key] < q["fixed"]["mean"][key] for q in pairs) for key in keys
        }
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n{summary['runs']} runs")
    print(f"  {'':<10}{'sources':>9}{'official':>10}{'searches':>10}{'approved':>10}{'cost':>9}{'time':>8}")
    for qid, groups in [*summary["by_question"].items(), ("all", summary["overall"])]:
        for researcher, g in groups.items():
            m = g["mean"]
            extra = f"   Q4 date given in {g['gives_current_annex_iii_date']}" if "gives_current_annex_iii_date" in g else ""
            print(
                f"  {qid} {researcher:<7}{m['sources']:>9.1f}{m['official_share']:>10.0%}{m['searches']:>10.1f}"
                f"{g['approved']:>6} of {g['runs']}{m['cost_usd']:>9.3f}{m['duration_seconds']:>7.0f}s{extra}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="only write summary.json from the saved runs")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--researchers", nargs="+", choices=RESEARCHERS, default=RESEARCHERS)
    args = parser.parse_args()

    if not args.summary:
        run_comparison(args.out, args.runs, args.questions, args.researchers)
    summary = summarize(args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"\nSaved {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
