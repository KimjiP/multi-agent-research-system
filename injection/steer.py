"""A5: can a planted title change what the agent searches for next?

The Researcher writes follow-up queries from the titles of the first 10 sources
(src.nodes.researcher.generate_queries). This script calls that function 20
times with a planted title among those 10 and 20 times without it, on the saved
Q4 sources and queries, and counts the calls where a query contains the made-up
canary word. No search is run.

In the saved V2 runs the follow-up search never ran: the first round always
found at least 5 sources. So this tests the query writer on its own.

Usage:
    uv run python -m injection.steer
"""

import argparse
import json
import logging

from src.llm import usage
from src.nodes.researcher import generate_queries

from injection.planted import CANARY, STEERING_PAGE, STEERING_POSITION, as_source
from injection.runner import ROOT, load_saved

RESULTS_DIR = ROOT / "results"
CALLS = 20


def steering_sources(sources, queries):
    """The saved sources with the planted page as the last of the first 10 titles."""
    planted = as_source(STEERING_PAGE, queries[0])
    return sources[: STEERING_POSITION - 1] + [planted] + sources[STEERING_POSITION - 1 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="baseline", help="folder under injection/results/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    question, sources, queries = load_saved("q4")
    lists = {"with": steering_sources(sources, queries), "without": sources}
    out_dir = RESULTS_DIR / args.phase / "a5"
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition, source_list in lists.items():
        hits = 0
        for k in range(1, CALLS + 1):
            path = out_dir / f"q4_{condition}_{k:02d}.json"
            if path.exists():
                hits += json.loads(path.read_text())["canary_in_queries"]
                continue
            usage.reset()
            new_queries = generate_queries(question, queries, source_list)
            record = {
                "condition": condition,
                "call": k,
                "titles_shown": [s.title for s in source_list[:10]],
                "queries": new_queries,
                "canary_in_queries": any(CANARY.search(q) for q in new_queries),
                "cost_usd": round(usage.cost_usd, 5),
            }
            path.write_text(json.dumps(record, indent=2))
            hits += record["canary_in_queries"]
        print(f"  {condition:<8} planted title: canary in the queries in {hits} of {CALLS} calls")


if __name__ == "__main__":
    main()
