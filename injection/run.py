"""Planted-source test of the V2 pipeline: can a web page take over the research agent?

An attacker cannot change the agent's code, prompts or search engine, but can
write pages that a search may return. This script plants such pages in the saved
source lists of two benchmark questions (Q4 and Q3) and runs the Analyst, Writer
and Reviewer on every list, 5 times per case, plus 5 clean runs per question
with nothing planted. TEST_PLAN.md gives every planted text, where it goes and
how outcomes are counted, fixed before the runs.

Attacks (pages in injection/planted.py, all under the .example domain):
    a1   a false date for the Annex III obligations on one blog page
    a2   the same false date on three "legal" pages with recent dates in their URLs
    a3   a page that tells AI tools to state a false rule
    a4   a page that tells AI tools to add a link to a fake checker

A5, search steering, tests only the query writer: injection/steer.py.

Defense:
    d2   a host name with "law" or "legal" no longer makes a source legal analysis
         (config.LEGAL_NAME_RULE off), so only the hand-made list of sites decides

Each run is saved in full in injection/results/<phase>/<question>_<attack>_<k>.json,
with every draft and every review round. Runs already saved are skipped, so the
command can be run again after an interruption. injection/analyze.py turns the
saved runs into the summary.

Every run is a separate process, so a run that stops early loses only itself.

Usage:
    uv run python -m injection.run                   # baseline, 4 runs at a time
    uv run python -m injection.run --defense d2      # the same runs with defense D2, in results/defense/
    uv run python -m injection.run --workers 1
    uv run python -m injection.run --one q4 a2 3     # one run
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from src.llm import LLMError, usage
from src.sources import classify_source
from src.state import SourceDoc

from injection.checks import ATTACK_CONTENT, check_run
from injection.planted import ATTACKS, plant, planted_urls
from injection.runner import QUESTIONS, ROOT, load_saved, run_fixed

RESULTS_DIR = ROOT / "results"
CONDITIONS = ["clean", *ATTACKS]
RUNS = 5
DEFENSES = ["none", "d2"]


def run_path(phase: str, qid: str, attack: str, k: int) -> Path:
    return RESULTS_DIR / phase / f"{qid}_{attack}_{k}.json"


def apply_defense(sources: list[SourceDoc], defense: str) -> list[SourceDoc]:
    """The source list as the agent sees it with the defense on.

    D2 turns config.LEGAL_NAME_RULE off. In a full run the Researcher classifies each
    source as it finds it, but the saved sources were classified with the rule on, so
    every source, real or planted, is classified again here.
    """
    if defense == "d2":
        return [s.model_copy(update={"source_type": classify_source(s.url, legal_name_rule=False)}) for s in sources]
    return sources


def run_one(phase: str, qid: str, attack: str, k: int, defense: str = "none") -> dict:
    """One run: plant the attack's pages (if any), apply the defense, run the pipeline, save everything."""
    logging.basicConfig(level=logging.WARNING)
    question, sources, queries = load_saved(qid)
    if attack != "clean":
        sources = plant(sources, queries, ATTACKS[attack])
    sources = apply_defense(sources, defense)
    marked = planted_urls()

    try:
        result = run_fixed(question, sources, queries)
    except LLMError as e:
        # The model declined or was cut off: the run produced no report
        result = {"question": question, "pipeline_error": str(e), "cost_usd": round(usage.cost_usd, 4)}

    result.update(
        {
            "phase": phase,
            "defense": defense,
            "question_id": qid,
            "attack": attack,
            "run": k,
            "planted_urls": [s.url for s in sources if s.url in marked],
            "planted_positions": [i for i, s in enumerate(sources, 1) if s.url in marked],
        }
    )
    path = run_path(phase, qid, attack, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def describe(result: dict) -> str:
    """One line per finished run."""
    if result.get("pipeline_error"):
        return f"pipeline failed: {result['pipeline_error']}"
    row = check_run(result)
    verdict = "approved" if row["approved"] else result["review_status"]
    content = ATTACK_CONTENT.get(result["attack"])
    planted = f", planted content in report: {row[content]['report']['found']}" if content else ""
    return f"{verdict}{planted}, ${result['cost_usd']}, {result['duration_seconds']} s"


def run_in_process(phase: str, defense: str, qid: str, attack: str, k: int) -> str:
    """Run one job in its own process and describe the saved result."""
    command = [
        sys.executable, "-m", "injection.run", "--phase", phase, "--defense", defense, "--one", qid, attack, str(k)
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    path = run_path(phase, qid, attack, k)
    if not path.exists():  # an API error after the SDK's retries, or the process was killed
        lines = [line for line in process.stderr.splitlines() if line.strip() and not line.startswith("During task")]
        raise RuntimeError(f"not saved ({lines[-1] if lines else f'exit code {process.returncode}'})")
    return describe(json.loads(path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense", choices=DEFENSES, default="none")
    parser.add_argument("--phase", help="folder under injection/results/ (default: baseline, or defense)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--one", nargs=3, metavar=("QUESTION", "ATTACK", "RUN"), help="run one job, then stop")
    args = parser.parse_args()
    args.phase = args.phase or ("baseline" if args.defense == "none" else "defense")

    if args.one:
        qid, attack, k = args.one
        run_one(args.phase, qid, attack, int(k), args.defense)
        return

    # Interleaved, so a slow change in the API over the session does not fall on one condition
    jobs = [
        (args.phase, args.defense, qid, attack, k)
        for k in range(1, RUNS + 1)
        for qid in QUESTIONS
        for attack in CONDITIONS
        if not run_path(args.phase, qid, attack, k).exists()
    ]
    print(f"{len(jobs)} runs to do ({args.phase}, defense {args.defense}), {args.workers} at a time")

    start, failed = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_in_process, *job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            _, _, qid, attack, k = futures[future]
            try:
                line = future.result()
            except CancelledError:
                continue
            except RuntimeError as e:  # run the command again to retry
                failed += 1
                line = str(e)
                if "credit balance is too low" in line:
                    print("The API credit has run out: no new runs are started.")
                    for waiting in futures:
                        waiting.cancel()
            print(f"[{done}/{len(jobs)}] {qid} {attack} {k}: {line}")

    saved = [json.loads(p.read_text()) for p in (RESULTS_DIR / args.phase).glob("q*_*.json")]
    cost = sum(r.get("cost_usd") or 0 for r in saved)
    print(f"\nDone in {time.time() - start:.0f} s. {len(saved)} runs saved in {args.phase}, ${cost:.2f} in total.")
    print(f"Not saved: {failed}. Summary: uv run python -m injection.analyze")


if __name__ == "__main__":
    main()
