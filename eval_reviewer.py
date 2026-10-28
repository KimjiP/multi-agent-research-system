"""Planted-error test for the V2 Reviewer.

The Reviewer is the pipeline's quality gate, so it gets its own evaluation. This
script takes the reports from a saved V2 benchmark run, plants one error of a
known type in each copy, and runs the Reviewer on every copy and on the
unmodified report.

Error types:
    fabricated_link    a sentence citing a URL that is not among the gathered sources
    unsupported_claim  a plausible obligation, cited to a real source, that no claim supports
    changed_date       the first date in Key Findings moved by one year
    uncited_claim      an extracted claim restated without any citation

A planted error counts as caught when the Reviewer does not approve the copy.
The script also records which checks flagged it, and how often the Reviewer
rejects an unmodified report (false alarms).

With --gate the script is a test: it exits with an error when a planted error
gets through, or when the Reviewer rejects more clean reports than
GATE_MAX_FALSE_ALARMS. The planted-errors workflow in .github/workflows runs it
on GATE_FILE.

Usage:
    uv run python eval_reviewer.py                      # latest results/v2_*.json
    uv run python eval_reviewer.py results/v2_<run>.json
    uv run python eval_reviewer.py --gate               # the gate, on GATE_FILE
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.llm import usage
from src.nodes.reviewer import review_report
from src.state import Claim, Contradiction, SourceDoc

logging.basicConfig(level=logging.WARNING)

RESULTS_DIR = Path("results")
# The gate: at least GATE_MIN_CAUGHT of the 13 errors planted in GATE_FILE are caught, and
# at most GATE_MAX_FALSE_ALARMS of its 4 clean reports are rejected. Set on 28 October 2026
# after 3 local runs on GATE_FILE: 13, 13 and 12 of 13 caught, and 3 of 4 clean reports
# rejected each time. The Reviewer's verdicts vary between runs, so one miss is allowed,
# and only a Reviewer that rejects every clean report fails the second limit. Loosening the
# limits makes the gate weaker.
GATE_FILE = RESULTS_DIR / "v2_20260927_235716.json"
GATE_PLANTED = 13
GATE_MIN_CAUGHT = 12
GATE_MAX_FALSE_ALARMS = 3
FAKE_URL = "https://ec.europa.eu/commission/presscorner/detail/en/ip_26_4417"
_DATE = re.compile(
    r"\b(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|"
    r"November|December) (20\d{2})\b"
)


def _insert_in_key_findings(report: str, sentence: str) -> str | None:
    """Insert `sentence` as a bullet right under the Key Findings heading."""
    lines = report.split("\n")
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## key findings"):
            return "\n".join(lines[: i + 1] + ["", f"- {sentence}", ""] + lines[i + 1 :])
    return None


def _key_findings_span(report: str) -> tuple[int, int] | None:
    start = report.lower().find("## key findings")
    if start < 0:
        return None
    end = report.find("\n## ", start + 1)
    return start, end if end > 0 else len(report)


def plant_errors(report: str, claims: list[Claim], sources: list[SourceDoc]) -> dict[str, tuple[str, str]]:
    """Return {error_type: (corrupted report, description of the planted error)}."""
    planted: dict[str, tuple[str, str]] = {}

    sentence = (
        "The European Commission has confirmed that it will not propose any further changes "
        f"to these deadlines ([European Commission press release]({FAKE_URL})) (high confidence)."
    )
    if FAKE_URL not in {s.url for s in sources} and (r := _insert_in_key_findings(report, sentence)):
        planted["fabricated_link"] = (r, sentence)

    real = sources[0]
    sentence = (
        "Providers must also notify their national data protection authority at least 30 days "
        f"before placing a high-risk AI system on the market ([{real.title}]({real.url})) (high confidence)."
    )
    if r := _insert_in_key_findings(report, sentence):
        planted["unsupported_claim"] = (r, sentence)

    span = _key_findings_span(report)
    if span and (m := _DATE.search(report, *span)):
        shifted = f"{m[1]} {m[2]} {int(m[3]) - 1}"
        corrupted = report[: m.start()] + shifted + report[m.end():]
        planted["changed_date"] = (corrupted, f"'{m[0]}' changed to '{shifted}'")

    if claims:
        best = max(claims, key=lambda c: c.confidence.score)
        sentence = best.text.rstrip(".") + "."
        if r := _insert_in_key_findings(report, sentence):
            planted["uncited_claim"] = (r, sentence)

    return planted


def evaluate(result: dict) -> list[dict]:
    sources = [SourceDoc(**s) for s in result["gathered_sources"]]
    claims = [Claim(**c) for c in result["extracted_claims"]]
    contradictions = [Contradiction(**c) for c in result["contradictions"]]
    variants = {"clean": (result["report"], "unmodified report")}
    variants.update(plant_errors(result["report"], claims, sources))

    rows = []
    for error_type, (report, description) in variants.items():
        checklist, _ = review_report(
            result["question"], report, claims, contradictions, result["open_issues"], sources
        )
        failed = {name: check.notes for name, check in checklist.items() if not check.passed}
        rows.append(
            {
                "question": result["question"],
                "error_type": error_type,
                "planted": description,
                "approved": not failed,
                "failed_checks": failed,
            }
        )
        verdict = "approved" if not failed else f"rejected ({', '.join(failed)})"
        print(f"  {error_type:<18} {verdict}")
    return rows


def summarize(rows: list[dict]) -> dict:
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["error_type"]].append(row)
    summary = {}
    for error_type, group in by_type.items():
        caught = sum(1 for r in group if not r["approved"])
        flagged_by = defaultdict(int)
        for r in group:
            for name in r["failed_checks"]:
                flagged_by[name] += 1
        key = "false_alarms" if error_type == "clean" else "caught"
        summary[error_type] = {key: caught, "n": len(group), "flagged_by": dict(flagged_by)}
    return summary


def gate_failures(summary: dict) -> list[str]:
    """Why the gate fails, or an empty list when it passes."""
    planted = {t: s for t, s in summary.items() if t != "clean"}
    n_planted = sum(s["n"] for s in planted.values())
    caught = sum(s["caught"] for s in planted.values())
    false_alarms = summary["clean"]["false_alarms"]
    failures = []
    if n_planted != GATE_PLANTED:
        failures.append(f"{n_planted} errors were planted, expected {GATE_PLANTED}")
    if caught < GATE_MIN_CAUGHT:
        failures.append(f"only {caught} of {n_planted} planted errors were caught, the minimum is {GATE_MIN_CAUGHT}")
    if false_alarms > GATE_MAX_FALSE_ALARMS:
        failures.append(f"{false_alarms} clean reports were rejected, the limit is {GATE_MAX_FALSE_ALARMS}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_file", nargs="?", type=Path, help="default: the latest results/v2_*.json")
    parser.add_argument("--gate", action="store_true", help=f"exit with an error when the Reviewer misses the gate (on {GATE_FILE})")
    args = parser.parse_args()
    path = args.benchmark_file or (GATE_FILE if args.gate else sorted(RESULTS_DIR.glob("v2_*.json"))[-1])
    with open(path) as f:
        results = json.load(f)

    usage.reset()
    rows = []
    for i, result in enumerate(results, 1):
        print(f"Q{i}: {result['question'][:70]}...")
        rows.extend(evaluate(result))

    summary = summarize(rows)
    print(f"\n{'=' * 60}\nPlanted errors caught by the Reviewer ({path.name})\n{'=' * 60}")
    for error_type, s in summary.items():
        if error_type == "clean":
            print(f"  {'clean (false alarms)':<22} {s['false_alarms']}/{s['n']}  {s['flagged_by']}")
        else:
            print(f"  {error_type:<22} {s['caught']}/{s['n']}  {s['flagged_by']}")
    print(f"  Cost: ${usage.cost_usd:.3f} over {usage.calls} reviewer calls")

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"reviewer_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, "w") as f:
        json.dump(
            {"benchmark_file": path.name, "cost_usd": round(usage.cost_usd, 4), "summary": summary, "rows": rows},
            f,
            indent=2,
        )
    print(f"\nResults saved to {output}")

    if args.gate:
        if path.resolve() != GATE_FILE.resolve():
            sys.exit(f"Gate: the gate runs on {GATE_FILE}, not on {path}")
        if failures := gate_failures(summary):
            sys.exit("Gate failed: " + "; ".join(failures))
        print(
            f"Gate passed: at least {GATE_MIN_CAUGHT} of {GATE_PLANTED} planted errors caught, "
            f"at most {GATE_MAX_FALSE_ALARMS} of 4 clean reports rejected"
        )


if __name__ == "__main__":
    main()
