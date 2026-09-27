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

Usage:
    uv run python eval_reviewer.py                      # latest results/v2_*.json
    uv run python eval_reviewer.py results/v2_<run>.json
"""

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


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(RESULTS_DIR.glob("v2_*.json"))[-1]
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


if __name__ == "__main__":
    main()
