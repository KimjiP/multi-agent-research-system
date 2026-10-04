"""Build the claim tables for the calibration analysis (see ANALYSIS_PLAN.md).

    uv run python -m calibration.extract

1. claims.csv: one row per scored claim from the V2 and V3 benchmark runs, with the
   confidence score and its parts, and the truth label once it has been checked.
2. labels.csv: the blind copy the labels are written in. It has only an ID, the
   question, the category and the claim text, so the score cannot steer the label.
   It is written once and never overwritten. Later runs copy its labels into claims.csv.
"""

import csv
import json
import random
from pathlib import Path

from src.confidence import confidence_label
from src.config import SOURCE_AUTHORITY_WEIGHTS
from src.sources import host_of

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    "V2": ROOT / "results" / "v2_20260927_235716.json",
    "V3": ROOT / "results" / "v3_20260927_235419.json",
}
CLAIMS_CSV = ROOT / "calibration" / "claims.csv"
LABELS_CSV = ROOT / "calibration" / "labels.csv"
SHUFFLE_SEED = 20261004

TRUTH_LABELS = {"correct", "incorrect", "unverifiable"}
# Filled from labels.csv; empty until a label has been checked. difficulty marks the judgment calls.
LABEL_COLUMNS = ["label", "partly_correct", "difficulty", "evidence_url", "evidence_quote", "notes"]
CLAIM_COLUMNS = [
    "key", "blind_id", "run", "question", "claim_id", "category", "text",
    "score", "confidence_label", "authority", "recency", "corroboration", "undated",
    "strongest_source_type", "n_sources", "n_domains", "supporting_urls", "supporting_types",
    *LABEL_COLUMNS,
]
BLIND_COLUMNS = [
    "blind_id", "question", "category", "text",
    "draft_label", "draft_partly_correct", "difficulty", "evidence_url", "evidence_quote", "notes",
    "label", "partly_correct", "review_note",
]
# Each authority weight belongs to one source type, so the stored authority names the
# type of the strongest source (the one that set authority and recency)
TYPE_OF_AUTHORITY = {weight: source_type for source_type, weight in SOURCE_AUTHORITY_WEIGHTS.items()}


def load_claims() -> list[dict]:
    """Every scored claim in the two runs, in file order."""
    rows = []
    for run, path in RUNS.items():
        for question, record in enumerate(json.loads(path.read_text()), 1):
            sources = {s["url"]: s for s in record["gathered_sources"]}
            for claim in record["extracted_claims"]:
                conf = claim["confidence"]
                urls = claim["supporting_sources"]
                rows.append({
                    "key": f"{run}-Q{question}-{claim['claim_id']}",
                    "run": run,
                    "question": question,
                    "claim_id": claim["claim_id"],
                    "category": claim["category"],
                    "text": claim["text"],
                    "score": conf["score"],
                    "confidence_label": confidence_label(conf["score"]),
                    "authority": conf["authority"],
                    "recency": conf["recency"],
                    "corroboration": conf["corroboration"],
                    "undated": conf["undated"],
                    "strongest_source_type": TYPE_OF_AUTHORITY[conf["authority"]],
                    "n_sources": len(urls),
                    "n_domains": len({host_of(u) for u in urls}),
                    "supporting_urls": " | ".join(urls),
                    "supporting_types": " | ".join(sources[u]["source_type"] for u in urls),
                })
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), "claim keys are not unique"
    return rows


def assign_blind_ids(rows: list[dict]) -> None:
    """IDs C001, C002, ... in question order, shuffled within each question so the order hides the run."""
    rng = random.Random(SHUFFLE_SEED)
    number = 0
    for question in sorted({r["question"] for r in rows}):
        group = [r for r in rows if r["question"] == question]
        rng.shuffle(group)
        for row in group:
            number += 1
            row["blind_id"] = f"C{number:03d}"


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_labels() -> dict[str, dict]:
    with LABELS_CSV.open(newline="", encoding="utf-8") as f:
        labels = {row["blind_id"]: row for row in csv.DictReader(f)}
    bad = {row["label"] for row in labels.values()} - TRUTH_LABELS - {""}
    assert not bad, f"unknown labels in labels.csv: {bad}"
    return labels


def main() -> None:
    rows = load_claims()
    assign_blind_ids(rows)
    rows.sort(key=lambda r: r["blind_id"])

    if not LABELS_CSV.exists():
        write_csv(LABELS_CSV, BLIND_COLUMNS, rows)
        print(f"Wrote the blind copy {LABELS_CSV.name} ({len(rows)} claims)")
    labels = read_labels()
    assert set(labels) == {r["blind_id"] for r in rows}, "labels.csv does not match the claims"
    for row in rows:
        row.update({column: labels[row["blind_id"]][column] for column in LABEL_COLUMNS})

    rows.sort(key=lambda r: (r["run"], r["question"], r["claim_id"]))
    write_csv(CLAIMS_CSV, CLAIM_COLUMNS, rows)
    checked = sum(1 for r in rows if r["label"])
    print(f"Wrote {CLAIMS_CSV.name}: {len(rows)} claims, {checked} with a checked label")


if __name__ == "__main__":
    main()
