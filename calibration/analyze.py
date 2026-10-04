"""How well the confidence score predicts whether a claim is correct (see ANALYSIS_PLAN.md).

    uv run python -m calibration.analyze

Reads calibration/claims.csv and writes calibration/results.json. No API calls.
Accuracy is computed on the claims labeled correct or incorrect; unverifiable
claims are counted and left out. Every number in a write-up comes from results.json.
"""

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.config import PRICE_PER_MTOK

ROOT = Path(__file__).resolve().parent
CLAIMS_CSV = ROOT / "claims.csv"
AUDIT_CSV = ROOT / "audit.csv"
SIGNALS_JSONL = ROOT / "signals.jsonl"
RESULTS_JSON = ROOT / "results.json"

LABEL_EDGES = [0.0, 0.4, 0.7]  # the report's labels: low, medium, high
FINE_EDGES = [0.0, 0.15, 0.4, 0.7, 0.9]  # used only if every bin keeps MIN_BIN labeled claims
MIN_BIN = 15
MIN_GROUP = 15
CUTOFFS = [0.4, 0.7]
N_BOOT = 2000
SEED = 20261004


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a share of `correct` out of `n`."""
    if n == 0:
        raise ValueError("no claims")
    p = correct / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return centre - margin, centre + margin


def bin_of(score: float, edges: list[float]) -> int:
    """Index of the bin [edges[i], edges[i + 1]) that holds the score; the last bin runs to 1.0."""
    return max(i for i, edge in enumerate(edges) if score >= edge)


def reliability_bins(scores: list[float], outcomes: list[int], edges: list[float]) -> list[dict]:
    """Mean score against share correct in each score bin."""
    bins = []
    for i, low in enumerate(edges):
        members = [(s, y) for s, y in zip(scores, outcomes) if bin_of(s, edges) == i]
        n, correct = len(members), sum(y for _, y in members)
        bins.append({
            "low": low,
            "high": edges[i + 1] if i + 1 < len(edges) else 1.0,
            "claims": n,
            "correct": correct,
            "mean_score": mean(s for s, _ in members) if n else None,
            "share_correct": correct / n if n else None,
            "wilson_95": wilson_interval(correct, n) if n else None,
        })
    return bins


def expected_calibration_error(bins: list[dict]) -> float:
    """Gap between mean score and share correct, averaged over the bins and weighted by their size."""
    total = sum(b["claims"] for b in bins)
    return sum(b["claims"] / total * abs(b["mean_score"] - b["share_correct"]) for b in bins if b["claims"])


def brier_score(scores: list[float], outcomes: list[int]) -> float:
    """Mean squared difference between the score and the outcome (1 correct, 0 incorrect)."""
    return mean((s - y) ** 2 for s, y in zip(scores, outcomes))


def auroc(scores: list[float], outcomes: list[int]) -> float:
    """Chance that a random correct claim scores higher than a random incorrect one; ties count half."""
    correct = [s for s, y in zip(scores, outcomes) if y == 1]
    incorrect = [s for s, y in zip(scores, outcomes) if y == 0]
    if not correct or not incorrect:
        raise ValueError("AUROC needs both correct and incorrect claims")
    wins = sum((c > i) + 0.5 * (c == i) for c in correct for i in incorrect)
    return wins / (len(correct) * len(incorrect))


def bootstrap_interval(statistic, pairs: list[tuple[float, int]], n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of statistic(scores, outcomes) over resamples of the claims.

    Resamples where the statistic is undefined (for example no incorrect claim drawn) are skipped.
    """
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        try:
            values.append(statistic([s for s, _ in sample], [y for _, y in sample]))
        except ValueError:
            continue
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def decision_row(all_scores: list[float], pairs: list[tuple[float, int]], cutoff: float) -> dict:
    """Every claim with a score below `cutoff` goes to a person; the rest pass unchecked."""
    sent = sum(s < cutoff for s in all_scores)
    incorrect = [s for s, y in pairs if y == 0]
    passed = [(s, y) for s, y in pairs if s >= cutoff]
    incorrect_passing = sum(s >= cutoff for s in incorrect)
    return {
        "cutoff": cutoff,
        "sent_to_person": sent,
        "share_sent": sent / len(all_scores),
        "incorrect_passing": incorrect_passing,
        "incorrect_total": len(incorrect),
        "share_of_incorrect_passing": incorrect_passing / len(incorrect),
        "labeled_passing": len(passed),
        "share_incorrect_among_passing": incorrect_passing / len(passed) if passed else None,
    }


def incorrect_passing_at_workload(values: list[float], outcomes: list[int | None], n_sent: int) -> float:
    """Incorrect claims that pass when the `n_sent` claims with the lowest values go to a person.

    outcomes are 1 (correct), 0 (incorrect) or None (unverifiable). Claims tied at the
    boundary are sent in proportion, so the result is the expected count under a random
    choice among them.
    """
    by_value = defaultdict(list)
    for value, outcome in zip(values, outcomes):
        by_value[value].append(outcome)
    slots, sent_incorrect = n_sent, 0.0
    for value in sorted(by_value):
        group = by_value[value]
        share = min(1.0, slots / len(group)) if slots > 0 else 0.0
        sent_incorrect += share * sum(o == 0 for o in group)
        slots -= len(group)
    return sum(o == 0 for o in outcomes) - sent_incorrect


def workload_curve(values: list[float], outcomes: list[int | None]) -> list[list[float]]:
    """[share of claims sent to a person, share of incorrect claims caught] for every workload,
    when the claims with the lowest values are sent first (ties split in proportion)."""
    total, wrong = len(values), sum(o == 0 for o in outcomes)
    return [[n / total, (wrong - incorrect_passing_at_workload(values, outcomes, n)) / wrong] for n in range(total + 1)]


def paired_auroc_difference(a: list[float], b: list[float], outcomes: list[int], n_boot: int = N_BOOT,
                            seed: int = SEED) -> tuple[float, float, float]:
    """AUROC of b minus AUROC of a on the same claims, with a 95% interval from shared resamples."""
    rng = random.Random(seed)
    n, diffs = len(outcomes), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [outcomes[i] for i in idx]
        try:
            diffs.append(auroc([b[i] for i in idx], ys) - auroc([a[i] for i in idx], ys))
        except ValueError:
            continue
    diffs.sort()
    return auroc(b, outcomes) - auroc(a, outcomes), diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def fit_logistic(x: list[float], y: list[int], iterations: int = 100) -> tuple[float, float]:
    """Intercept and slope of P(correct) = sigmoid(a + b * x), by Newton's method (Platt scaling)."""
    a = b = 0.0
    for _ in range(iterations):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for xi, yi in zip(x, y):
            p = sigmoid(a + b * xi)
            w = p * (1 - p)
            g0, g1 = g0 + (yi - p), g1 + (yi - p) * xi
            h00, h01, h11 = h00 + w, h01 + w * xi, h11 + w * xi * xi
        det = h00 * h11 - h01 * h01
        if det <= 1e-12:
            raise ValueError("the scores do not vary enough to fit a slope")
        da, db = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def cross_validated_recalibration(scores: list[float], outcomes: list[int], folds: int = 5, seed: int = SEED) -> dict:
    """Platt scaling with k-fold cross-validation: every claim is predicted by a fit that never saw it.

    Folds are drawn separately from correct and incorrect claims, so each fold holds both.
    The constant guess (the training folds' share correct) is scored the same way for comparison.
    """
    rng = random.Random(seed)
    fold_of = {}
    for outcome in (1, 0):
        members = [i for i, y in enumerate(outcomes) if y == outcome]
        rng.shuffle(members)
        fold_of.update({i: k % folds for k, i in enumerate(members)})
    predicted, constant, fits = [0.0] * len(scores), [0.0] * len(scores), []
    for fold in range(folds):
        train = [i for i in range(len(scores)) if fold_of[i] != fold]
        a, b = fit_logistic([scores[i] for i in train], [outcomes[i] for i in train])
        fits.append({"intercept": a, "slope": b})
        base_rate = mean(outcomes[i] for i in train)
        for i in range(len(scores)):
            if fold_of[i] == fold:
                predicted[i], constant[i] = sigmoid(a + b * scores[i]), base_rate
    return {
        "folds": folds,
        "fits": fits,
        "held_out_prediction_range": [min(predicted), max(predicted)],
        "held_out_ece": expected_calibration_error(reliability_bins(predicted, outcomes, LABEL_EDGES)),
        "held_out_brier": brier_score(predicted, outcomes),
        "held_out_constant_guess_ece": expected_calibration_error(reliability_bins(constant, outcomes, LABEL_EDGES)),
        "held_out_constant_guess_brier": brier_score(constant, outcomes),
    }


def accuracy_summary(rows: list[dict]) -> dict:
    """Counts, share correct with its Wilson interval, and mean score of the labeled claims in `rows`."""
    labeled = [r for r in rows if r["label"] != "unverifiable"]
    correct = sum(r["label"] == "correct" for r in labeled)
    summary = {
        "claims": len(rows),
        "unverifiable": len(rows) - len(labeled),
        "labeled": len(labeled),
        "correct": correct,
        "incorrect": len(labeled) - correct,
        "partly_correct": sum(r["partly_correct"] == "yes" for r in labeled),
    }
    if labeled:
        summary.update(
            share_correct=correct / len(labeled),
            wilson_95=wilson_interval(correct, len(labeled)),
            mean_score=mean(r["score"] for r in labeled),
            score_range=[min(r["score"] for r in labeled), max(r["score"] for r in labeled)],
        )
    return summary


def breakdown(rows: list[dict], key: str) -> list[dict]:
    """Accuracy per group, analysed only for groups with at least MIN_GROUP labeled claims."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = []
    for name, members in sorted(groups.items(), key=lambda g: -len(g[1])):
        summary = accuracy_summary(members)
        if summary["labeled"] < MIN_GROUP:
            summary = {k: summary[k] for k in ("claims", "unverifiable", "labeled")}
            summary["note"] = f"fewer than {MIN_GROUP} labeled claims, not analysed"
        out.append({"group": name, **summary})
    return out


def without_judgment_calls(rows: list[dict]) -> dict:
    """Added after the first results, not in the analysis plan: the main numbers without the
    claims marked as judgment calls during labeling, to see whether they drive the result."""
    clear = [r for r in rows if r["difficulty"] != "judgment"]
    labeled = [r for r in clear if r["label"] != "unverifiable"]
    pairs = [(r["score"], int(r["label"] == "correct")) for r in labeled]
    scores, outcomes = [s for s, _ in pairs], [y for _, y in pairs]
    return {
        "note": "added after the first results; not in the analysis plan",
        "judgment_calls_left_out": len(rows) - len(clear),
        "overall": accuracy_summary(clear),
        "per_confidence_label": {
            name: accuracy_summary([r for r in clear if r["confidence_label"] == name])
            for name in ("high", "medium", "low")
        },
        "incorrect_scores": sorted((s for s, y in pairs if y == 0), reverse=True),
        "auroc": {"value": auroc(scores, outcomes), "bootstrap_95": bootstrap_interval(auroc, pairs)},
    }


def load_signals() -> dict[str, dict]:
    """Stated confidence (0 to 1) and agreement share per claim key, from signals.jsonl."""
    if not SIGNALS_JSONL.exists():
        return {}
    signals = {}
    for line in SIGNALS_JSONL.read_text().splitlines():
        record = json.loads(line) if line.strip() else {}
        if record and "error" not in record:
            signals[record["key"]] = record
    return signals


def other_signals(rows: list[dict], signals: dict[str, dict]) -> dict:
    """The step 4 metrics for the score, the stated confidence and the agreement, on the same claims."""
    missing = [r["key"] for r in rows if r["key"] not in signals]
    if missing:
        return {"status": f"not complete: {len(rows) - len(missing)} of {len(rows)} claims have both signals"}
    values = {
        "score": [r["score"] for r in rows],
        "stated_confidence": [signals[r["key"]]["stated_probability"] / 100 for r in rows],
        "agreement": [signals[r["key"]]["support_share"] for r in rows],
    }
    outcomes_all = [None if r["label"] == "unverifiable" else int(r["label"] == "correct") for r in rows]
    labeled = [i for i, o in enumerate(outcomes_all) if o is not None]
    outcomes = [outcomes_all[i] for i in labeled]
    score_workloads = {c: sum(s < c for s in values["score"]) for c in CUTOFFS}

    out = {"claims": len(rows), "labeled": len(labeled), "per_signal": {}}
    for name, vals in values.items():
        lab = [vals[i] for i in labeled]
        pairs = list(zip(lab, outcomes))
        bins = reliability_bins(lab, outcomes, LABEL_EDGES)
        out["per_signal"][name] = {
            "mean_value": mean(lab),
            "distinct_values": len(set(vals)),
            "label_bins": bins,
            "label_bins_ece": expected_calibration_error(bins),
            "brier": brier_score(lab, outcomes),
            "auroc": auroc(lab, outcomes),
            "auroc_bootstrap_95": bootstrap_interval(auroc, pairs),
            "decision": [decision_row(vals, pairs, c) for c in CUTOFFS],
            "workload_curve": workload_curve(vals, outcomes_all),
            "equal_workload": [
                {"sent_to_person": n, "matches_score_cutoff": c,
                 "incorrect_passing": incorrect_passing_at_workload(vals, outcomes_all, n),
                 "incorrect_total": outcomes.count(0)}
                for c, n in score_workloads.items()
            ],
        }
        if name != "score":
            diff, low, high = paired_auroc_difference([values["score"][i] for i in labeled], lab, outcomes)
            out["per_signal"][name]["auroc_minus_score"] = {"value": diff, "bootstrap_95": [low, high]}

    answers = [signals[r["key"]]["support_answers"] for r in rows]
    out["agreement_variation"] = {
        "claims_all_yes": sum(all(a) for a in answers),
        "claims_all_no": sum(not any(a) for a in answers),
        "claims_with_mixed_answers": sum(len(set(a)) > 1 for a in answers),
        "share_with_mixed_answers": sum(len(set(a)) > 1 for a in answers) / len(answers),
    }
    calls = [signals[r["key"]] for r in rows]
    tokens_in, tokens_out = sum(c["input_tokens"] for c in calls), sum(c["output_tokens"] for c in calls)
    out["cost"] = {
        "calls": sum(c["calls"] for c in calls),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "usd": (tokens_in * PRICE_PER_MTOK["input"] + tokens_out * PRICE_PER_MTOK["output"]) / 1_000_000,
    }
    out["per_claim"] = [
        {"key": r["key"], "blind_id": r["blind_id"], "label": r["label"], "score": r["score"],
         "stated_confidence": signals[r["key"]]["stated_probability"], "agreement": signals[r["key"]]["support_share"]}
        for r in rows
    ]
    return out


def load_claims() -> list[dict]:
    with CLAIMS_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unchecked = [r["key"] for r in rows if not r["label"]]
    assert not unchecked, f"claims without a checked label: {unchecked}"
    for r in rows:
        r["score"] = float(r["score"])
    return rows


def audit_summary() -> dict:
    if not AUDIT_CSV.exists():
        return {"status": "not done yet"}
    with AUDIT_CSV.open(newline="", encoding="utf-8") as f:
        checked = [r for r in csv.DictReader(f) if r["verdict"]]
    if not checked:
        return {"status": "not done yet"}
    agreed = sum(r["verdict"] == "agree" for r in checked)
    return {"checked": len(checked), "agreed": agreed, "share_agreed": agreed / len(checked),
            "wilson_95": wilson_interval(agreed, len(checked))}


def rounded(value):
    """Round floats to 4 decimals for results.json; display rounding happens in the write-up."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def analyse(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r["label"] != "unverifiable"]
    pairs = [(r["score"], int(r["label"] == "correct")) for r in labeled]
    scores, outcomes = [s for s, _ in pairs], [y for _, y in pairs]
    all_scores = [r["score"] for r in rows]

    label_bins = reliability_bins(scores, outcomes, LABEL_EDGES)
    fine_bins = reliability_bins(scores, outcomes, FINE_EDGES)
    fine_ok = all(b["claims"] >= MIN_BIN for b in fine_bins)
    base_rate = mean(outcomes)

    return {
        "inputs": {
            "claims_csv_sha256": hashlib.sha256(CLAIMS_CSV.read_bytes()).hexdigest(),
            "bootstrap_resamples": N_BOOT,
            "seed": SEED,
        },
        "overall": accuracy_summary(rows) | {"distinct_scores": len(set(all_scores))},
        "per_confidence_label": {
            name: accuracy_summary([r for r in rows if r["confidence_label"] == name])
            for name in ("high", "medium", "low")
        },
        "reliability": {
            "label_bins": label_bins,
            "label_bins_ece": expected_calibration_error(label_bins),
            "fine_bins": fine_bins if fine_ok else None,
            "fine_bins_ece": expected_calibration_error(fine_bins) if fine_ok else None,
            "fine_bins_note": None if fine_ok else (
                f"not used: a finer bin has fewer than {MIN_BIN} labeled claims "
                f"(counts {[b['claims'] for b in fine_bins]})"),
        },
        "brier": {
            "score": brier_score(scores, outcomes),
            "constant_guess": brier_score([base_rate] * len(outcomes), outcomes),
            "constant_guess_value": base_rate,
            "bootstrap_95": bootstrap_interval(brier_score, pairs),
        },
        "auroc": {
            "value": auroc(scores, outcomes),
            "bootstrap_95": bootstrap_interval(auroc, pairs),
        },
        "decision": {
            "rows": [decision_row(all_scores, pairs, c) for c in CUTOFFS],
            "check_everything": decision_row(all_scores, pairs, 1.01),
            "check_nothing": decision_row(all_scores, pairs, 0.0),
            "workload_curve": workload_curve(
                all_scores, [None if r["label"] == "unverifiable" else int(r["label"] == "correct") for r in rows]),
        },
        "breakdowns": {key: breakdown(rows, key) for key in ("strongest_source_type", "category", "run")},
        "sensitivity_without_judgment_calls": without_judgment_calls(rows),
        "recalibration": cross_validated_recalibration(scores, outcomes),
        "other_signals": other_signals(rows, load_signals()),
        "audit": audit_summary(),
    }


def main() -> None:
    results = rounded(analyse(load_claims()))
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")
    overall, auc = results["overall"], results["auroc"]
    print(f"Wrote {RESULTS_JSON.name}: {overall['labeled']} labeled claims, "
          f"{overall['correct']} correct, AUROC {auc['value']} {auc['bootstrap_95']}")


if __name__ == "__main__":
    main()
