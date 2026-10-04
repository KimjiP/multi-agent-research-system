"""Tests for the calibration metrics (no data files, no API calls)."""

import math

import pytest
from calibration.analyze import (
    auroc,
    bin_of,
    bootstrap_interval,
    brier_score,
    decision_row,
    cross_validated_recalibration,
    expected_calibration_error,
    fit_logistic,
    incorrect_passing_at_workload,
    paired_auroc_difference,
    reliability_bins,
    wilson_interval,
)


def test_wilson_interval_known_value():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)


def test_wilson_interval_stays_inside_zero_and_one():
    assert wilson_interval(0, 5)[0] == pytest.approx(0.0)
    assert wilson_interval(5, 5)[1] == pytest.approx(1.0)


def test_bin_edges_belong_to_the_upper_bin():
    edges = [0.0, 0.4, 0.7]
    assert [bin_of(s, edges) for s in (0.0, 0.39, 0.4, 0.69, 0.7, 1.0)] == [0, 0, 1, 1, 2, 2]


def test_reliability_bins_and_ece():
    scores = [0.1, 0.3, 0.5, 0.9, 0.9]
    outcomes = [0, 1, 1, 1, 0]
    bins = reliability_bins(scores, outcomes, [0.0, 0.4, 0.7])
    assert [b["claims"] for b in bins] == [2, 1, 2]
    assert bins[0]["share_correct"] == 0.5 and bins[0]["mean_score"] == pytest.approx(0.2)
    # weighted gaps: 2/5 * |0.2 - 0.5| + 1/5 * |0.5 - 1| + 2/5 * |0.9 - 0.5|
    assert expected_calibration_error(bins) == pytest.approx(0.12 + 0.10 + 0.16)


def test_empty_bin_is_skipped_in_ece():
    bins = reliability_bins([0.8, 0.9], [1, 1], [0.0, 0.4, 0.7])
    assert bins[0]["claims"] == 0 and bins[0]["share_correct"] is None
    assert expected_calibration_error(bins) == pytest.approx(0.15)


def test_brier_score():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_auroc_separation_and_ties():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5


def test_auroc_needs_both_outcomes():
    with pytest.raises(ValueError):
        auroc([0.5, 0.6], [1, 1])


def test_bootstrap_interval_is_repeatable_and_brackets_the_estimate():
    pairs = [(0.9, 1), (0.8, 1), (0.7, 0), (0.6, 1), (0.3, 0), (0.2, 0), (0.4, 1), (0.1, 0)]
    first = bootstrap_interval(auroc, pairs, n_boot=500, seed=1)
    assert first == bootstrap_interval(auroc, pairs, n_boot=500, seed=1)
    estimate = auroc([s for s, _ in pairs], [y for _, y in pairs])
    assert first[0] <= estimate <= first[1]


def test_decision_row_counts():
    all_scores = [0.1, 0.3, 0.5, 0.8, 0.95, 0.2]  # the last claim is unverifiable
    pairs = [(0.1, 0), (0.3, 1), (0.5, 0), (0.8, 1), (0.95, 0)]
    row = decision_row(all_scores, pairs, 0.4)
    assert row["sent_to_person"] == 3 and row["share_sent"] == pytest.approx(0.5)
    assert row["incorrect_passing"] == 2 and row["incorrect_total"] == 3
    assert row["share_incorrect_among_passing"] == pytest.approx(2 / 3)


def test_equal_workload_without_ties_matches_the_cutoff():
    values = [0.1, 0.3, 0.5, 0.8, 0.95, 0.2]
    outcomes = [0, 1, 0, 1, 0, None]  # None: unverifiable
    # sending the 3 lowest (0.1, 0.2, 0.3) catches one incorrect claim; two pass
    assert incorrect_passing_at_workload(values, outcomes, 3) == 2


def test_equal_workload_splits_ties_in_proportion():
    values = [0.0, 0.0, 1.0, 1.0]
    outcomes = [0, 1, 0, 1]
    # one slot among two tied claims: half an incorrect claim is sent on average
    assert incorrect_passing_at_workload(values, outcomes, 1) == pytest.approx(1.5)
    assert incorrect_passing_at_workload(values, outcomes, 0) == 2
    assert incorrect_passing_at_workload(values, outcomes, 4) == 0


def test_workload_curve_runs_from_nothing_to_everything():
    from calibration.analyze import workload_curve

    curve = workload_curve([0.1, 0.9, 0.5, 0.2], [0, 1, None, 0])
    assert curve[0] == [0.0, 0.0] and curve[-1] == [1.0, 1.0]
    assert curve[2] == [0.5, 1.0]  # the two lowest values are the two incorrect claims


def test_paired_auroc_difference():
    outcomes = [1, 1, 0, 0]
    perfect, reversed_ = [0.9, 0.8, 0.2, 0.1], [0.1, 0.2, 0.8, 0.9]
    diff, low, high = paired_auroc_difference(reversed_, perfect, outcomes, n_boot=200, seed=1)
    assert diff == 1.0 and low <= diff <= high


def test_fit_logistic_recovers_group_rates():
    # P(correct) is 0.25 at score 0 and 0.75 at score 1, so a = logit(0.25) and b = logit(0.75) - a
    x = [0, 0, 0, 0, 1, 1, 1, 1]
    y = [0, 0, 0, 1, 1, 1, 1, 0]
    a, b = fit_logistic(x, y)
    assert a == pytest.approx(math.log(0.25 / 0.75), abs=1e-6)
    assert b == pytest.approx(2 * math.log(3), abs=1e-6)


def test_fit_logistic_needs_varying_scores():
    with pytest.raises(ValueError):
        fit_logistic([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0])


def test_cross_validated_recalibration_predicts_every_claim_out_of_fold():
    scores = [i / 20 for i in range(20)]
    outcomes = [0, 1] * 10
    result = cross_validated_recalibration(scores, outcomes, folds=5, seed=1)
    assert len(result["fits"]) == 5
    low, high = result["held_out_prediction_range"]
    assert 0 < low <= high < 1


def test_reference_rows():
    pairs = [(0.2, 0), (0.9, 1)]
    assert decision_row([0.2, 0.9], pairs, 1.01)["incorrect_passing"] == 0
    assert decision_row([0.2, 0.9], pairs, 0.0)["sent_to_person"] == 0
