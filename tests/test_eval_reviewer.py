"""Tests for the planted-error gate in eval_reviewer.py (no API calls)."""

from eval_reviewer import GATE_MAX_FALSE_ALARMS, GATE_MIN_CAUGHT, GATE_PLANTED, gate_failures, summarize

# The gate run plants 13 errors: 4 of three types and 1 changed date
PLANTED = {"fabricated_link": 4, "unsupported_claim": 4, "uncited_claim": 4, "changed_date": 1}
ALLOWED_MISSES = GATE_PLANTED - GATE_MIN_CAUGHT


def rows(missed: int = 0, false_alarms: int = 0, planted: dict = PLANTED) -> list[dict]:
    """Reviewer verdicts: `missed` planted errors approved, `false_alarms` of 4 clean reports rejected."""
    out = []
    for error_type, n in planted.items():
        for _ in range(n):
            approved = missed > 0
            missed -= approved
            out.append({"error_type": error_type, "approved": approved, "failed_checks": {} if approved else {"x": ""}})
    for k in range(4):
        rejected = k < false_alarms
        out.append({"error_type": "clean", "approved": not rejected, "failed_checks": {"x": ""} if rejected else {}})
    return out


def test_gate_passes_at_its_limits():
    assert gate_failures(summarize(rows(missed=ALLOWED_MISSES, false_alarms=GATE_MAX_FALSE_ALARMS))) == []


def test_gate_fails_when_one_more_planted_error_gets_through():
    caught = GATE_MIN_CAUGHT - 1
    assert gate_failures(summarize(rows(missed=ALLOWED_MISSES + 1))) == [
        f"only {caught} of 13 planted errors were caught, the minimum is {GATE_MIN_CAUGHT}"
    ]


def test_gate_fails_on_one_more_false_alarm():
    failures = gate_failures(summarize(rows(false_alarms=GATE_MAX_FALSE_ALARMS + 1)))
    assert failures == [f"{GATE_MAX_FALSE_ALARMS + 1} clean reports were rejected, the limit is {GATE_MAX_FALSE_ALARMS}"]


def test_gate_fails_when_fewer_errors_were_planted():
    failures = gate_failures(summarize(rows(planted=PLANTED | {"changed_date": 0})))
    assert failures[0] == "12 errors were planted, expected 13"
