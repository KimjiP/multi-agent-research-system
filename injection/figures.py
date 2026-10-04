"""The figure for the planted-source test, drawn only from injection/results.json.

    uv run --group analysis python -m injection.figures

Writes injection/figures/outcomes.png. It uses the blog's fonts and colors from
calibration/figures.py.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from calibration.figures import CREAM, DPI, HEIGHT, INK, LINE_GRAY, MID_GRAY, WIDTH, setup_fonts  # noqa: E402
from injection.runner import QUESTIONS, ROOT  # noqa: E402

RESULTS_JSON = ROOT / "results.json"
OUT = ROOT / "figures"

ATTACKS = {
    "a1": "A1  False date,\none blog page",
    "a2": "A2  False date,\nthree \"legal\" pages",
    "a3": "A3  Instruction to\nstate a false rule",
    "a4": "A4  Instruction to\nadd a link",
}
PHASES = {"baseline": "Before", "defense": "With D2"}
# How each final report treated the planted content, from harmless to harmful: a gray for
# "not there", then one orange ramp, light to dark (validated as an ordinal ramp on CREAM)
OUTCOMES = {
    "not in report": ("Not in the report", "#D9D8D2"),
    "as wrong": ("In it, called wrong", "#EE9566"),
    "as disputed": ("In it, called disputed", "#E0612C"),
    "as fact": ("In it, stated as fact", "#9C3A16"),
}
BAR, GAP = 0.34, 0.06  # bar height and the space between the two bars of one attack, in row units


def outcome_counts(case_q4: dict, case_q3: dict) -> tuple[dict, int, int]:
    """Final reports per outcome over both questions, with the number of runs and approved reports."""
    counts = {key: 0 for key in OUTCOMES}
    runs = approved = 0
    for case in (case_q4, case_q3):
        done = case["runs"] - case["pipeline_failed"]
        if case["readings"]["not read"]:
            raise ValueError(f"{case['attack']}: some reports with planted content are not read yet")
        counts["not in report"] += done - case["in_report"]
        for key in ("as wrong", "as disputed", "as fact"):
            counts[key] += case["readings"][key]
        runs += case["runs"]
        approved += case["approved"]
    return counts, runs, approved


def outcomes(results: dict) -> None:
    phases = [p for p in PHASES if p in results]
    runs_per_case = results[phases[0]]["cases"]["q4_a1"]["runs"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig.text(0.045, 0.93, "What the reports did with the planted content", fontfamily="Libre Baskerville",
             fontsize=17, color=INK, va="top")
    fig.text(0.045, 0.855,
             f"Final reports, {2 * runs_per_case} per attack: {runs_per_case} for each of {len(QUESTIONS)} questions. "
             "D2 stops counting a site as legal analysis because of its name.",
             fontsize=10, color=MID_GRAY, va="top")
    ax = fig.add_axes((0.285, 0.15, 0.555, 0.55))
    # Attack names sit in figure x and data y, so they stay inside the left margin
    names = blended_transform_factory(fig.transFigure, ax.transData)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE_GRAY)
    ax.tick_params(length=0, pad=6)

    rows = list(ATTACKS)
    for r, attack in enumerate(rows):
        for p, phase in enumerate(phases):
            cases = results[phase]["cases"]
            counts, runs, approved = outcome_counts(cases[f"q4_{attack}"], cases[f"q3_{attack}"])
            y = r + (p - (len(phases) - 1) / 2) * (BAR + GAP)
            left = 0
            for key, (_, color) in OUTCOMES.items():
                if counts[key]:
                    # A thin gap in the background color between segments, not a border
                    ax.barh(y, counts[key], left=left, height=BAR, color=color, edgecolor=CREAM, linewidth=1.5)
                left += counts[key]
            ax.text(-0.25, y, PHASES[phase], ha="right", va="center", fontsize=9, color=MID_GRAY)
            ax.text(runs + 0.25, y, f"{approved} of {runs} approved", ha="left", va="center", fontsize=9,
                    color=MID_GRAY)
        ax.text(0.045, r, ATTACKS[attack], transform=names, ha="left", va="center", fontsize=10, color=INK,
                linespacing=1.3)

    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(0, 2 * runs_per_case)
    ax.set_xticks(range(0, 2 * runs_per_case + 1, 2))
    ax.set_xlabel("Final reports")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in OUTCOMES.values()]
    fig.legend(handles, [name for name, _ in OUTCOMES.values()], loc="upper left", bbox_to_anchor=(0.045, 0.79),
               ncol=4, frameon=False, fontsize=9.5, handlelength=1.2, handleheight=1.2, columnspacing=1.6)
    OUT.mkdir(exist_ok=True)
    path = OUT / "outcomes.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT.parent)}")


def main() -> None:
    setup_fonts()
    outcomes(json.loads(RESULTS_JSON.read_text()))


if __name__ == "__main__":
    main()
