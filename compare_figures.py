"""The figure for the researcher comparison, drawn only from results/researcher_comparison/summary.json.

    uv run --group analysis python compare_figures.py

Writes results/researcher_comparison/searches.png. It uses the blog's fonts and colors from
calibration/figures.py.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from calibration.figures import CREAM, DPI, HEIGHT, INK, LINE_GRAY, MID_GRAY, WIDTH, setup_fonts  # noqa: E402
# compare_researchers.OUT_DIR, kept here so drawing the figure does not import the pipeline
OUT_DIR = Path("results") / "researcher_comparison"
SUMMARY_JSON = OUT_DIR / "summary.json"

QUESTIONS = {
    "q1": "Q1  Hiring tool\nrequirements",
    "q2": "Q2  EU and US\npolicy compared",
    "q3": "Q3  Conformity\nassessment",
    "q4": "Q4  Annex III date",
}
RESEARCHERS = {"fixed": ("Fixed searches", "#8A8A82"), "tools": ("With tools", "#E0612C")}
# (key in summary.json, panel title, scale for the values, x limits, x ticks, tick format)
PANELS = [
    ("searches", "Searches per run", 1, (4, 10), [4, 6, 8, 10], "{:.0f}"),
    ("sources", "Sources per run", 1, (15, 45), [15, 25, 35, 45], "{:.0f}"),
    ("official_share", "From official EU sites", 100, (20, 70), [20, 45, 70], "{:.0f}%"),
]
OFFSET = 0.17  # how far each researcher's row sits from the question's center, in row units


def searches(summary: dict) -> None:
    overall = summary["overall"]
    runs = summary["by_question"]["q1"]["fixed"]["runs"]
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig.text(0.045, 0.93, "The same approvals, spent on different searches", fontfamily="Libre Baskerville",
             fontsize=17, color=INK, va="top")
    fig.text(0.045, 0.855,
             f"Reports approved: {overall['fixed']['approved']} of {overall['fixed']['runs']} with fixed searches, "
             f"{overall['tools']['approved']} of {overall['tools']['runs']} with tools.\n"
             f"Small dots are single runs, {runs} per question. The large dot is their mean.",
             fontsize=10, color=MID_GRAY, va="top", linespacing=1.5)

    left, right, top, bottom, gap = 0.215, 0.965, 0.62, 0.12, 0.06
    width = (right - left - gap * (len(PANELS) - 1)) / len(PANELS)
    rows = list(QUESTIONS)
    for p, (key, title, scale, xlim, xticks, fmt) in enumerate(PANELS):
        ax = fig.add_axes((left + p * (width + gap), bottom, width, top - bottom))
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(LINE_GRAY)
        ax.tick_params(length=0, pad=6)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
        for r, question in enumerate(rows):
            # A faint band behind every other question, so the rows read across the panels
            if r % 2 == 0:
                ax.axhspan(r - 0.5, r + 0.5, color="#F1F1EC", zorder=0, linewidth=0)
            for i, (researcher, (_, color)) in enumerate(RESEARCHERS.items()):
                data = summary["by_question"][question][researcher]
                y = r + (i - 0.5) * 2 * OFFSET
                ax.scatter([v * scale for v in data[key]], [y] * len(data[key]), s=18, color=color, alpha=0.45,
                           linewidths=0, zorder=2)
                ax.scatter(data["mean"][key] * scale, y, s=70, color=color, edgecolors=CREAM, linewidths=1.2,
                           zorder=3)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks, [fmt.format(t) for t in xticks])
        ax.grid(axis="x", color=LINE_GRAY, linewidth=0.5, alpha=0.6, zorder=1)
        ax.set_axisbelow(True)
        if p == 0:
            for r, question in enumerate(rows):
                ax.text(-0.04, r, QUESTIONS[question], transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=9.5, color=INK, linespacing=1.3)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=color) for _, color in RESEARCHERS.values()]
    fig.legend(handles, [name for name, _ in RESEARCHERS.values()], loc="upper left", bbox_to_anchor=(0.045, 0.75),
               ncol=2, frameon=False, fontsize=9.5, handletextpad=0.4, columnspacing=1.6)
    path = OUT_DIR / "searches.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    setup_fonts()
    searches(json.loads(SUMMARY_JSON.read_text()))


if __name__ == "__main__":
    main()
