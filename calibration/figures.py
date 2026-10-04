"""Figures for the calibration write-up, drawn only from calibration/results.json.

    uv run --group analysis python -m calibration.figures

Writes PNGs to calibration/figures/. The fonts are the blog's (Libre Baskerville for
titles, Manrope for text), downloaded once into calibration/.fontcache/.
"""

import json
import math
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fontTools.ttLib import TTFont  # noqa: E402
from fontTools.varLib.instancer import instantiateVariableFont  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_JSON = ROOT / "results.json"
OUT = ROOT / "figures"
FONT_CACHE = ROOT / ".fontcache"
GOOGLE_FONTS = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FONTS = {
    "LibreBaskerville.ttf": f"{GOOGLE_FONTS}/librebaskerville/LibreBaskerville%5Bwght%5D.ttf",
    "Manrope.ttf": f"{GOOGLE_FONTS}/manrope/Manrope%5Bwght%5D.ttf",
}
# matplotlib reads only the default weight of a variable font (Manrope's is 200), so fixed
# weights are cut from the variable files: (source, output, weight)
STATIC_FONTS = [
    ("LibreBaskerville.ttf", "LibreBaskerville-Regular.ttf", 400),
    ("Manrope.ttf", "Manrope-Regular.ttf", 400),
    ("Manrope.ttf", "Manrope-SemiBold.ttf", 600),
]

# The blog's palette: cream ground, warm near-black and grays
CREAM, INK, MID_GRAY, SOFT_GRAY, LINE_GRAY = "#FBFBF9", "#151513", "#5C5C56", "#E6E6E0", "#B8B8B0"
# Categorical slots 1-3, fixed order (validated light-mode set)
SERIES = {"score": "#2a78d6", "stated_confidence": "#eb6834", "agreement": "#1baf7a"}
SERIES_NAMES = {
    "score": "Evidence score",
    "stated_confidence": "Model's stated confidence",
    "agreement": "Agreement over 5 checks",
}
# A second cue next to color, so the lines are told apart without it
SERIES_STYLES = {"score": "-", "stated_confidence": (0, (6, 2)), "agreement": (0, (1.5, 1.5))}
WIDTH, HEIGHT, DPI = 8.2, 4.8, 200  # 1640 x 960 px


def setup_fonts() -> None:
    FONT_CACHE.mkdir(exist_ok=True)
    for name, url in FONTS.items():
        if not (FONT_CACHE / name).exists():
            urllib.request.urlretrieve(url, FONT_CACHE / name)
    for source, output, weight in STATIC_FONTS:
        path = FONT_CACHE / output
        if not path.exists():
            instantiateVariableFont(TTFont(FONT_CACHE / source), {"wght": weight}, updateFontNames=True).save(path)
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "Manrope",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": MID_GRAY,
        "xtick.color": MID_GRAY,
        "ytick.color": MID_GRAY,
        "axes.edgecolor": LINE_GRAY,
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
    })


def new_figure(title: str, subtitle: str):
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    fig.text(0.045, 0.93, title, fontfamily="Libre Baskerville", fontsize=17, color=INK, va="top")
    fig.text(0.045, 0.855, subtitle, fontsize=10, color=MID_GRAY, va="top")
    ax = fig.add_axes((0.09, 0.13, 0.86, 0.64))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=6)
    ax.grid(color=SOFT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def percent(value: float) -> str:
    return f"{round(value * 100)}%"


def label_line(ax, x0: float, y0: float, x1: float, y1: float, x: float, text: str, below: bool = False) -> None:
    """Write `text` along the line from (x0, y0) to (x1, y1), starting at data x, at the line's screen angle."""
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, y0), (x1, y1)])
    angle = math.degrees(math.atan2(py1 - py0, px1 - px0))
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    ax.text(x, y + (-0.02 if below else 0.015), text, rotation=angle, rotation_mode="anchor",
            va="top" if below else "baseline", fontsize=9, color=MID_GRAY)


def reliability(results: dict) -> Path:
    """Share correct against mean score for the three confidence labels, with Wilson intervals."""
    labels = results["per_confidence_label"]
    overall = results["overall"]
    fig, ax = new_figure(
        "How often each confidence label was right",
        f"{overall['labeled']} claims from 8 reports, checked against official sources. "
        "Bars show 95% intervals.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for edge in (0.4, 0.7):
        ax.axvline(edge, color=LINE_GRAY, linewidth=0.8, zorder=1)
    for low, high, name in ((0.0, 0.4, "Low"), (0.4, 0.7, "Medium"), (0.7, 1.0, "High")):
        ax.text((low + high) / 2, 0.99, f"{name} label", ha="center", va="top", fontsize=9.5,
                fontfamily="Manrope SemiBold", color=MID_GRAY)
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=LINE_GRAY, linewidth=1.2, zorder=1)
    label_line(ax, 0, 0, 1, 1, 0.24, "If the score were a probability")
    share = overall["share_correct"]
    ax.axhline(share, color=MID_GRAY, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.text(0.245, share + 0.015, f"All claims: {percent(share)} correct", fontsize=9, color=MID_GRAY,
            bbox={"facecolor": CREAM, "edgecolor": "none", "pad": 1})

    for name in ("low", "medium", "high"):
        s = labels[name]
        x, y = s["mean_score"], s["share_correct"]
        ax.plot([x, x], s["wilson_95"], color=INK, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot(x, y, "o", markersize=9, color=INK, markeredgecolor=CREAM, markeredgewidth=2, zorder=4)
        right = name == "low"
        ax.text(x + (0.025 if right else -0.025), s["wilson_95"][0], f"{s['correct']} of {s['labeled']} correct",
                ha="left" if right else "right", va="bottom", fontsize=9.5, color=INK)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Confidence score (mean of the claims in each label)")
    ax.set_ylabel("Share of claims that were correct")
    path = OUT / "reliability.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def workload(results: dict) -> Path:
    """Share of wrong claims a person catches, by the share of claims they check, for each signal."""
    curves = {"score": results["decision"]["workload_curve"]}
    other = results["other_signals"]
    if "per_signal" in other:
        curves.update({k: v["workload_curve"] for k, v in other["per_signal"].items() if k != "score"})
    decision = results["decision"]
    wrong, labeled = decision["rows"][0]["incorrect_total"], results["overall"]["labeled"]
    fig, ax = new_figure(
        "How many wrong claims a person would catch",
        f"A person checks the least confident claims first. {wrong} of the {labeled} checked claims were wrong.",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=LINE_GRAY, linewidth=1.2, zorder=1)
    label_line(ax, 0, 0, 1, 1, 0.6, "Checking claims at random", below=True)
    for name, curve in curves.items():
        xs, ys = [p[0] for p in curve], [p[1] for p in curve]
        ax.plot(xs, ys, color=SERIES[name], linewidth=2, linestyle=SERIES_STYLES[name], zorder=3,
                solid_joinstyle="round", dash_joinstyle="round", label=SERIES_NAMES[name])
    if len(curves) > 1:
        ax.legend(loc="upper left", frameon=False, fontsize=9.5, handlelength=3, borderaxespad=0.6)
    for row in decision["rows"]:
        x, y = row["share_sent"], 1 - row["share_of_incorrect_passing"]
        ax.plot(x, y, "o", markersize=8, color=SERIES["score"], markeredgecolor=CREAM, markeredgewidth=2, zorder=4)
        ax.text(x + 0.015, y - 0.03, f"Below {row['cutoff']}: check {percent(x)}, catch {percent(y)}",
                ha="left", va="top", fontsize=9, color=INK,
                bbox={"facecolor": CREAM, "edgecolor": "none", "pad": 1})
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks, [percent(t) for t in ticks])
    ax.set_yticks(ticks, [percent(t) for t in ticks])
    ax.set_xlabel("Share of all claims a person checks")
    ax.set_ylabel("Share of wrong claims caught")
    path = OUT / "workload.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> None:
    setup_fonts()
    OUT.mkdir(exist_ok=True)
    results = json.loads(RESULTS_JSON.read_text())
    for path in (reliability(results), workload(results)):
        print(f"Wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
