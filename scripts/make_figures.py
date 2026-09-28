"""Draw the two figures of the report from results/analysis/paired_tests.csv.

  figures/fig1_focal_effects.png     focal attraction effect per model, 95% CI
  figures/fig2_attractor_effects.png attractor effect (dz) in all four cells

With --poster, larger versions for an A0 poster are written as
figures/poster_fig1_focal_effects.png and figures/poster_fig2_attractor_effects.png
(bigger fonts, no figure title - the poster box has its own heading).
With --poster --column, figures/poster_col_*.png are sized for a narrow column
(the two heat maps stacked vertically).

Run analyze_results.py first.
Usage (from the repository root):  python scripts/make_figures.py [--poster]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "results" / "analysis" / "paired_tests.csv"
FIGURES = ROOT / "figures"

MODELS = ["GPT-2", "GPT-2-large", "BLOOM-560m"]
CONSTRUCTIONS = {"ORC": "Object relative clause (ORC)", "PP": "Prepositional phrase (PP)"}
MEASURES = {
    "surprisal": ("Surprisal (bits)", "attraction = negative"),
    "entropy": ("Attention entropy (bits)", "attraction = positive"),
}
CELLS = {
    "SG_ungram": "SG subj.\nungram.",
    "SG_gram": "SG subj.\ngram.",
    "PL_ungram": "PL subj.\nungram.",
    "PL_gram": "PL subj.\ngram.",
}

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#b8302f"])

# Font size (pt) and figure size (inches) for the report and the poster versions.
STYLES = {
    "report": {"font": 9, "fig1": (7.2, 5.0), "fig2": (7.2, 3.6), "title": True,
               "labels": MODELS, "note_y": -0.12, "note1_y": -0.005},
    "poster": {"font": 22, "fig1": (13.0, 9.6), "fig2": (20.0, 9.2), "title": False,
               "labels": ["GPT-2", "GPT-2\nlarge", "BLOOM\n560m"], "note_y": -0.07, "note1_y": -0.02},
    # narrow poster column: the two heat maps stacked instead of side by side
    "poster_column": {"font": 20, "fig1": (10.5, 8.8), "short_ylabel": True, "fig2": (10.5, 10.0), "title": False,
                      "labels": ["GPT-2", "GPT-2\nlarge", "BLOOM\n560m"], "note_y": -0.07,
                      "note1_y": -0.02, "stack": True},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def focal_effects(tests, style):
    """Rows: construction; columns: measure. Dot = mean item-level effect,
    bar = 95% CI of the paired difference, hollow = CI includes zero."""
    fs = style["font"]
    k = fs / 9  # scales line widths and markers with the font
    focal = tests[tests.cell == "SG_ungram"]
    fig, axes = plt.subplots(2, 2, figsize=style["fig1"], sharex=True)
    for row, construction in enumerate(CONSTRUCTIONS):
        for col, (metric, (label, direction)) in enumerate(MEASURES.items()):
            ax = axes[row, col]
            d = focal[(focal.construction == construction) & (focal.metric == metric)]
            d = d.set_index("model").loc[MODELS]
            x = np.arange(len(MODELS))
            ax.axhline(0, color=AXIS, lw=1 * k, zorder=1)
            ax.vlines(x, d.ci95_low, d.ci95_high, color=BLUE, lw=2 * k, zorder=2)
            crosses_zero = (d.ci95_low < 0) & (d.ci95_high > 0)
            ax.scatter(x, d.mean_effect, s=45 * k * k, zorder=3, linewidths=2 * k,
                       edgecolors=BLUE, facecolors=np.where(crosses_zero, "white", BLUE))
            ax.grid(axis="y", color=GRID, lw=0.6 * k)
            ax.set_axisbelow(True)
            ax.set_xticks(x, style["labels"])
            ax.tick_params(labelsize=fs, width=k, length=3 * k)
            ax.set_xlim(-0.5, len(MODELS) - 0.5)
            if row == 0:
                ax.set_title(f"{label}\n{direction}", fontsize=fs)
            if col == 0:
                ylabel = construction if style.get("short_ylabel") else \
                    f"{construction}\nplural − singular attractor"
                ax.set_ylabel(ylabel, fontsize=fs, fontweight="bold" if style.get("short_ylabel") else None)
    if style["title"]:
        fig.suptitle("Focal attraction effect (singular subject, ungrammatical verb)",
                     x=0.02, ha="left", fontsize=fs + 1, color=INK)
    note = ("Mean item-level difference with 95% CI (ORC: 48 items, PP: 24). "
            "Hollow marker = CI includes zero.")
    if style.get("short_ylabel"):
        note = ("Plural − singular attractor: mean item-level difference with 95% CI\n"
                "(ORC: 48 items, PP: 24). Hollow marker = CI includes zero.")
    fig.text(0.02, style["note1_y"], note, fontsize=fs * 0.83, color=MUTED, va="top")
    fig.tight_layout()
    return fig


def attractor_effects_by_cell(tests, style):
    """Heat map of the paired effect size dz in all four cells. Humans
    (Wagers et al., 2009) show an effect only in the first column."""
    fs = style["font"]
    small = fs * 0.83
    stack = style.get("stack", False)
    if stack:  # ORC above PP, colour bar in a thin row underneath
        fig = plt.figure(figsize=style["fig2"])
        grid = fig.add_gridspec(3, 1, height_ratios=[6, 6, 0.3], hspace=0.55)
        axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
        colorbar_ax = fig.add_subplot(grid[2])
    else:
        fig, axes = plt.subplots(1, 2, figsize=style["fig2"], sharey=True)
    norm = TwoSlopeNorm(vcenter=0, vmin=-3, vmax=3)
    rows = [(m, model) for m in MEASURES for model in MODELS]
    for ax, construction in zip(axes, CONSTRUCTIONS):
        d = tests[tests.construction == construction]
        grid = np.array([[d[(d.metric == m) & (d.model == model) & (d.cell == c)].dz.item()
                          for c in CELLS] for m, model in rows])
        p = np.array([[d[(d.metric == m) & (d.model == model) & (d.cell == c)].p.item()
                       for c in CELLS] for m, model in rows])
        image = ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                text = f"{grid[i, j]:.2f}" + ("*" if p[i, j] < .05 else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=small,
                        color="white" if abs(grid[i, j]) > 1.6 else INK)
        ax.axhline(len(MODELS) - 0.5, color="white", lw=3 * fs / 9)
        ax.set_xticks(range(len(CELLS)), CELLS.values(), fontsize=small)
        ax.set_yticks(range(len(rows)), [f"{MEASURES[m][0].split(' (')[0]} · {model}"
                                         for m, model in rows], fontsize=small)
        ax.set_title(CONSTRUCTIONS[construction], fontsize=fs)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if stack:
        bar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    else:
        bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label("dz (plural − singular attractor)", color=INK_2, fontsize=small)
    bar.ax.tick_params(labelsize=small)
    bar.outline.set_visible(False)
    if style["title"]:
        fig.suptitle("Attractor effect in all four subject × grammaticality cells",
                     x=0.02, ha="left", fontsize=fs + 1, color=INK)
    if stack:
        fig.text(0.02, 0.0, "dz = mean item-level effect / its SD across items.\nHumans show an "
                 "effect only in the first column. * p < .05 (uncorrected).\nIn plural-subject cells "
                 "the mismatching attractor is singular,\nso an attraction-like effect has the opposite "
                 "sign.", fontsize=small, color=MUTED, va="top")
        return fig
    fig.text(0.02, style["note_y"], "dz = mean item-level effect / its SD across items. Humans show an "
             "effect only in the first column.\n* p < .05 (uncorrected). In plural-subject cells "
             "the mismatching attractor is singular, so an attraction-like effect has the "
             "opposite sign.", fontsize=small, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poster", action="store_true", help="large versions for an A0 poster")
    parser.add_argument("--column", action="store_true",
                        help="with --poster: versions sized for a narrow poster column")
    args = parser.parse_args()

    if args.poster and args.column:
        style, prefix = STYLES["poster_column"], "poster_col_"
    elif args.poster:
        style, prefix = STYLES["poster"], "poster_"
    else:
        style, prefix = STYLES["report"], ""
    FIGURES.mkdir(exist_ok=True)
    tests = pd.read_csv(TESTS)
    focal_effects(tests, style).savefig(FIGURES / f"{prefix}fig1_focal_effects.png")
    attractor_effects_by_cell(tests, style).savefig(FIGURES / f"{prefix}fig2_attractor_effects.png")
    plt.close("all")
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
