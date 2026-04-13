"""Regenerate the wrong-structure negative-control figure as a clean 3-panel
layout with short titles that do not collide at narrow figure widths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "2d" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})

GREEN = "#2ca02c"; RED = "#d62728"; GRAY = "#8c8c8c"


def _draw(ax, own_rates, wrong_rates, wrong_color, wrong_label, title, cats, w, x, show_ylabel):
    b1 = ax.bar(x - w / 2, own_rates, w, color=GREEN, edgecolor="black",
                linewidth=0.5, label="Correct")
    b2 = ax.bar(x + w / 2, wrong_rates, w, color=wrong_color, edgecolor="black",
                linewidth=0.5, label=wrong_label)
    # Correct bars: label on TOP (all are big)
    for b, v in zip(b1, own_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.0f}",
                ha="center", va="bottom", fontsize=7, color="black")
    # Wrong bars: label on TOP of the bar (even when small, since y-axis
    # is 0-115 the text has room). For visibility on small bars we place
    # the label slightly higher than the bar top.
    for b, v in zip(b2, wrong_rates):
        y_text = max(v, 2) + 1.8
        ax.text(b.get_x() + b.get_width() / 2, y_text, f"{v:.0f}",
                ha="center", va="bottom", fontsize=7, color=wrong_color)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Pass rate (%)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 120)
    ax.legend(frameon=False, loc="upper right", fontsize=7,
              handlelength=1.2, handletextpad=0.4)


def main():
    with (ROOT / "experiments" / "results_2d" / "realistic_isomer_control.json").open() as f:
        iso = json.load(f)
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        rand = json.load(f)

    iso_corr = iso["constitutional_isomer"]
    iso_sc = iso["scaffold_neighbor"]
    rand_own = rand["wrong_structure"]["own"]
    rand_wrong = rand["wrong_structure"]["wrong"]

    cats = ["$^{1}$H", "$^{13}$C", "Joint"]
    x = np.arange(len(cats))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9))

    # Panel (a): constitutional isomer — HEADLINE
    iso_own_rates = [iso_corr["own_h_rate"] * 100,
                     iso_corr["own_c_rate"] * 100,
                     iso_corr["own_both_rate"] * 100]
    iso_wrong_rates = [iso_corr["wrong_h_rate"] * 100,
                       iso_corr["wrong_c_rate"] * 100,
                       iso_corr["wrong_both_rate"] * 100]
    _draw(axes[0], iso_own_rates, iso_wrong_rates, RED, "Isomer",
          "(a) Const. isomer, n=74", cats, w, x, show_ylabel=True)

    # Panel (b): scaffold neighbor
    sc_own_rates = [iso_sc["own_h_rate"] * 100,
                    iso_sc["own_c_rate"] * 100,
                    iso_sc["own_both_rate"] * 100]
    sc_wrong_rates = [iso_sc["wrong_h_rate"] * 100,
                      iso_sc["wrong_c_rate"] * 100,
                      iso_sc["wrong_both_rate"] * 100]
    _draw(axes[1], sc_own_rates, sc_wrong_rates, "#ff7f0e", "Scaffold",
          "(b) Scaffold nb., n=93", cats, w, x, show_ylabel=False)

    # Panel (c): random pair
    rand_own_rates = [rand_own["h_rate"] * 100,
                      rand_own["c_rate"] * 100,
                      rand_own["both_rate"] * 100]
    rand_wrong_rates = [rand_wrong["h_rate"] * 100,
                        rand_wrong["c_rate"] * 100,
                        rand_wrong["both_rate"] * 100]
    _draw(axes[2], rand_own_rates, rand_wrong_rates, GRAY, "Random",
          "(c) Random, n=155", cats, w, x, show_ylabel=False)

    fig.subplots_adjust(wspace=0.32)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_wrong_struct_v4.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_wrong_struct_v4.png")


if __name__ == "__main__":
    main()
