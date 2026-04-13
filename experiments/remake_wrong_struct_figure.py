"""Wrong-structure negative-control figure.

Single panel, clean grouped bar chart. X-axis lists the three control
types (constitutional isomer, scaffold neighbour, random pair), with two
bars per control showing the joint pass rate for the correct structure
and for the wrong candidate. This is the one number that matters for
structure-verification discrimination.
"""

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
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})

GREEN = "#2ca02c"; RED = "#d62728"


def main():
    with (ROOT / "experiments" / "results_2d" / "realistic_isomer_control.json").open() as f:
        iso = json.load(f)
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        rand = json.load(f)

    iso_corr = iso["constitutional_isomer"]
    iso_sc = iso["scaffold_neighbor"]
    rand_own = rand["wrong_structure"]["own"]
    rand_wrong = rand["wrong_structure"]["wrong"]

    n_iso = iso_corr["own_n"]
    n_sc = iso_sc["own_n"]
    n_rand = rand["wrong_structure"]["n_test"]
    labels = [
        f"Constitutional isomer\n(n = {n_iso})",
        f"Scaffold neighbour\n(n = {n_sc})",
        f"Random pair\n(n = {n_rand})",
    ]
    correct = [
        iso_corr["own_both_rate"] * 100,
        iso_sc["own_both_rate"] * 100,
        rand_own["both_rate"] * 100,
    ]
    wrong = [
        iso_corr["wrong_both_rate"] * 100,
        iso_sc["wrong_both_rate"] * 100,
        rand_wrong["both_rate"] * 100,
    ]
    ratio = [c / max(w, 1e-9) for c, w in zip(correct, wrong)]

    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    b1 = ax.bar(x - w / 2, correct, w, color=GREEN, edgecolor="black",
                linewidth=0.7, label="Correct structure")
    b2 = ax.bar(x + w / 2, wrong, w, color=RED, edgecolor="black",
                linewidth=0.7, label="Wrong candidate")

    for b, v in zip(b1, correct):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, color="black")
    for b, v in zip(b2, wrong):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, color=RED)

    # Discrimination ratio annotation above each group
    for i, r in enumerate(ratio):
        ax.text(i, 108, rf"${r:.1f}\times$",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Joint pass rate (%)")
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # Put legend below the plot so it does not collide with any bar
    ax.legend(frameon=False, loc="upper center", fontsize=10,
              bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_wrong_struct_v4.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_wrong_struct_v4.png")


if __name__ == "__main__":
    main()
