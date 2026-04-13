"""Constitutional-isomer + scaffold-neighbor discrimination figure."""

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


def main():
    with (ROOT / "experiments" / "results_2d" / "realistic_isomer_control.json").open() as f:
        d = json.load(f)

    # Load random-pair wrong-struct for comparison
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        rev = json.load(f)
    rand_own = rev["wrong_structure"]["own"]
    rand_wrong = rev["wrong_structure"]["wrong"]

    categories = ["$^{1}$H pass", "$^{13}$C pass", "Joint"]
    iso_own = [d["constitutional_isomer"]["own_h_rate"] * 100,
               d["constitutional_isomer"]["own_c_rate"] * 100,
               d["constitutional_isomer"]["own_both_rate"] * 100]
    iso_wrong = [d["constitutional_isomer"]["wrong_h_rate"] * 100,
                 d["constitutional_isomer"]["wrong_c_rate"] * 100,
                 d["constitutional_isomer"]["wrong_both_rate"] * 100]
    sc_own = [d["scaffold_neighbor"]["own_h_rate"] * 100,
              d["scaffold_neighbor"]["own_c_rate"] * 100,
              d["scaffold_neighbor"]["own_both_rate"] * 100]
    sc_wrong = [d["scaffold_neighbor"]["wrong_h_rate"] * 100,
                d["scaffold_neighbor"]["wrong_c_rate"] * 100,
                d["scaffold_neighbor"]["wrong_both_rate"] * 100]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    x = np.arange(len(categories))
    w = 0.36

    ax = axes[0]
    b1 = ax.bar(x - w/2, iso_own, w, color=GREEN, edgecolor="black", linewidth=0.5, label="Correct structure")
    b2 = ax.bar(x + w/2, iso_wrong, w, color=RED, edgecolor="black", linewidth=0.5, label="Constitutional isomer")
    for bar, v in zip(b1, iso_own):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b2, iso_wrong):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Consistency pass rate (%)")
    ax.set_title("(a) vs. constitutional isomers (n=74 test mols)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    ax = axes[1]
    b1 = ax.bar(x - w/2, sc_own, w, color=GREEN, edgecolor="black", linewidth=0.5, label="Correct structure")
    b2 = ax.bar(x + w/2, sc_wrong, w, color=RED, edgecolor="black", linewidth=0.5, label="Scaffold neighbor")
    for bar, v in zip(b1, sc_own):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b2, sc_wrong):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Consistency pass rate (%)")
    ax.set_title("(b) vs. Bemis–Murcko scaffold neighbors (n=93)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_isomer_control.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_isomer_control.png")


if __name__ == "__main__":
    main()
