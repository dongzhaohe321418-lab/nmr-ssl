"""Regenerate the wrong-structure negative-control figure with the
constitutional-isomer control as the HEADLINE panel (instead of the easier
random-pair control)."""

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
        iso = json.load(f)
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        rand = json.load(f)

    iso_corr = iso["constitutional_isomer"]
    iso_sc = iso["scaffold_neighbor"]
    rand_own = rand["wrong_structure"]["own"]
    rand_wrong = rand["wrong_structure"]["wrong"]

    cats = ["$^{1}$H", "$^{13}$C", "Joint"]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.9))
    x = np.arange(len(cats))
    w = 0.36

    # Panel A: constitutional isomer (HEADLINE)
    ax = axes[0]
    iso_own_rates = [iso_corr["own_h_rate"] * 100, iso_corr["own_c_rate"] * 100, iso_corr["own_both_rate"] * 100]
    iso_wrong_rates = [iso_corr["wrong_h_rate"] * 100, iso_corr["wrong_c_rate"] * 100, iso_corr["wrong_both_rate"] * 100]
    b1 = ax.bar(x - w / 2, iso_own_rates, w, color=GREEN, edgecolor="black", linewidth=0.5, label="Correct")
    b2 = ax.bar(x + w / 2, iso_wrong_rates, w, color=RED, edgecolor="black", linewidth=0.5, label="Isomer (hard)")
    for b, v in zip(b1, iso_own_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for b, v in zip(b2, iso_wrong_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("(a) Constitutional isomer (n=74)")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    # Panel B: scaffold neighbor
    ax = axes[1]
    sc_own_rates = [iso_sc["own_h_rate"] * 100, iso_sc["own_c_rate"] * 100, iso_sc["own_both_rate"] * 100]
    sc_wrong_rates = [iso_sc["wrong_h_rate"] * 100, iso_sc["wrong_c_rate"] * 100, iso_sc["wrong_both_rate"] * 100]
    b1 = ax.bar(x - w / 2, sc_own_rates, w, color=GREEN, edgecolor="black", linewidth=0.5, label="Correct")
    b2 = ax.bar(x + w / 2, sc_wrong_rates, w, color="#ff7f0e", edgecolor="black", linewidth=0.5, label="Scaffold nb.")
    for b, v in zip(b1, sc_own_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for b, v in zip(b2, sc_wrong_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_title("(b) Scaffold nb. (n=93)")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, loc="upper left", fontsize=7)

    # Panel C: random pair
    ax = axes[2]
    rand_own_rates = [rand_own["h_rate"] * 100, rand_own["c_rate"] * 100, rand_own["both_rate"] * 100]
    rand_wrong_rates = [rand_wrong["h_rate"] * 100, rand_wrong["c_rate"] * 100, rand_wrong["both_rate"] * 100]
    b1 = ax.bar(x - w / 2, rand_own_rates, w, color=GREEN, edgecolor="black", linewidth=0.5, label="Correct")
    b2 = ax.bar(x + w / 2, rand_wrong_rates, w, color=GRAY, edgecolor="black", linewidth=0.5, label="Random")
    for b, v in zip(b1, rand_own_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for b, v in zip(b2, rand_wrong_rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_title("(c) Random (n=155)")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, loc="upper left", fontsize=7)

    fig.subplots_adjust(wspace=0.35)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_wrong_struct_v4.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_wrong_struct_v4.png")


if __name__ == "__main__":
    main()
