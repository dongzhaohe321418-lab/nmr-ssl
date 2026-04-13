"""Realistic HSQC degradation figure."""

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

GREEN = "#2ca02c"; ORANGE = "#ff7f0e"


def main():
    with (ROOT / "experiments" / "results_2d" / "realistic_hsqc.json").open() as f:
        d = json.load(f)
    recipes = ["clean", "realistic", "merge", "worst"]
    labels = ["clean", "realistic", "+ merging", "aggressive"]
    c_mae = [d["results"][r]["test_c_mae"] for r in recipes]
    h_mae = [d["results"][r]["test_h_mae"] for r in recipes]

    x = np.arange(len(recipes))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]
    ax.bar(x, c_mae, color=GREEN, edgecolor="black", linewidth=0.5, width=0.58)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C robustness to realistic degradation")
    for i, v in enumerate(c_mae):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(c_mae) * 1.22)
    ax.axhline(c_mae[0], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    ax = axes[1]
    ax.bar(x, h_mae, color=ORANGE, edgecolor="black", linewidth=0.5, width=0.58)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H robustness to realistic degradation")
    for i, v in enumerate(h_mae):
        ax.text(i, v + 0.014, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(h_mae) * 1.28)
    ax.axhline(h_mae[0], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_realistic_hsqc.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_realistic_hsqc.png")


if __name__ == "__main__":
    main()
