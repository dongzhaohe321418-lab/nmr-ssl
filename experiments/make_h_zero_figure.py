"""Figure: H-zero causal-audit ablation bar chart."""

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

GREEN = "#2ca02c"; ORANGE = "#ff7f0e"; RED = "#d62728"; GRAY = "#8c8c8c"


def main():
    with (ROOT / "experiments" / "results_2d" / "h_zero_ablation.json").open() as f:
        hz = json.load(f)

    # Baseline K=16 numbers (from revision_batch3.json)
    baseline_c = 4.869
    baseline_h = 0.455
    hz_c = hz["aggregate"]["c_mean"]
    hz_h = hz["aggregate"]["h_mean"]

    # Also the supervised-1D baseline on H (random floor)
    sup1d_h = 2.473
    sup1d_c = 5.600

    labels = [
        "Supervised-1D\n($^{1}$H head untrained)",
        "2-D SSL\n(HSQC $^{1}$H zeroed)",
        "2-D SSL\n(full HSQC, baseline)",
    ]
    c_vals = [sup1d_c, hz_c, baseline_c]
    h_vals = [sup1d_h, hz_h, baseline_h]
    colors = [GRAY, RED, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    x = np.arange(len(labels))

    ax = axes[0]
    bars = ax.bar(x, c_vals, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    for b, v in zip(bars, c_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C — unaffected by HSQC $^{1}$H zeroing")
    ax.set_ylim(0, max(c_vals) * 1.22)

    ax = axes[1]
    bars = ax.bar(x, h_vals, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    for b, v in zip(bars, h_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H — collapses $\\sim$10$\\times$ when HSQC $^{1}$H zeroed")
    ax.set_ylim(0, max(h_vals) * 1.25)

    # Annotate the ~10x collapse with a subtle red bracket between red and green bars
    ax.annotate("", xy=(2, baseline_h + 0.18), xytext=(1, hz_h - 0.35),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(1.5, (hz_h + baseline_h) / 2 + 0.1,
            f"{hz_h / baseline_h:.0f}$\\times$",
            color=RED, fontsize=11, fontweight="bold", ha="center")

    fig.subplots_adjust(bottom=0.22, wspace=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_h_zero.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_h_zero.png")


if __name__ == "__main__":
    main()
