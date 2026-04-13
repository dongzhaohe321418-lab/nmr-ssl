"""Additional figures for the revised preprint:

Fig 3: K-sweep line plot (C MAE and H MAE vs K).
Fig 4: Noise-injection robustness (C/H MAE vs noise level).
Fig 5: Wrong-structure negative control (bar chart: correct vs wrong pass rates).
Fig 6: Scatter plot: predicted vs observed across test molecules with R^2.
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
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"


def fig_k_sweep():
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        d = json.load(f)
    ks = sorted([int(k) for k in d["k_sweep"].keys()])
    c_mae = [d["k_sweep"][str(k)]["c_mae"] for k in ks]
    h_mae = [d["k_sweep"][str(k)]["h_mae"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    ax.plot(ks, c_mae, "o-", color=GREEN, markersize=7, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("$K$ (number of random directions)")
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C vs $K$")
    ax.axvline(8, color="gray", alpha=0.3, linestyle="--")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(ks, h_mae, "s-", color=ORANGE, markersize=7, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("$K$ (number of random directions)")
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H vs $K$")
    ax.axvline(8, color="gray", alpha=0.3, linestyle="--")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_k_sweep.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_k_sweep.png")


def fig_noise_sweep():
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        d = json.load(f)
    order = ["clean", "low", "medium", "high"]
    labels = ["clean", "low", "medium", "high"]
    c_mae = [d["noise_sweep"][k]["c_mae"] for k in order]
    h_mae = [d["noise_sweep"][k]["h_mae"] for k in order]

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]
    ax.bar(x, c_mae, color=GREEN, edgecolor="black", linewidth=0.5, width=0.58)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C robustness to HSQC noise")
    for i, v in enumerate(c_mae):
        ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(c_mae) * 1.22)

    ax = axes[1]
    ax.bar(x, h_mae, color=ORANGE, edgecolor="black", linewidth=0.5, width=0.58)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H robustness to HSQC noise")
    for i, v in enumerate(h_mae):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(h_mae) * 1.28)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_noise_sweep.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_noise_sweep.png")


def fig_wrong_struct():
    with (ROOT / "experiments" / "results_2d" / "reviewer_experiments.json").open() as f:
        d = json.load(f)
    own = d["wrong_structure"]["own"]
    wrong = d["wrong_structure"]["wrong"]
    categories = ["$^{1}$H pass", "$^{13}$C pass", "Joint pass"]
    own_rates = [own["h_rate"] * 100, own["c_rate"] * 100, own["both_rate"] * 100]
    wrong_rates = [wrong["h_rate"] * 100, wrong["c_rate"] * 100, wrong["both_rate"] * 100]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    b1 = ax.bar(x - w/2, own_rates, w, color=GREEN, edgecolor="black", linewidth=0.5, label="Correct structure")
    b2 = ax.bar(x + w/2, wrong_rates, w, color=RED, edgecolor="black", linewidth=0.5, label="Wrong structure (neg. ctrl)")
    for bar, v in zip(b1, own_rates):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b2, wrong_rates):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("Structure-consistency pass rate (%)")
    ax.set_title("Negative control: discrimination at 95% conformal level")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_wrong_struct.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_wrong_struct.png")


if __name__ == "__main__":
    fig_k_sweep()
    fig_noise_sweep()
    fig_wrong_struct()
