"""Reviewer figure: predicted vs observed scatter for all test atoms.

Reads experiments/results_2d/scatter_points.npz (produced by
experiments/collect_scatter_points.py) and emits a two-panel scatter
plot with y=x line and Pearson R^2 annotated in the top-left of each
panel. Saves to docs/2d/figures/fig_scatter_r2.{png,pdf}.
"""

from __future__ import annotations

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

GREEN = "#2ca02c"
ORANGE = "#ff7f0e"


def pearson_r2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = pred.astype(np.float64)
    true = true.astype(np.float64)
    pm = pred - pred.mean()
    tm = true - true.mean()
    denom = np.sqrt((pm * pm).sum() * (tm * tm).sum())
    if denom == 0.0:
        return float("nan")
    r = float((pm * tm).sum() / denom)
    return r * r


def _panel(ax, obs, pred, color, title, xlabel, ylabel, pad):
    lo = float(min(obs.min(), pred.min())) - pad
    hi = float(max(obs.max(), pred.max())) + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7, zorder=1)
    ax.scatter(
        obs,
        pred,
        s=8,
        c=color,
        alpha=0.4,
        edgecolors="none",
        zorder=2,
    )
    r2 = pearson_r2(pred, obs)
    mae = float(np.mean(np.abs(pred - obs)))
    ax.text(
        0.04,
        0.96,
        f"$R^2 = {r2:.3f}$\nMAE = {mae:.2f}\n$n = {len(obs)}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="0.6",
            lw=0.6,
            alpha=0.85,
        ),
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")


def main():
    npz_path = ROOT / "experiments" / "results_2d" / "scatter_points.npz"
    d = np.load(npz_path, allow_pickle=True)
    c_pred = d["c_pred"]
    c_true = d["c_true"]
    h_pred = d["h_pred"]
    h_true = d["h_true"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))

    _panel(
        axes[0],
        c_true,
        c_pred,
        GREEN,
        r"(a) $^{13}$C test set",
        r"Observed $^{13}$C shift (ppm)",
        r"Predicted $^{13}$C shift (ppm)",
        pad=5.0,
    )
    _panel(
        axes[1],
        h_true,
        h_pred,
        ORANGE,
        r"(b) $^{1}$H test set",
        r"Observed $^{1}$H shift (ppm)",
        r"Predicted $^{1}$H shift (ppm)",
        pad=0.3,
    )

    fig.tight_layout()
    out_png = OUT / "fig_scatter_r2.png"
    out_pdf = OUT / "fig_scatter_r2.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
