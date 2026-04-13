"""Generate publication-quality figures for the v2 preprint.

Fig 1: bar chart of ¹³C and ¹H test MAE across the three variants (3 seeds).
Fig 2: predicted vs observed scatter for ¹³C and ¹H on the test molecules
       from the chemistry demo, with conformal bands.
Fig 3: residual histograms with conformal quantile markers.
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


def fig_mae_bars():
    with (ROOT / "experiments" / "results_2d" / "summary.json").open() as f:
        summary = json.load(f)
    agg = summary["aggregate"]
    variants = ["supervised_1d", "sort_match_ssl_1d", "sort_match_ssl_2d"]
    labels = ["Supervised\n(13C only)", "1-D SSL\n(sort-match 1D)", "2-D SSL\n(sliced, ours)"]
    c_means = [agg[v]["c_mean"] for v in variants]
    c_stds = [agg[v]["c_std"] for v in variants]
    h_means = [agg[v]["h_mean"] for v in variants]
    h_stds = [agg[v]["h_std"] for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    x = np.arange(len(variants))
    colors = [BLUE, ORANGE, GREEN]

    ax = axes[0]
    bars = ax.bar(x, c_means, yerr=c_stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C test error")
    ax.set_ylim(0, max(c_means) * 1.25)
    for i, (m, s) in enumerate(zip(c_means, c_stds)):
        ax.text(i, m + s + 0.15, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, h_means, yerr=h_stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H test error")
    ax.set_ylim(0, max(h_means) * 1.25)
    for i, (m, s) in enumerate(zip(h_means, h_stds)):
        ax.text(i, m + s + 0.07, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = OUT / "fig_mae_bars.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "fig_mae_bars.pdf", bbox_inches="tight")
    print(f"wrote {out}")


def fig_chem_scatter():
    with (ROOT / "experiments" / "results_2d" / "chemistry_demo.json").open() as f:
        demo = json.load(f)
    q_h = demo["h_quantile_ppm"]
    q_c = demo["c_quantile_ppm"]
    obs_h, pred_h, obs_c, pred_c = [], [], [], []
    for m in demo["demos"]:
        for (oh, oc), (ph, pc) in zip(m["hsqc_observed"], m["hsqc_predicted"]):
            obs_h.append(oh); pred_h.append(ph)
            obs_c.append(oc); pred_c.append(pc)
    obs_h = np.array(obs_h); pred_h = np.array(pred_h)
    obs_c = np.array(obs_c); pred_c = np.array(pred_c)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    ax = axes[0]
    lo = min(obs_c.min(), pred_c.min()) - 5
    hi = max(obs_c.max(), pred_c.max()) + 5
    xs = np.linspace(lo, hi, 100)
    ax.fill_between(xs, xs - q_c, xs + q_c, color=GREEN, alpha=0.12,
                    label=f"95% conformal band ({q_c:.1f} ppm)")
    ax.plot(xs, xs, "k--", lw=0.8, alpha=0.6, label="y = x")
    ax.scatter(obs_c, pred_c, s=30, c=GREEN, edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed $^{13}$C shift (ppm)")
    ax.set_ylabel("Predicted $^{13}$C shift (ppm)")
    ax.set_title("(a) $^{13}$C  (5 demo molecules)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal")

    ax = axes[1]
    lo = min(obs_h.min(), pred_h.min()) - 0.3
    hi = max(obs_h.max(), pred_h.max()) + 0.3
    xs = np.linspace(lo, hi, 100)
    ax.fill_between(xs, xs - q_h, xs + q_h, color=ORANGE, alpha=0.12,
                    label=f"95% conformal band ({q_h:.2f} ppm)")
    ax.plot(xs, xs, "k--", lw=0.8, alpha=0.6, label="y = x")
    ax.scatter(obs_h, pred_h, s=30, c=ORANGE, edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed $^{1}$H shift (ppm)")
    ax.set_ylabel("Predicted $^{1}$H shift (ppm)")
    ax.set_title("(b) $^{1}$H  (5 demo molecules)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_aspect("equal")

    fig.tight_layout()
    out = OUT / "fig_chem_scatter.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "fig_chem_scatter.pdf", bbox_inches="tight")
    print(f"wrote {out}")


def fig_overview():
    """Schematic: 1-D sort-match vs sliced 2-D sort-match (conceptual)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    rng = np.random.default_rng(0)
    n = 8

    ax = axes[0]
    pred = rng.uniform(0, 10, n)
    targ = pred + rng.normal(0, 0.6, n)
    ax.scatter(np.zeros(n), pred, s=60, c=BLUE, label="$\\hat{\\delta}$", zorder=3)
    ax.scatter(np.ones(n), targ, s=60, c=ORANGE, marker="s", label="$\\delta^\\star$", zorder=3)
    for p, t in zip(np.sort(pred), np.sort(targ)):
        ax.plot([0, 1], [p, t], "gray", lw=0.8, alpha=0.6)
    ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["pred", "target"])
    ax.set_ylabel("shift (ppm)")
    ax.set_title("(a) 1-D sort-match\n[Paper 1]")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    pH = rng.uniform(0, 8, n); pC = rng.uniform(20, 150, n)
    tH = pH + rng.normal(0, 0.4, n); tC = pC + rng.normal(0, 5, n)
    ax.scatter(pH, pC, s=60, c=BLUE, label="$\\hat{P}$", zorder=3)
    ax.scatter(tH, tC, s=60, c=ORANGE, marker="s", label="$P^\\star$", zorder=3)
    # draw some projection directions
    center = np.array([4, 85])
    for k, theta in enumerate(np.linspace(0, np.pi, 4, endpoint=False)):
        d = np.array([np.cos(theta), np.sin(theta) * 20])
        ax.plot([center[0] - d[0], center[0] + d[0]],
                [center[1] - d[1], center[1] + d[1]],
                "gray", lw=0.6, alpha=0.5)
    ax.set_xlabel("$^{1}$H shift (ppm)")
    ax.set_ylabel("$^{13}$C shift (ppm)")
    ax.set_title("(b) Sliced 2-D sort-match\n(Theorem 2, ours)")
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out = OUT / "fig_overview.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_mae_bars()
    fig_chem_scatter()
    fig_overview()
    print(f"all figures in {OUT}")
