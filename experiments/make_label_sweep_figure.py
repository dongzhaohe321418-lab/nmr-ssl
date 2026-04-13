"""Label-sweep data-efficiency figure."""

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

BLUE = "#1f77b4"; GREEN = "#2ca02c"; ORANGE = "#ff7f0e"


def main():
    with (ROOT / "experiments" / "results_2d" / "label_sweep.json").open() as f:
        d = json.load(f)
    fracs = sorted(float(k) for k in d["results"].keys())
    sup_c, sup_h, ssl_c, ssl_h, n_labs = [], [], [], [], []
    for f in fracs:
        r = d["results"][str(f)]
        sup_c.append(r["variants"]["supervised_1d"]["test_c_mae"])
        sup_h.append(r["variants"]["supervised_1d"]["test_h_mae"])
        ssl_c.append(r["variants"]["sort_match_ssl_2d"]["test_c_mae"])
        ssl_h.append(r["variants"]["sort_match_ssl_2d"]["test_h_mae"])
        n_labs.append(r["n_labeled"])

    pct = [f * 100 for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))

    ax = axes[0]
    ax.plot(pct, sup_c, "s--", color=BLUE, linewidth=1.8, markersize=8, label="Supervised-1D")
    ax.plot(pct, ssl_c, "o-",  color=GREEN, linewidth=2.0, markersize=9, label="2-D SSL (ours)")
    ax.set_xscale("log")
    ax.set_xticks(pct); ax.set_xticklabels([f"{p:g}%" for p in pct])
    ax.set_xlabel("Labeled-$^{13}$C fraction")
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C data efficiency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(pct, sup_h, "s--", color=BLUE, linewidth=1.8, markersize=8, label="Supervised-1D (random)")
    ax.plot(pct, ssl_h, "o-",  color=ORANGE, linewidth=2.0, markersize=9, label="2-D SSL (ours)")
    ax.set_xscale("log")
    ax.set_xticks(pct); ax.set_xticklabels([f"{p:g}%" for p in pct])
    ax.set_xlabel("Labeled-$^{13}$C fraction")
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H data efficiency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_label_sweep.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}/fig_label_sweep.png")


if __name__ == "__main__":
    main()
