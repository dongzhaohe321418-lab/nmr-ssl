"""Reviewer figure: 13C / 1H test MAE decomposed by (carbon) type.

Reads experiments/results_2d/error_decomposition.json and emits a
two-panel bar chart sorted descending by MAE, with n= annotations on
each bar. Matches the publication style from make_2d_figures.py
(serif fonts, DPI 300, PNG + PDF).
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

GREEN = "#2ca02c"
ORANGE = "#ff7f0e"

# Pretty labels for the raw dictionary keys in error_decomposition.json.
C_LABELS = {
    "aromatic": "aromatic",
    "carbonyl/imino": "C=O / C=N",
    "olefinic": "olefinic",
    "sp3_CH3": r"sp$^3$ CH$_3$",
    "sp3_CH2": r"sp$^3$ CH$_2$",
    "sp3_CH": r"sp$^3$ CH",
    "sp3_quat": r"sp$^3$ quat.",
    "sp3_other": r"sp$^3$ other",
}
H_LABELS = {
    "aromatic": "aromatic",
    "carbonyl/imino": "C=O / C=N",
    "olefinic": "olefinic",
    "sp3_H_CH3": r"sp$^3$ CH$_3$",
    "sp3_H_CH2": r"sp$^3$ CH$_2$",
    "sp3_H_CH": r"sp$^3$ CH",
    "sp3_H_quat": r"sp$^3$ quat.",
    "non_C_H": "non-C",
}


def _sorted_rows(block: dict) -> list[tuple[str, int, float]]:
    rows = [(t, d["n"], d["mae"]) for t, d in block.items()]
    rows.sort(key=lambda r: -r[2])
    return rows


def main():
    with (ROOT / "experiments" / "results_2d" / "error_decomposition.json").open() as f:
        data = json.load(f)

    c_rows = _sorted_rows(data["c_by_type"])
    h_rows = _sorted_rows(data["h_by_type"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    # 13C panel
    ax = axes[0]
    labels = [C_LABELS.get(r[0], r[0]) for r in c_rows]
    ns = [r[1] for r in c_rows]
    maes = [r[2] for r in c_rows]
    x = np.arange(len(labels))
    bars = ax.bar(x, maes, color=GREEN, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$^{13}$C test MAE (ppm)")
    ax.set_title(r"(a) $^{13}$C error by carbon type")
    ax.set_ylim(0, max(maes) * 1.22)
    for xi, (m, n) in enumerate(zip(maes, ns)):
        ax.text(
            xi,
            m + max(maes) * 0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    # 1H panel
    ax = axes[1]
    labels = [H_LABELS.get(r[0], r[0]) for r in h_rows]
    ns = [r[1] for r in h_rows]
    maes = [r[2] for r in h_rows]
    x = np.arange(len(labels))
    bars = ax.bar(x, maes, color=ORANGE, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$^{1}$H test MAE (ppm)")
    ax.set_title(r"(b) $^{1}$H error by carbon type")
    ax.set_ylim(0, max(maes) * 1.22)
    for xi, (m, n) in enumerate(zip(maes, ns)):
        ax.text(
            xi,
            m + max(maes) * 0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    fig.tight_layout()
    out_png = OUT / "fig_err_by_carbon_type.png"
    out_pdf = OUT / "fig_err_by_carbon_type.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
