"""Generate publication-quality figures from experiment results.

Reads experiment JSON logs from experiments/results_*/ and writes figures
to the figures/ directory. Uses matplotlib only, APA-style (no seaborn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

VARIANT_COLORS = {
    "supervised": "#888888",
    "naive_ssl": "#E07A5F",
    "sort_match_ssl": "#2C6E9C",
}

VARIANT_LABELS = {
    "supervised": "Supervised (labeled only)",
    "naive_ssl": "Naive SSL (wrong assignment)",
    "sort_match_ssl": "Sort-match SSL (ours)",
}


def load_main_results(results_dir: Path) -> dict[str, dict]:
    out = {}
    for variant in ("supervised", "naive_ssl", "sort_match_ssl"):
        p = results_dir / f"{variant}.json"
        if not p.exists():
            print(f"  missing {p}")
            continue
        with p.open() as f:
            out[variant] = json.load(f)
    return out


def fig_training_curves(
    results: dict[str, dict], out_path: Path, title: str | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    for variant, result in results.items():
        history = result["history"]
        xs = [h["epoch"] + 1 for h in history]
        ys = [h["val_mae"] for h in history]
        ax.plot(
            xs,
            ys,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant, None),
            lw=2,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE (ppm, $^{13}$C)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_test_mae_bar(results: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    variants = list(results.keys())
    values = [results[v]["test_mae"] for v in variants]
    colors = [VARIANT_COLORS.get(v, "#666666") for v in variants]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]
    bars = ax.bar(range(len(variants)), values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(
        ["Supervised", "Naive SSL", "Sort-match SSL\n(ours)"], fontsize=9
    )
    ax.set_ylabel("Test MAE (ppm, $^{13}$C)")
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def fig_ablation(
    ablation_dir: Path, out_path: Path, fractions: list[float]
) -> None:
    """Load multiple runs at different labeled fractions and plot the curve."""
    by_variant: dict[str, list[tuple[float, float]]] = {}
    for frac in fractions:
        sub = ablation_dir / f"frac_{int(frac * 1000):04d}"
        summary = sub / "summary.json"
        if not summary.exists():
            continue
        with summary.open() as f:
            data = json.load(f)
        for v, res in data["results"].items():
            by_variant.setdefault(v, []).append((frac, res["test_mae"]))

    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    for variant, points in by_variant.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant),
            lw=2,
            marker="o",
            markersize=5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Labeled fraction")
    ax.set_ylabel("Test MAE (ppm, $^{13}$C)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("experiments/results_main"))
    parser.add_argument(
        "--ablation", type=Path, default=Path("experiments/results_ablation")
    )
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)

    if args.main.exists():
        results = load_main_results(args.main)
        if results:
            fig_training_curves(results, args.figures / "fig_training_curves.pdf")
            fig_training_curves(results, args.figures / "fig_training_curves.png")
            fig_test_mae_bar(results, args.figures / "fig_test_mae.pdf")
            fig_test_mae_bar(results, args.figures / "fig_test_mae.png")
    else:
        print(f"  no main results at {args.main}")

    if args.ablation.exists():
        fractions = [0.02, 0.05, 0.1, 0.2, 0.5]
        fig_ablation(args.ablation, args.figures / "fig_ablation.pdf", fractions)
        fig_ablation(args.ablation, args.figures / "fig_ablation.png", fractions)
    else:
        print(f"  no ablation results at {args.ablation}")


if __name__ == "__main__":
    main()
