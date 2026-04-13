"""Nature CS-style multi-panel figures.

Reads everything under experiments/results_{main,ablation,suite}/ and produces
the three figures a Nature CS paper needs:

  Figure 1: Conceptual + Theorem verification (panels a-c)
  Figure 2: Main result + training curves + error bars (panels a-c)
  Figure 3: Ablation across labeled fraction + scaffold vs random +
            1H comparison (panels a-c)

All figures use Nature's default typography (Helvetica-like DejaVu Sans,
7-10pt) and 2-column width (~180 mm for wide figures, ~89 mm for single).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

NATURE_SINGLE = 89 / 25.4   # inches — 89 mm single column
NATURE_DOUBLE = 183 / 25.4  # inches — 183 mm double column

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    }
)

COLORS = {
    "supervised": "#555555",
    "naive_ssl": "#D76843",
    "sort_match_ssl": "#2B6CB0",
}

LABELS = {
    "supervised": "Supervised",
    "naive_ssl": "Naive SSL",
    "sort_match_ssl": "Sort-match SSL",
}


def panel_label(ax, letter):
    ax.text(
        -0.13,
        1.05,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )


def load_json(path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def extended_data_sinkhorn(out_dir: Path):
    """Extended Data figure: Sinkhorn OT relaxation vs sort-match vs Hungarian."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from src.losses import hungarian_reference, sort_match_loss
    from tests.test_sinkhorn_comparison import sinkhorn_matching_loss

    torch.manual_seed(2026)
    n_trials = 40
    sort_errs = []
    sink_large = []
    sink_small = []
    for _ in range(n_trials):
        n = int(torch.randint(5, 20, (1,)).item())
        y_hat = (torch.randn(n) * 50).double()
        y_star = (torch.randn(n) * 50).double()
        hung = hungarian_reference(
            y_hat.unsqueeze(0), y_star.unsqueeze(0), kind="mse"
        ).item()
        sort_val = sort_match_loss(y_hat, y_star, kind="mse", reduction="none").item()
        cost = (y_hat.unsqueeze(1) - y_star.unsqueeze(0)) ** 2
        cs = cost.mean().item()
        sl = sinkhorn_matching_loss(y_hat, y_star, epsilon=cs * 1e-1, n_iter=300).item()
        ss = sinkhorn_matching_loss(y_hat, y_star, epsilon=cs * 1e-3, n_iter=500).item()
        denom = max(hung, 1e-12)
        sort_errs.append(abs(sort_val - hung) / denom + 1e-20)
        sink_large.append(abs(sl - hung) / denom + 1e-20)
        sink_small.append(abs(ss - hung) / denom + 1e-20)

    fig, ax = plt.subplots(figsize=(NATURE_SINGLE, 2.6), constrained_layout=True)
    parts = ax.boxplot(
        [sort_errs, sink_large, sink_small],
        labels=["Sort-match\n(ours)", "Sinkhorn\nε/C=0.1", "Sinkhorn\nε/C=1e-3"],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
    )
    colors = [COLORS["sort_match_ssl"], "#D76843", "#D76843"]
    for p, c in zip(parts["boxes"], colors):
        p.set_facecolor(c)
        p.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Relative error vs Hungarian")
    ax.set_title("Sinkhorn OT is not a drop-in replacement", pad=2, fontsize=8.5)
    ax.axhline(2 ** -52, color="#888", ls=":", lw=0.8, label="float64 eps")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    ax.grid(axis="y", alpha=0.3, ls="--")
    fig.savefig(out_dir / "fig_ed1_sinkhorn.pdf")
    fig.savefig(out_dir / "fig_ed1_sinkhorn.png")
    plt.close(fig)
    print("  wrote fig_ed1_sinkhorn.pdf/png")


def figure1_theorem(out_dir: Path, main_results_dir: Path):
    """Figure 1: (a) schematic idea, (b) theorem verification test results,
    (c) training-time comparison Hungarian vs sort.

    Currently generates (b) and (c) from real code; (a) is a placeholder
    schematic that a designer would replace.
    """
    fig = plt.figure(figsize=(NATURE_DOUBLE, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel a — conceptual schematic of sort-match
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    panel_label(ax_a, "a")
    ax_a.set_title("Sort-match set supervision", pad=2)

    # Predicted and target tick marks
    pred = [1.5, 2.4, 4.1, 5.3, 6.8]
    targ = [1.8, 3.0, 4.6, 5.8, 6.1]
    for y_off, vals, color, lab in [(7.2, pred, COLORS["sort_match_ssl"], "predicted"),
                                    (2.4, targ, "#444444", "unassigned target")]:
        ax_a.plot([v for v in vals], [y_off] * len(vals), "|", color=color, ms=14, mew=2)
        ax_a.text(0.3, y_off + 0.6, lab, fontsize=7, color=color)
        ax_a.annotate(
            "", xy=(9.5, y_off), xytext=(0.5, y_off),
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
        )
    # Connect matched pairs (after sort)
    pred_sorted = sorted(pred)
    targ_sorted = sorted(targ)
    for p, t in zip(pred_sorted, targ_sorted):
        ax_a.plot([p, t], [7.2, 2.4], "-", color=COLORS["sort_match_ssl"], alpha=0.6, lw=0.8)
    ax_a.text(5.0, 0.6, "Sort both; pair in order.",
              ha="center", fontsize=7, style="italic", color="#444444")
    ax_a.text(5.0, 9.3, "Optimal matching under convex cost",
              ha="center", fontsize=7.5, fontweight="bold")

    # Panel b — numerical verification of theorem (real data)
    # We regenerate the test statistics by running the test directly.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from src.losses import hungarian_reference, sort_match_loss

    torch.manual_seed(0)
    ns = [2, 4, 8, 12, 16, 20, 24]
    rels = {k: [] for k in ("mae", "mse", "huber")}
    for n in ns:
        for kind in rels:
            trials = []
            for _ in range(30):
                a = (torch.randn(n) * 50).double()
                b = (torch.randn(n) * 50).double()
                s = sort_match_loss(a, b, kind=kind, reduction="none").item()
                h = hungarian_reference(a.unsqueeze(0), b.unsqueeze(0), kind=kind).item()
                denom = max(abs(s), abs(h), 1e-12)
                trials.append(abs(s - h) / denom)
            rels[kind].append(max(trials))

    for kind in ("mae", "mse", "huber"):
        ax_b.plot(ns, [max(v, 1e-18) for v in rels[kind]], "o-", label=kind.upper(), ms=4)
    ax_b.set_yscale("log")
    ax_b.set_ylim(1e-18, 1e-5)
    ax_b.set_xlabel("Set size n")
    ax_b.set_ylabel("Max relative error")
    ax_b.axhline(2 ** -52, color="#888888", ls=":", lw=0.8, label="float64 eps")
    ax_b.legend(loc="upper right", frameon=False)
    ax_b.set_title("Sort = Hungarian (numerical)", pad=2)
    panel_label(ax_b, "b")
    ax_b.grid(alpha=0.3, ls="--")

    # Panel c — complexity illustration: sort O(n log n) vs Hungarian O(n^3)
    xs = np.array(ns, dtype=float)
    sort_cost = xs * np.log2(xs)
    hung_cost = xs ** 3
    ax_c.plot(xs, sort_cost / sort_cost[0], "o-", label="Sort O(n log n)",
              color=COLORS["sort_match_ssl"])
    ax_c.plot(xs, hung_cost / hung_cost[0], "s-", label="Hungarian O(n³)",
              color=COLORS["naive_ssl"])
    ax_c.set_yscale("log")
    ax_c.set_xlabel("Set size n")
    ax_c.set_ylabel("Relative cost")
    ax_c.legend(loc="upper left", frameon=False)
    ax_c.set_title("Asymptotic cost", pad=2)
    ax_c.grid(alpha=0.3, ls="--")
    panel_label(ax_c, "c")

    fig.savefig(out_dir / "fig1_theorem.pdf")
    fig.savefig(out_dir / "fig1_theorem.png")
    plt.close(fig)
    print("  wrote fig1_theorem.pdf/png")


def figure2_main(out_dir: Path, main_dir: Path, suite_dir: Path):
    """Figure 2: main empirical result.

    (a) Training curves from the single-seed main experiment
    (b) Multi-seed error-bar comparison (from suite A_main)
    (c) Table / annotation of numerical results
    """
    fig = plt.figure(figsize=(NATURE_DOUBLE, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.1, 0.9])
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel a — training curves (from main results)
    for variant in ("supervised", "naive_ssl", "sort_match_ssl"):
        data = load_json(main_dir / f"{variant}.json")
        if data is None:
            continue
        hist = data["history"]
        xs = [h["epoch"] + 1 for h in hist]
        ys = [h["val_mae"] for h in hist]
        ax_a.plot(xs, ys, "-", color=COLORS[variant], label=LABELS[variant], lw=1.5)
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Validation MAE (ppm, $^{13}$C)")
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.grid(alpha=0.3, ls="--")
    ax_a.set_title("Training dynamics", pad=2)
    panel_label(ax_a, "a")

    # Panel b — multi-seed bar with error bars (from suite A_main)
    suite_summary = load_json(suite_dir / "A_main" / "summary.json")
    if suite_summary is None:
        # Fall back to single-seed
        sup = load_json(main_dir / "supervised.json")
        naive = load_json(main_dir / "naive_ssl.json")
        sm = load_json(main_dir / "sort_match_ssl.json")
        means = [sup["test_mae"], naive["test_mae"], sm["test_mae"]]
        stds = [0, 0, 0]
        ns_note = "n=1 seed"
    else:
        agg = suite_summary["aggregate"]
        means = [agg[v]["mean"] for v in ("supervised", "naive_ssl", "sort_match_ssl")]
        stds = [agg[v]["std"] for v in ("supervised", "naive_ssl", "sort_match_ssl")]
        ns_note = f"n={agg['supervised']['n']} seeds"

    xs = np.arange(3)
    colors = [COLORS["supervised"], COLORS["naive_ssl"], COLORS["sort_match_ssl"]]
    bars = ax_b.bar(xs, means, yerr=stds, color=colors, width=0.6, capsize=4)
    for bar, m in zip(bars, means):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(stds) * 0.3 + 0.1,
                  f"{m:.2f}",
                  ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(["Supervised", "Naive SSL", "Sort-match\n(ours)"], fontsize=7)
    ax_b.set_ylabel("Test MAE (ppm, $^{13}$C)")
    ax_b.set_title(f"Main comparison ({ns_note})", pad=2)
    ax_b.grid(axis="y", alpha=0.3, ls="--")
    ax_b.set_ylim(0, max(means) * 1.3 + 1)
    panel_label(ax_b, "b")

    # Panel c — text panel summarizing numerical improvement
    ax_c.axis("off")
    panel_label(ax_c, "c")
    # Simple text
    ax_c.set_title("Improvement", pad=2)
    if len(means) == 3:
        sup_m, _, sm_m = means
        rel_gain = (sup_m - sm_m) / sup_m * 100
        abs_gain = sup_m - sm_m
        lines = [
            f"Supervised: {sup_m:.2f} ppm",
            f"Sort-match: {sm_m:.2f} ppm",
            "",
            f"Absolute: −{abs_gain:.2f} ppm",
            f"Relative: −{rel_gain:.1f}%",
            "",
            "Same model,",
            "same data,",
            "same compute.",
            "",
            "Only difference:",
            "sort-match loss",
            "on unlabeled data.",
        ]
        ax_c.text(
            0.0, 0.95,
            "\n".join(lines),
            transform=ax_c.transAxes,
            va="top", fontsize=7.5, family="monospace",
        )

    fig.savefig(out_dir / "fig2_main.pdf")
    fig.savefig(out_dir / "fig2_main.png")
    plt.close(fig)
    print("  wrote fig2_main.pdf/png")


def figure3_generalization(out_dir: Path, ablation_dir: Path, suite_dir: Path):
    """Figure 3: (a) Ablation over labeled fraction, (b) Scaffold vs random,
    (c) Robustness to corrupt unlabeled data."""
    fig = plt.figure(figsize=(NATURE_DOUBLE, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel a — ablation over labeled fraction
    # New format (overnight): single summary.json with "by_fraction" key holding aggregates
    # Old format (results_ablation): per-fraction subdirectories with summary.json files
    by_variant: dict[str, list[tuple[float, float, float]]] = {}
    overnight_summary = load_json(ablation_dir / "summary.json")
    if overnight_summary and "by_fraction" in overnight_summary:
        for frac_key, frac_data in overnight_summary["by_fraction"].items():
            frac = float(frac_key)
            for variant, agg in frac_data["aggregate"].items():
                by_variant.setdefault(variant, []).append(
                    (frac, agg["mean"], agg["std"])
                )
    else:
        # Legacy per-fraction directories
        fracs = [0.02, 0.05, 0.1, 0.2, 0.5]
        for f in fracs:
            sub = ablation_dir / f"frac_{int(f * 1000):04d}"
            summary = load_json(sub / "summary.json")
            if summary is None:
                continue
            for variant, res in summary["results"].items():
                by_variant.setdefault(variant, []).append(
                    (f, res["test_mae"], 0.0)
                )

    for variant, pts in by_variant.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [p[2] for p in pts]
        ax_a.errorbar(
            xs, ys, yerr=yerr,
            fmt="o-", color=COLORS.get(variant),
            label=LABELS.get(variant, variant),
            capsize=3, markersize=4,
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Labeled fraction")
    ax_a.set_ylabel("Test MAE (ppm, $^{13}$C)")
    ax_a.set_title("Low-label regime", pad=2)
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.grid(alpha=0.3, ls="--")
    panel_label(ax_a, "a")

    # Panel b — scaffold vs random (from suite)
    pairs = [("A_main", "Random split"), ("B_scaffold", "Scaffold split")]
    variants = ["supervised", "sort_match_ssl"]
    means = {v: [] for v in variants}
    stds = {v: [] for v in variants}
    xlabels = []
    for key, lbl in pairs:
        summary = load_json(suite_dir / key / "summary.json")
        if summary is None:
            continue
        for v in variants:
            means[v].append(summary["aggregate"][v]["mean"])
            stds[v].append(summary["aggregate"][v]["std"])
        xlabels.append(lbl)
    if xlabels:
        x = np.arange(len(xlabels))
        width = 0.36
        ax_b.bar(x - width/2, means["supervised"], width, yerr=stds["supervised"],
                 label="Supervised", color=COLORS["supervised"], capsize=3)
        ax_b.bar(x + width/2, means["sort_match_ssl"], width, yerr=stds["sort_match_ssl"],
                 label="Sort-match", color=COLORS["sort_match_ssl"], capsize=3)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(xlabels, fontsize=7)
        ax_b.set_ylabel("Test MAE (ppm, $^{13}$C)")
        ax_b.legend(loc="upper left", frameon=False)
        ax_b.grid(axis="y", alpha=0.3, ls="--")
    ax_b.set_title("OOD generalization", pad=2)
    panel_label(ax_b, "b")

    # Panel c — Robustness to corrupt unlabeled data (D_robustness)
    # and the solvent-conditioning result (E_solvent).
    # Top: robustness bar chart; bottom half: not drawn. Use a single axis.
    rob_summary = load_json(suite_dir / "D_robustness" / "summary.json")
    if rob_summary and "per_corruption" in rob_summary:
        corr_order = [
            ("clean", "Clean"),
            ("noise_1ppm", "1 ppm\nnoise"),
            ("drop_15", "15%\ndrop"),
            ("noise_1ppm_drop_10_spurious_10", "Combined\ncorrupt"),
        ]
        means_c = []
        stds_c = []
        labels_c = []
        for key, lab in corr_order:
            if key not in rob_summary["per_corruption"]:
                continue
            d = rob_summary["per_corruption"][key]
            means_c.append(d["mean"])
            stds_c.append(d["std"])
            labels_c.append(lab)
        x = np.arange(len(labels_c))
        bars = ax_c.bar(
            x, means_c, yerr=stds_c,
            color=COLORS["sort_match_ssl"], capsize=3, width=0.6,
        )
        # Overlay the supervised-clean baseline as a horizontal line
        main_summary = load_json(suite_dir / "A_main" / "summary.json")
        if main_summary and "aggregate" in main_summary:
            sup_mean = main_summary["aggregate"]["supervised"]["mean"]
            ax_c.axhline(
                sup_mean, color=COLORS["supervised"],
                ls="--", lw=1.2, label=f"Supervised (clean)\n{sup_mean:.2f} ppm",
            )
        for bar, m, s in zip(bars, means_c, stds_c):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.02,
                f"{m:.2f}", ha="center", va="bottom", fontsize=6.5,
            )
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(labels_c, fontsize=6.5)
        ax_c.set_ylabel("Test MAE (ppm, $^{13}$C)")
        ax_c.legend(loc="upper left", frameon=False, fontsize=6.5)
        ax_c.grid(axis="y", alpha=0.3, ls="--")
    ax_c.set_title("Robustness (sort-match SSL)", pad=2)
    panel_label(ax_c, "c")

    fig.savefig(out_dir / "fig3_generalization.pdf")
    fig.savefig(out_dir / "fig3_generalization.png")
    plt.close(fig)
    print("  wrote fig3_generalization.pdf/png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("experiments/results_main"))
    parser.add_argument("--ablation", type=Path, default=Path("experiments/results_ablation"))
    parser.add_argument("--suite", type=Path, default=Path("experiments/results_suite"))
    parser.add_argument("--overnight", type=Path, default=Path("experiments/results_overnight"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Prefer overnight results when present; fall back to earlier runs.
    suite_for_figures = args.overnight if (args.overnight / "A_main" / "summary.json").exists() else args.suite
    ablation_for_figures = args.overnight / "C_ablation" if (args.overnight / "C_ablation" / "summary.json").exists() else args.ablation
    print(f"  suite source   : {suite_for_figures}")
    print(f"  ablation source: {ablation_for_figures}")

    figure1_theorem(args.out, args.main)
    figure2_main(args.out, args.main, suite_for_figures)
    figure3_generalization(args.out, ablation_for_figures, suite_for_figures)
    extended_data_sinkhorn(args.out)


if __name__ == "__main__":
    main()
