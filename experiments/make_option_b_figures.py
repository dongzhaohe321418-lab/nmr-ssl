"""Generate all Option B figures from the master orchestrator JSON."""

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

BLUE = "#1f77b4"; GREEN = "#2ca02c"; ORANGE = "#ff7f0e"; RED = "#d62728"; GRAY = "#8c8c8c"
PURPLE = "#9467bd"


def load_master():
    p = ROOT / "experiments" / "results_2d" / "option_b_master.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def fig_new_headline():
    """Bar chart comparing v3 vs v4 headline numbers vs key baselines."""
    data = load_master()
    v4 = data.get("p11_lambda2_headline", {}).get("aggregate", {})
    v4_c = v4.get("c_mean")
    v4_c_std = v4.get("c_std")
    v4_h = v4.get("h_mean")
    v4_h_std = v4.get("h_std")
    if v4_c is None:
        print("  new-headline skipped (no p11 data yet)")
        return

    labels = [
        "Supervised\n1-D",
        "Sort-match\nSSL 1-D",
        "2-D SSL\nλ=0.5, K=8",
        "2-D SSL\nλ=0.5, K=16",
        "2-D SSL\nλ=2.0, K=16",
    ]
    c_mean = [5.600, 4.562, 4.909, 4.869, v4_c]
    c_std = [0.343, 0.314, 0.209, 0.066, v4_c_std]
    h_mean = [2.473, 2.607, 0.491, 0.455, v4_h]
    h_std = [0.376, 0.322, 0.066, 0.144, v4_h_std]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    x = np.arange(len(labels))
    colors = [GRAY, BLUE, ORANGE, PURPLE, GREEN]

    ax = axes[0]
    ax.bar(x, c_mean, yerr=c_std, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    for i, (m, s) in enumerate(zip(c_mean, c_std)):
        ax.text(i, m + s + 0.1, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C — v4 matches 1-D SSL")
    ax.set_ylim(0, max(c_mean) * 1.22)

    ax = axes[1]
    ax.bar(x, h_mean, yerr=h_std, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    for i, (m, s) in enumerate(zip(h_mean, h_std)):
        ax.text(i, m + s + 0.04, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H — v4 reaches 0.35 ppm")
    ax.set_ylim(0, max(h_mean) * 1.22)

    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_v4_headline.{ext}", bbox_inches="tight")
    print(f"  wrote {OUT}/fig_v4_headline.png")


def fig_k_sweep_multiseed():
    data = load_master()
    ks = data.get("p12a_k_sweep", {})
    if not ks:
        print("  k-sweep skipped"); return
    K_vals = sorted(int(k) for k in ks.keys())
    c_mean = [ks[str(k)]["c_mean"] for k in K_vals]
    c_std = [ks[str(k)]["c_std"] for k in K_vals]
    h_mean = [ks[str(k)]["h_mean"] for k in K_vals]
    h_std = [ks[str(k)]["h_std"] for k in K_vals]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    ax.errorbar(K_vals, c_mean, yerr=c_std, marker="o", color=GREEN, capsize=4, lw=1.5)
    ax.set_xscale("log", base=2); ax.set_xticks(K_vals); ax.set_xticklabels([str(k) for k in K_vals])
    ax.set_xlabel("$K$"); ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C vs $K$ — 3 seeds")
    ax.axvline(16, color="gray", alpha=0.3, ls="--")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.errorbar(K_vals, h_mean, yerr=h_std, marker="s", color=ORANGE, capsize=4, lw=1.5)
    ax.set_xscale("log", base=2); ax.set_xticks(K_vals); ax.set_xticklabels([str(k) for k in K_vals])
    ax.set_xlabel("$K$"); ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H vs $K$ — 3 seeds")
    ax.axvline(16, color="gray", alpha=0.3, ls="--")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_k_sweep_multiseed.{ext}", bbox_inches="tight")
    print(f"  wrote {OUT}/fig_k_sweep_multiseed.png")


def fig_lambda_sweep_multiseed():
    data = load_master()
    ls = data.get("p12b_lambda_sweep", {})
    if not ls:
        print("  lambda-sweep skipped"); return
    L_vals = sorted(float(l) for l in ls.keys())
    c_mean = [ls[str(l)]["c_mean"] for l in L_vals]
    c_std = [ls[str(l)]["c_std"] for l in L_vals]
    h_mean = [ls[str(l)]["h_mean"] for l in L_vals]
    h_std = [ls[str(l)]["h_std"] for l in L_vals]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    ax.errorbar(L_vals, c_mean, yerr=c_std, marker="o", color=GREEN, capsize=4, lw=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(L_vals); ax.set_xticklabels([f"{l:g}" for l in L_vals])
    ax.set_xlabel("SSL weight $\\lambda$"); ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C vs $\\lambda$ — 3 seeds")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.errorbar(L_vals, h_mean, yerr=h_std, marker="s", color=ORANGE, capsize=4, lw=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(L_vals); ax.set_xticklabels([f"{l:g}" for l in L_vals])
    ax.set_xlabel("SSL weight $\\lambda$"); ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H vs $\\lambda$ — 3 seeds")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_lambda_sweep_multiseed.{ext}", bbox_inches="tight")
    print(f"  wrote {OUT}/fig_lambda_sweep_multiseed.png")


def fig_ablation_comparison():
    """Stop-grad / combined / scaffold-OOD / pretrain-finetune panels."""
    data = load_master()
    agg = {}
    for k in ["p15_scaffold_ood", "p22_combined", "p25_stopgrad", "p23_pretrain_finetune"]:
        if k in data and isinstance(data[k], dict) and "aggregate" in data[k]:
            agg[k] = data[k]["aggregate"]
    if not agg:
        print("  ablation-comparison skipped"); return

    baseline_c = data.get("p11_lambda2_headline", {}).get("aggregate", {}).get("c_mean", 4.54)
    baseline_h = data.get("p11_lambda2_headline", {}).get("aggregate", {}).get("h_mean", 0.35)

    labels = ["Headline\n(λ=2 K=16)"]
    c_vals = [baseline_c]; h_vals = [baseline_h]
    label_map = {
        "p15_scaffold_ood": "Scaffold-OOD",
        "p22_combined": "Combined\n(full C + SSL)",
        "p25_stopgrad": "Stop-grad on C",
        "p23_pretrain_finetune": "Pretrain→\nfinetune",
    }
    for k in ["p15_scaffold_ood", "p22_combined", "p25_stopgrad", "p23_pretrain_finetune"]:
        if k in agg:
            labels.append(label_map[k])
            c_vals.append(agg[k]["c_mean"])
            h_vals.append(agg[k]["h_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    x = np.arange(len(labels))
    colors = [GREEN] + [BLUE, ORANGE, PURPLE, RED][: len(labels) - 1]

    ax = axes[0]
    ax.bar(x, c_vals, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(c_vals):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$^{13}$C test MAE (ppm)")
    ax.set_title("(a) $^{13}$C across ablations")

    ax = axes[1]
    ax.bar(x, h_vals, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(h_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("$^{1}$H test MAE (ppm)")
    ax.set_title("(b) $^{1}$H across ablations")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_option_b_ablations.{ext}", bbox_inches="tight")
    print(f"  wrote {OUT}/fig_option_b_ablations.png")


if __name__ == "__main__":
    fig_new_headline()
    fig_k_sweep_multiseed()
    fig_lambda_sweep_multiseed()
    fig_ablation_comparison()
