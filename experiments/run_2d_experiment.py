"""End-to-end 2-D NMR experiment: supervised / 1D-SSL / 2D-SSL comparison.

Runs three variants of the 2-D NMR SSL experiment with multi-seed statistics.
Each variant uses the same dataset, model, optimizer, and labeled/unlabeled
split; only the loss on the unlabeled portion changes.

Usage:
    python3 experiments/run_2d_experiment.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.data_2d import HSQCMolecule, build_hsqc_molecules
from src.nmr2d.train_2d import (
    Config2D,
    HSQCDataset,
    compute_target_stats,
    train_variant,
)


def split_indices(n: int, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--max-atoms", type=int, default=60)
    parser.add_argument("--labeled-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--ssl-weight", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[2d-exp] parsing SDF")
    t0 = time.time()
    molecules = build_hsqc_molecules(args.sdf, max_records=args.max_records, max_atoms=args.max_atoms)
    print(f"  kept {len(molecules)} HSQC molecules in {time.time()-t0:.1f}s")
    dataset = HSQCDataset(molecules)

    variants = ["supervised_1d", "sort_match_ssl_1d", "sort_match_ssl_2d"]
    per_seed = {}

    for seed in args.seeds:
        print(f"\n========== seed {seed} ==========")
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        print(f"  c: mean={c_mean:.2f} std={c_std:.2f}  h: mean={h_mean:.2f} std={h_std:.2f}")

        rng = random.Random(seed + 1)
        train_shuffled = train_idx.copy()
        rng.shuffle(train_shuffled)
        n_lab = max(1, int(len(train_shuffled) * args.labeled_frac))
        labeled = train_shuffled[:n_lab]
        unlabeled = train_shuffled[n_lab:]
        print(f"  labeled={len(labeled)} unlabeled={len(unlabeled)}")

        per_seed[seed] = {}
        for variant in variants:
            print(f"\n--- variant: {variant} ---")
            cfg = Config2D(
                variant=variant,
                hidden=args.hidden,
                n_layers=args.n_layers,
                epochs=args.epochs,
                labeled_frac=args.labeled_frac,
                ssl_weight=args.ssl_weight,
                K_directions=args.K,
                seed=seed,
                c_mean=c_mean,
                c_std=c_std,
                h_mean=h_mean,
                h_std=h_std,
            )
            log_path = args.out / f"seed_{seed}" / f"{variant}.json"
            result = train_variant(
                cfg, dataset, train_idx, val_idx, test_idx,
                labeled_indices=labeled, unlabeled_indices=unlabeled,
                log_path=log_path,
            )
            per_seed[seed][variant] = {
                "test_c_mae": result["test_c_mae"],
                "test_h_mae": result["test_h_mae"],
                "elapsed_sec": result["elapsed_sec"],
            }
            print(f"  {variant}: test C MAE = {result['test_c_mae']:.3f}  H MAE = {result['test_h_mae']:.3f}  ({result['elapsed_sec']:.0f}s)")

    # Aggregate across seeds
    agg = {}
    for v in variants:
        c_vals = [per_seed[s][v]["test_c_mae"] for s in args.seeds]
        h_vals = [per_seed[s][v]["test_h_mae"] for s in args.seeds]
        agg[v] = {
            "c_mean": float(np.mean(c_vals)),
            "c_std": float(np.std(c_vals)),
            "h_mean": float(np.mean(h_vals)),
            "h_std": float(np.std(h_vals)),
            "c_values": c_vals,
            "h_values": h_vals,
        }

    summary = {
        "args": vars(args) | {"sdf": str(args.sdf), "out": str(args.out)},
        "n_molecules": len(molecules),
        "per_seed": per_seed,
        "aggregate": agg,
    }
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n============ 2D EXPERIMENT SUMMARY ============")
    print(f"  {'variant':25s}  {'13C MAE':>12s}  {'1H MAE':>12s}")
    print("  " + "-" * 53)
    for v in variants:
        a = agg[v]
        print(f"  {v:25s}  {a['c_mean']:.3f} ± {a['c_std']:.3f}  {a['h_mean']:.3f} ± {a['h_std']:.3f}")
    sup = agg["supervised_1d"]["c_mean"]
    sm1 = agg["sort_match_ssl_1d"]["c_mean"]
    sm2 = agg["sort_match_ssl_2d"]["c_mean"]
    print(f"\n  1D SSL vs Supervised: {(sup-sm1)/sup*100:+.1f}% relative 13C improvement")
    print(f"  2D SSL vs Supervised: {(sup-sm2)/sup*100:+.1f}% relative 13C improvement")
    print(f"  2D SSL vs 1D SSL   : {(sm1-sm2)/sm1*100:+.1f}% relative 13C improvement")


if __name__ == "__main__":
    main()
