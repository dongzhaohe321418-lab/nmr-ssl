"""Scale-up experiment: re-run the 3 variants on ~5k molecules instead of 1.5k.

Relaxes two filters:
  - min_hsqc_peaks: 3 -> 1 (accept any molecule with at least one HSQC peak)
  - max_records: 20000 -> 50000 (search more of the NMRShiftDB2 dump)

Uses 30 epochs, 3 seeds like the main experiment. Purpose: demonstrate that
the 2-D SSL effect persists (or amplifies) at larger scale, and close some
of the 13C gap via more training signal.

Writes results_2d/scale_up.json.
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

from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.train_2d import Config2D, HSQCDataset, compute_target_stats, train_variant
from experiments.run_2d_experiment import split_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "scale_up.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-records", type=int, default=60000)
    parser.add_argument("--min-hsqc", type=int, default=1)
    parser.add_argument("--max-atoms", type=int, default=120)
    args = parser.parse_args()

    print(f"[scale-up] loading dataset (max_records={args.max_records}, max_atoms={args.max_atoms}, min_hsqc={args.min_hsqc})")
    molecules = build_hsqc_molecules(
        args.sdf,
        max_records=args.max_records,
        max_atoms=args.max_atoms,
        min_hsqc_peaks=args.min_hsqc,
    )
    print(f"  {len(molecules)} molecules  (vs. 1542 in main experiment)")
    dataset = HSQCDataset(molecules)

    variants = ["supervised_1d", "sort_match_ssl_1d", "sort_match_ssl_2d"]
    per_seed = {}
    for seed in args.seeds:
        print(f"\n========== scale-up seed {seed} ==========")
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        rng = random.Random(seed + 1)
        shuf = train_idx.copy()
        rng.shuffle(shuf)
        n_lab = max(1, int(len(shuf) * 0.1))
        labeled, unlabeled = shuf[:n_lab], shuf[n_lab:]
        print(f"  n_train={len(train_idx)}  labeled={len(labeled)}  unlabeled={len(unlabeled)}")
        per_seed[seed] = {}
        for v in variants:
            print(f"\n--- {v} ---")
            cfg = Config2D(
                variant=v, hidden=192, n_layers=4, epochs=args.epochs,
                labeled_frac=0.1, ssl_weight=0.5, K_directions=16,
                seed=seed, c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
            )
            log = args.out.parent / f"scale_up_seed{seed}" / f"{v}.json"
            result = train_variant(cfg, dataset, train_idx, val_idx, test_idx, labeled, unlabeled, log_path=log)
            per_seed[seed][v] = {
                "test_c_mae": result["test_c_mae"],
                "test_h_mae": result["test_h_mae"],
                "elapsed": result["elapsed_sec"],
            }
            print(f"  {v}: C {result['test_c_mae']:.3f}  H {result['test_h_mae']:.3f}  ({result['elapsed_sec']:.0f}s)")

    agg = {}
    for v in variants:
        c_vals = [per_seed[s][v]["test_c_mae"] for s in args.seeds]
        h_vals = [per_seed[s][v]["test_h_mae"] for s in args.seeds]
        agg[v] = {
            "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
            "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
            "c_values": c_vals, "h_values": h_vals,
        }

    summary = {
        "n_molecules": len(molecules),
        "per_seed": per_seed,
        "aggregate": agg,
        "args": {"seeds": args.seeds, "epochs": args.epochs, "max_records": args.max_records,
                 "max_atoms": args.max_atoms, "min_hsqc": args.min_hsqc},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n============ SCALE-UP SUMMARY ============")
    print(f"  {'variant':25s}  {'13C MAE':>14s}  {'1H MAE':>14s}")
    for v in variants:
        a = agg[v]
        print(f"  {v:25s}  {a['c_mean']:.3f} ± {a['c_std']:.3f}  {a['h_mean']:.3f} ± {a['h_std']:.3f}")


if __name__ == "__main__":
    main()
