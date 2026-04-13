"""Labeled-fraction sweep: 1%, 5%, 10%, 20%, 50%.

Trains 2-D SSL at each label fraction (single seed, 20 epochs) to show the
data-efficiency curve. Reviewers asked: when does 2-D SSL win relative to
supervised-1D, and when does it saturate?
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
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "label_sweep.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.50])
    args = parser.parse_args()

    print(f"[label-sweep] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)

    results = {}
    for frac in args.fracs:
        rng = random.Random(args.seed + 1)
        shuf = train_idx.copy()
        rng.shuffle(shuf)
        n_lab = max(1, int(len(shuf) * frac))
        labeled = shuf[:n_lab]
        unlabeled = shuf[n_lab:]

        print(f"\n[label-sweep] frac={frac:.2f}  labeled={len(labeled)}  unlabeled={len(unlabeled)}")
        per_variant = {}
        for v in ["supervised_1d", "sort_match_ssl_2d"]:
            cfg = Config2D(
                variant=v, hidden=192, n_layers=4, epochs=args.epochs,
                labeled_frac=frac, ssl_weight=0.5, K_directions=8,
                seed=args.seed, c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
            )
            log = args.out.parent / f"label_sweep" / f"frac{frac:.2f}_{v}.json"
            t0 = time.time()
            r = train_variant(cfg, dataset, train_idx, val_idx, test_idx, labeled, unlabeled, log_path=log)
            dt = time.time() - t0
            per_variant[v] = {
                "test_c_mae": r["test_c_mae"], "test_h_mae": r["test_h_mae"], "elapsed": dt,
            }
            print(f"  {v}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}  ({dt:.0f}s)")
        results[str(frac)] = {"n_labeled": len(labeled), "variants": per_variant}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"seed": args.seed, "epochs": args.epochs, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
