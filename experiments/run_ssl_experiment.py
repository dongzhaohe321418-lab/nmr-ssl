"""End-to-end experiment runner: three variants on NMRShiftDB2 13C.

Usage:
    python3 experiments/run_ssl_experiment.py \
        --sdf data/nmrshiftdb2.sd \
        --max-records 8000 \
        --labeled-frac 0.1 \
        --epochs 25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import NMRDataset, iter_nmrshiftdb2_sdf, mol_to_graph_tensors  # noqa: E402
from src.train import (  # noqa: E402
    TrainConfig,
    set_labeled_cache,
    train_one_variant,
)

IN_DIM = mol_to_graph_tensors.__globals__["_NUM_ATOM_FEATS"] if "_NUM_ATOM_FEATS" in mol_to_graph_tensors.__globals__ else 20


def filter_valid(molecules, max_atoms: int = 60, min_peaks: int = 3):
    """Keep only molecules where (a) n_atoms <= max_atoms (memory budget),
    (b) all peaks correspond to valid atoms of the target nucleus,
    (c) the peak count equals the number of atoms of the target element
        (no degeneracy), and (d) at least ``min_peaks`` peaks.
    """
    kept = []
    for m in molecules:
        if m.n_atoms > max_atoms:
            continue
        target = m.target_for_nucleus()
        if target is None:
            continue
        shifts, indices = target
        if len(shifts) < min_peaks:
            continue
        # Count atoms of the target element in the molecule
        element = "C" if m.nucleus == "13C" else "H"
        element_count = sum(1 for a in m.mol.GetAtoms() if a.GetSymbol() == element)
        if len(shifts) != element_count:
            continue  # degenerate environments or incomplete assignment
        kept.append(m)
    return kept


def split_indices(n: int, train_frac: float, val_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--max-atoms", type=int, default=60)
    parser.add_argument("--labeled-frac", type=float, default=0.1)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ssl-weight", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--nucleus", type=str, default="13C", choices=["13C", "1H"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Parsing SDF (max {args.max_records} records)...")
    t0 = time.time()
    raw = list(
        iter_nmrshiftdb2_sdf(args.sdf, nucleus=args.nucleus, max_records=args.max_records)
    )
    print(f"  parsed {len(raw)} spectra in {time.time() - t0:.1f}s")

    print("[2/5] Filtering to non-degenerate clean molecules...")
    kept = filter_valid(raw, max_atoms=args.max_atoms)
    print(f"  {len(kept)} kept after filter")
    if len(kept) < 200:
        print("  too few molecules — increase --max-records")
        sys.exit(2)

    print("[3/5] Building dataset splits...")
    train_idx, val_idx, test_idx = split_indices(
        len(kept), args.train_frac, args.val_frac, args.seed
    )
    print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # Compute target mean/std from the TRAIN split only (no leakage).
    all_train_shifts = []
    for i in train_idx:
        t = kept[i].target_for_nucleus()
        if t is not None:
            all_train_shifts.extend(t[0])
    target_mean = float(np.mean(all_train_shifts))
    target_std = float(np.std(all_train_shifts))
    print(f"  target shift stats: mean={target_mean:.2f}  std={target_std:.2f}")

    full_dataset = NMRDataset(kept)
    train_dataset = full_dataset
    val_dataset = NMRDataset([kept[i] for i in val_idx])
    test_dataset = NMRDataset([kept[i] for i in test_idx])

    # Shuffle the train split into a labeled / unlabeled partition.
    rng = random.Random(args.seed + 1)
    train_shuffled = train_idx.copy()
    rng.shuffle(train_shuffled)
    n_labeled = max(1, int(len(train_shuffled) * args.labeled_frac))
    labeled_indices = train_shuffled[:n_labeled]
    unlabeled_indices = train_shuffled[n_labeled:]
    labeled_ids = {kept[i].nmrshift_id for i in labeled_indices}
    set_labeled_cache(labeled_ids)
    print(f"  labeled={len(labeled_indices)}  unlabeled={len(unlabeled_indices)}")

    # Determine input dim from a single graph.
    x0, _ = mol_to_graph_tensors(kept[0].mol)
    in_dim = x0.shape[1]
    print(f"  atom feature dim = {in_dim}")

    print("[4/5] Training three variants...")
    variants = ["supervised", "naive_ssl", "sort_match_ssl"]
    all_results = {}
    for variant in variants:
        print(f"\n  === {variant} ===")
        cfg = TrainConfig(
            variant=variant,
            hidden=args.hidden,
            n_layers=args.n_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            labeled_frac=args.labeled_frac,
            ssl_weight=args.ssl_weight,
            seed=args.seed,
            target_mean=target_mean,
            target_std=target_std,
        )
        log_path = args.out / f"{variant}.json"
        result = train_one_variant(
            cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            in_dim=in_dim,
            log_path=log_path,
        )
        all_results[variant] = result
        print(f"  [{variant}] test MAE = {result['test_mae']:.3f} ppm  ({result['elapsed_sec']:.0f}s)")

    print("\n[5/5] Summary:")
    print(f"  labeled fraction  : {args.labeled_frac:.3f}  ({n_labeled}/{len(train_idx)})")
    for v, r in all_results.items():
        print(f"    {v:18s} test MAE = {r['test_mae']:.3f} ppm")

    summary_path = args.out / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "args": vars(args) | {"sdf": str(args.sdf), "out": str(args.out)},
                "n_kept": len(kept),
                "n_labeled": n_labeled,
                "n_unlabeled": len(unlabeled_indices),
                "results": {v: {k: r[k] for k in ("test_mae", "best_val_mae", "elapsed_sec")} for v, r in all_results.items()},
            },
            f,
            indent=2,
        )
    print(f"  summary saved to {summary_path}")


if __name__ == "__main__":
    main()
