"""Ablation: sweep labeled_frac and compare supervised vs sort_match_ssl.

For each labeled fraction, trains all three variants with a shared dataset
split and saves per-fraction summary JSON files under
``experiments/results_ablation/frac_XXXX/``.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import NMRDataset, iter_nmrshiftdb2_sdf, mol_to_graph_tensors  # noqa: E402
from src.train import TrainConfig, set_labeled_cache, train_one_variant  # noqa: E402
from experiments.run_ssl_experiment import filter_valid, split_indices  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_ablation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] parsing SDF...")
    raw = list(iter_nmrshiftdb2_sdf(args.sdf, nucleus="13C", max_records=args.max_records))
    kept = filter_valid(raw)
    print(f"  {len(kept)} clean molecules")

    train_idx, val_idx, test_idx = split_indices(len(kept), 0.8, 0.1, args.seed)
    all_train_shifts = []
    for i in train_idx:
        t = kept[i].target_for_nucleus()
        if t is not None:
            all_train_shifts.extend(t[0])
    target_mean = float(np.mean(all_train_shifts))
    target_std = float(np.std(all_train_shifts))
    print(f"  target stats mean={target_mean:.2f}  std={target_std:.2f}")

    full_dataset = NMRDataset(kept)
    val_dataset = NMRDataset([kept[i] for i in val_idx])
    test_dataset = NMRDataset([kept[i] for i in test_idx])

    rng = random.Random(args.seed + 1)
    train_shuffled = train_idx.copy()
    rng.shuffle(train_shuffled)

    all_results = {}
    for frac in args.fractions:
        print(f"\n=== labeled_frac = {frac} ===")
        n_labeled = max(1, int(len(train_shuffled) * frac))
        labeled_indices = train_shuffled[:n_labeled]
        unlabeled_indices = train_shuffled[n_labeled:]
        labeled_ids = {kept[i].nmrshift_id for i in labeled_indices}
        set_labeled_cache(labeled_ids)
        print(f"  labeled={n_labeled}  unlabeled={len(unlabeled_indices)}")

        frac_results = {}
        for variant in ("supervised", "sort_match_ssl"):
            cfg = TrainConfig(
                variant=variant,
                hidden=128,
                n_layers=4,
                epochs=args.epochs,
                batch_size=32,
                labeled_frac=frac,
                ssl_weight=0.5,
                seed=args.seed,
                target_mean=target_mean,
                target_std=target_std,
            )
            sub = args.out / f"frac_{int(frac * 1000):04d}"
            sub.mkdir(parents=True, exist_ok=True)
            log_path = sub / f"{variant}.json"
            result = train_one_variant(
                cfg,
                train_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                in_dim=20,
                log_path=log_path,
            )
            frac_results[variant] = result
            print(f"  [{variant}] test MAE = {result['test_mae']:.3f} ppm  ({result['elapsed_sec']:.0f}s)")

        summary_path = args.out / f"frac_{int(frac * 1000):04d}" / "summary.json"
        with summary_path.open("w") as f:
            json.dump(
                {
                    "labeled_frac": frac,
                    "n_labeled": n_labeled,
                    "n_unlabeled": len(unlabeled_indices),
                    "results": {v: {k: r[k] for k in ("test_mae", "best_val_mae", "elapsed_sec")} for v, r in frac_results.items()},
                },
                f,
                indent=2,
            )
        all_results[frac] = frac_results

    print("\n[ablation] SUMMARY:")
    print(f"  {'labeled_frac':>14s}  {'supervised':>12s}  {'sort_match':>12s}  {'improvement':>14s}")
    for frac in args.fractions:
        if frac not in all_results:
            continue
        r = all_results[frac]
        sup = r["supervised"]["test_mae"]
        sm = r["sort_match_ssl"]["test_mae"]
        imp = (sup - sm) / sup * 100 if sup > 0 else 0
        print(f"  {frac:14.3f}  {sup:12.3f}  {sm:12.3f}  {imp:13.1f}%")

    combined = args.out / "combined_summary.json"
    with combined.open("w") as f:
        json.dump(
            {
                "args": {"sdf": str(args.sdf), "max_records": args.max_records, "fractions": args.fractions},
                "target_mean": target_mean,
                "target_std": target_std,
                "by_fraction": {
                    str(frac): {
                        v: {k: r[k] for k in ("test_mae", "best_val_mae", "elapsed_sec")}
                        for v, r in res.items()
                    }
                    for frac, res in all_results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"  combined summary saved to {combined}")


if __name__ == "__main__":
    main()
