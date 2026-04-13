"""Full experiment suite for the Nature CS push.

Runs, in sequence, all the experiments a Nature CS reviewer will demand:

  A. Main experiment, 3 seeds, labeled_frac=0.1, 13C, random split
  B. Scaffold-split OOD experiment, 3 seeds, labeled_frac=0.1, 13C
  C. 1H experiment, 3 seeds, labeled_frac=0.1, random split

Each writes results under experiments/results_suite/<name>/seed_<s>/.
A combined summary JSON is written to experiments/results_suite/suite_summary.json.

Naive SSL is kept in the main experiment (already established as a strawman)
but dropped from scaffold / 1H runs to save compute. The decisive comparison is
always supervised vs sort_match_ssl.
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

from src.data import (  # noqa: E402
    NMRDataset,
    iter_nmrshiftdb2_sdf,
    mol_to_graph_tensors,
    scaffold_split,
)
from src.train import TrainConfig, set_labeled_cache, train_one_variant  # noqa: E402
from experiments.run_ssl_experiment import filter_valid, split_indices  # noqa: E402


def build_dataset(
    sdf: Path, nucleus: str, max_records: int, max_atoms: int
):
    print(f"[data] parsing {sdf.name} for {nucleus}, max={max_records}")
    t0 = time.time()
    raw = list(iter_nmrshiftdb2_sdf(sdf, nucleus=nucleus, max_records=max_records))
    kept = filter_valid(raw, max_atoms=max_atoms)
    print(f"  parsed {len(raw)}, kept {len(kept)} clean molecules in {time.time() - t0:.1f}s")
    return kept


def build_split(
    kept, *, mode: str, seed: int, train_frac: float = 0.8, val_frac: float = 0.1
):
    if mode == "random":
        return split_indices(len(kept), train_frac, val_frac, seed)
    if mode == "scaffold":
        return scaffold_split(kept, train_frac=train_frac, val_frac=val_frac, seed=seed)
    raise ValueError(mode)


def compute_target_stats(kept, train_idx):
    shifts = []
    for i in train_idx:
        t = kept[i].target_for_nucleus()
        if t is not None:
            shifts.extend(t[0])
    return float(np.mean(shifts)), float(np.std(shifts))


def run_one_config(
    *,
    name: str,
    out_dir: Path,
    kept,
    split_mode: str,
    seeds: list[int],
    variants: list[str],
    labeled_frac: float,
    epochs: int,
    hidden: int,
    n_layers: int,
    in_dim: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed: dict[int, dict[str, dict]] = {}

    for seed in seeds:
        print(f"\n[{name}] === seed {seed} ===")
        train_idx, val_idx, test_idx = build_split(
            kept, mode=split_mode, seed=seed
        )
        print(f"  split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        target_mean, target_std = compute_target_stats(kept, train_idx)

        full_dataset = NMRDataset(kept)
        val_dataset = NMRDataset([kept[i] for i in val_idx])
        test_dataset = NMRDataset([kept[i] for i in test_idx])

        rng = random.Random(seed + 1)
        train_shuffled = list(train_idx)
        rng.shuffle(train_shuffled)
        n_lab = max(1, int(len(train_shuffled) * labeled_frac))
        labeled_idx = train_shuffled[:n_lab]
        unlabeled_idx = train_shuffled[n_lab:]
        labeled_ids = {kept[i].nmrshift_id for i in labeled_idx}
        set_labeled_cache(labeled_ids)
        print(f"  labeled={n_lab} unlabeled={len(unlabeled_idx)}")

        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_variant = {}

        for variant in variants:
            cfg = TrainConfig(
                variant=variant,
                hidden=hidden,
                n_layers=n_layers,
                epochs=epochs,
                batch_size=32,
                labeled_frac=labeled_frac,
                ssl_weight=0.5,
                seed=seed,
                target_mean=target_mean,
                target_std=target_std,
            )
            log_path = seed_dir / f"{variant}.json"
            result = train_one_variant(
                cfg,
                train_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                labeled_indices=labeled_idx,
                unlabeled_indices=unlabeled_idx,
                in_dim=in_dim,
                log_path=log_path,
            )
            per_variant[variant] = {
                "test_mae": result["test_mae"],
                "best_val_mae": result["best_val_mae"],
                "elapsed_sec": result["elapsed_sec"],
            }
            print(f"  [{variant}] test MAE = {result['test_mae']:.3f} ppm")

        per_seed[seed] = per_variant

    summary = {
        "name": name,
        "split_mode": split_mode,
        "labeled_frac": labeled_frac,
        "epochs": epochs,
        "hidden": hidden,
        "n_layers": n_layers,
        "seeds": seeds,
        "variants": variants,
        "per_seed": per_seed,
    }

    # Aggregate mean / std across seeds per variant
    agg = {}
    for variant in variants:
        vals = [per_seed[s][variant]["test_mae"] for s in seeds]
        agg[variant] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "values": vals,
        }
    summary["aggregate"] = agg

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{name}] aggregate test MAE (mean ± std across {len(seeds)} seeds):")
    for variant, a in agg.items():
        print(f"  {variant:20s} {a['mean']:.3f} ± {a['std']:.3f} ppm")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--max-records", type=int, default=10000)
    parser.add_argument("--max-atoms", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--labeled-frac", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_suite")
    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="Experiments to skip: main / scaffold / h1",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Shared atom feature dim is set once we load data
    kept_13c = build_dataset(args.sdf, "13C", args.max_records, args.max_atoms)
    x0, _ = mol_to_graph_tensors(kept_13c[0].mol)
    in_dim = x0.shape[1]
    print(f"[data] atom feature dim = {in_dim}")

    full_summary: dict = {
        "args": {
            "sdf": str(args.sdf),
            "max_records": args.max_records,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "labeled_frac": args.labeled_frac,
            "seeds": args.seeds,
        }
    }

    if "main" not in args.skip:
        print("\n########################################")
        print("# Experiment A: main (random split, 13C)")
        print("########################################")
        full_summary["A_main"] = run_one_config(
            name="A_main",
            out_dir=args.out / "A_main",
            kept=kept_13c,
            split_mode="random",
            seeds=args.seeds,
            variants=["supervised", "naive_ssl", "sort_match_ssl"],
            labeled_frac=args.labeled_frac,
            epochs=args.epochs,
            hidden=args.hidden,
            n_layers=args.n_layers,
            in_dim=in_dim,
        )

    if "scaffold" not in args.skip:
        print("\n########################################")
        print("# Experiment B: scaffold split (OOD), 13C")
        print("########################################")
        full_summary["B_scaffold"] = run_one_config(
            name="B_scaffold",
            out_dir=args.out / "B_scaffold",
            kept=kept_13c,
            split_mode="scaffold",
            seeds=args.seeds,
            variants=["supervised", "sort_match_ssl"],
            labeled_frac=args.labeled_frac,
            epochs=args.epochs,
            hidden=args.hidden,
            n_layers=args.n_layers,
            in_dim=in_dim,
        )

    # Experiment C (1H) is deferred: NMRShiftDB2 stores 1H peaks indexed by
    # heavy atom with variable H multiplicity per atom, and the MVP filter
    # for non-degenerate cases retains almost nothing. A proper 1H experiment
    # requires AddHs expansion and partial-label handling. Left for follow-up.

    with (args.out / "suite_summary.json").open("w") as f:
        json.dump(full_summary, f, indent=2)

    print("\n######## FULL SUITE COMPLETE ########")
    for name in ("A_main", "B_scaffold", "C_1h"):
        if name not in full_summary:
            continue
        summary = full_summary[name]
        print(f"\n{name} ({summary['split_mode']} split):")
        for variant, a in summary["aggregate"].items():
            print(f"  {variant:20s} {a['mean']:.3f} ± {a['std']:.3f} ppm  (n={a['n']})")


if __name__ == "__main__":
    main()
