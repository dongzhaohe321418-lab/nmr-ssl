"""Overnight experiment orchestrator for Nature CS-scale results.

Runs five stages in sequence with per-run checkpointing:

  Stage A (Main):        3 variants x 3 seeds x 1 labeled_frac (0.1)
  Stage B (Scaffold):    2 variants x 3 seeds x 1 labeled_frac (0.1)
  Stage C (Ablation):    2 variants x 3 seeds x 5 fractions {0.01..0.20}
  Stage D (Robustness):  4 corruption variants x 3 seeds
  Stage E (Solvent):     2 variants x 3 seeds

Settings:
  model: 256 hidden, 5 layers (upgrade from 128/4)
  data : ~25k clean molecules from full NMRShiftDB2 (upgrade from 10k)
  epochs: 35

Each individual result is written to its own JSON and skipped on resume
if already present. Full summary written at the end of each stage and
at the end of the whole overnight run.
"""

from __future__ import annotations

import argparse
import copy
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

from src.data import (  # noqa: E402
    NMRDataset,
    iter_nmrshiftdb2_sdf,
    mol_to_graph_tensors,
    scaffold_split,
)
from src.train import TrainConfig, set_labeled_cache, train_one_variant  # noqa: E402
from experiments.run_ssl_experiment import filter_valid, split_indices  # noqa: E402
from experiments.run_robustness import train_sort_match_with_corruption  # noqa: E402


# ------------------------------ config ---------------------------------

# Settings tuned for a reasonable overnight window (~8 hours) on Apple M4 Pro.
# Larger than the single-seed MVP (128/4/10k) but smaller than "we have all
# the compute in the world".
HIDDEN = 192
N_LAYERS = 4
EPOCHS_MAIN = 30
EPOCHS_ABLATION = 20
MAX_RECORDS = 20000
SEEDS_MAIN = [0, 1, 2]
SEEDS_ABLATION = [0, 1]
SEEDS_ROBUSTNESS = [0, 1]
DEFAULT_FRAC = 0.10

SSL_WEIGHT = 0.5


# ------------------------------ helpers ---------------------------------


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def train_if_absent(
    variant,
    out_path: Path,
    *,
    train_dataset,
    val_dataset,
    test_dataset,
    labeled_indices,
    unlabeled_indices,
    in_dim,
    target_mean,
    target_std,
    epochs,
    seed,
    ssl_weight=SSL_WEIGHT,
):
    """Run (variant, seed) once and cache. Returns the result dict."""
    existing = load_json(out_path)
    if existing is not None:
        log(f"  [cached] {out_path.name}: test MAE = {existing['test_mae']:.3f} ppm")
        return existing

    cfg = TrainConfig(
        variant=variant,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        epochs=epochs,
        batch_size=32,
        labeled_frac=DEFAULT_FRAC,
        ssl_weight=ssl_weight,
        seed=seed,
        target_mean=target_mean,
        target_std=target_std,
    )
    labeled_ids = {train_dataset.molecules[i].nmrshift_id for i in labeled_indices}
    set_labeled_cache(labeled_ids)
    result = train_one_variant(
        cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        in_dim=in_dim,
        log_path=out_path,
    )
    log(f"  [done] {out_path.name}: test MAE = {result['test_mae']:.3f} ppm  ({result['elapsed_sec']:.0f}s)")
    return result


def make_split(kept, mode, seed, train_frac=0.8, val_frac=0.1):
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


def make_labeled_unlabeled(train_idx, kept, labeled_frac, seed):
    rng = random.Random(seed + 1)
    train_shuffled = list(train_idx)
    rng.shuffle(train_shuffled)
    n_labeled = max(1, int(len(train_shuffled) * labeled_frac))
    labeled_idx = train_shuffled[:n_labeled]
    unlabeled_idx = train_shuffled[n_labeled:]
    return labeled_idx, unlabeled_idx


def aggregate(per_seed_results, variants):
    agg = {}
    for v in variants:
        vals = [per_seed_results[s][v]["test_mae"] for s in per_seed_results]
        agg[v] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
            "values": vals,
        }
    return agg


# ------------------------------ stages ----------------------------------


def stage_a_main(kept, out_root, in_dim):
    log("=" * 60)
    log("STAGE A: Main experiment (random split, labeled_frac=0.10)")
    log("=" * 60)
    stage_dir = out_root / "A_main"
    stage_dir.mkdir(parents=True, exist_ok=True)
    variants = ["supervised", "naive_ssl", "sort_match_ssl"]
    per_seed = {}
    for seed in SEEDS_MAIN:
        log(f"  seed {seed}")
        train_idx, val_idx, test_idx = make_split(kept, "random", seed)
        tm, ts = compute_target_stats(kept, train_idx)
        full_dataset = NMRDataset(kept)
        val_dataset = NMRDataset([kept[i] for i in val_idx])
        test_dataset = NMRDataset([kept[i] for i in test_idx])
        lab_idx, unlab_idx = make_labeled_unlabeled(train_idx, kept, DEFAULT_FRAC, seed)
        log(f"    train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  "
            f"labeled={len(lab_idx)} unlabeled={len(unlab_idx)}")

        seed_dir = stage_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_variant = {}
        for variant in variants:
            res = train_if_absent(
                variant,
                seed_dir / f"{variant}.json",
                train_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                labeled_indices=lab_idx,
                unlabeled_indices=unlab_idx,
                in_dim=in_dim,
                target_mean=tm,
                target_std=ts,
                epochs=EPOCHS_MAIN,
                seed=seed,
            )
            per_variant[variant] = {
                "test_mae": res["test_mae"],
                "best_val_mae": res["best_val_mae"],
                "elapsed_sec": res["elapsed_sec"],
            }
        per_seed[seed] = per_variant

    summary = {
        "name": "A_main",
        "split_mode": "random",
        "labeled_frac": DEFAULT_FRAC,
        "seeds": SEEDS_MAIN,
        "variants": variants,
        "aggregate": aggregate(per_seed, variants),
        "per_seed": per_seed,
    }
    dump_json(stage_dir / "summary.json", summary)
    log(f"[Stage A] saved {stage_dir/'summary.json'}")
    for v, a in summary["aggregate"].items():
        log(f"  {v:20s} {a['mean']:.3f} ± {a['std']:.3f}")
    return summary


def stage_b_scaffold(kept, out_root, in_dim):
    log("=" * 60)
    log("STAGE B: Scaffold split (OOD, labeled_frac=0.10)")
    log("=" * 60)
    stage_dir = out_root / "B_scaffold"
    stage_dir.mkdir(parents=True, exist_ok=True)
    variants = ["supervised", "sort_match_ssl"]
    per_seed = {}
    for seed in SEEDS_MAIN:
        log(f"  seed {seed}")
        train_idx, val_idx, test_idx = make_split(kept, "scaffold", seed)
        tm, ts = compute_target_stats(kept, train_idx)
        full_dataset = NMRDataset(kept)
        val_dataset = NMRDataset([kept[i] for i in val_idx])
        test_dataset = NMRDataset([kept[i] for i in test_idx])
        lab_idx, unlab_idx = make_labeled_unlabeled(train_idx, kept, DEFAULT_FRAC, seed)
        log(f"    train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  "
            f"labeled={len(lab_idx)} unlabeled={len(unlab_idx)}")

        seed_dir = stage_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_variant = {}
        for variant in variants:
            res = train_if_absent(
                variant,
                seed_dir / f"{variant}.json",
                train_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                labeled_indices=lab_idx,
                unlabeled_indices=unlab_idx,
                in_dim=in_dim,
                target_mean=tm,
                target_std=ts,
                epochs=EPOCHS_MAIN,
                seed=seed,
            )
            per_variant[variant] = {
                "test_mae": res["test_mae"],
                "best_val_mae": res["best_val_mae"],
                "elapsed_sec": res["elapsed_sec"],
            }
        per_seed[seed] = per_variant

    summary = {
        "name": "B_scaffold",
        "split_mode": "scaffold",
        "labeled_frac": DEFAULT_FRAC,
        "seeds": SEEDS_MAIN,
        "variants": variants,
        "aggregate": aggregate(per_seed, variants),
        "per_seed": per_seed,
    }
    dump_json(stage_dir / "summary.json", summary)
    log(f"[Stage B] saved {stage_dir/'summary.json'}")
    for v, a in summary["aggregate"].items():
        log(f"  {v:20s} {a['mean']:.3f} ± {a['std']:.3f}")
    return summary


def stage_c_ablation(kept, out_root, in_dim):
    log("=" * 60)
    log("STAGE C: Labeled-fraction ablation (2 seeds, 4 fractions)")
    log("=" * 60)
    stage_dir = out_root / "C_ablation"
    stage_dir.mkdir(parents=True, exist_ok=True)
    variants = ["supervised", "sort_match_ssl"]
    fractions = [0.02, 0.05, 0.10, 0.20]
    per_frac = {}

    # Share train/val/test split across all fractions for a given seed, so
    # the only thing that changes is how many train molecules are labeled.
    full_dataset = NMRDataset(kept)
    split_cache = {}  # seed -> (train_idx, val_idx, test_idx, tm, ts)
    val_datasets = {}
    test_datasets = {}
    for seed in SEEDS_ABLATION:
        train_idx, val_idx, test_idx = make_split(kept, "random", seed)
        tm, ts = compute_target_stats(kept, train_idx)
        split_cache[seed] = (train_idx, val_idx, test_idx, tm, ts)
        val_datasets[seed] = NMRDataset([kept[i] for i in val_idx])
        test_datasets[seed] = NMRDataset([kept[i] for i in test_idx])

    for frac in fractions:
        log(f"  labeled_frac={frac}")
        per_seed_frac = {}
        for seed in SEEDS_ABLATION:
            train_idx, val_idx, test_idx, tm, ts = split_cache[seed]
            lab_idx, unlab_idx = make_labeled_unlabeled(train_idx, kept, frac, seed)

            seed_dir = stage_dir / f"frac_{int(frac*1000):04d}" / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            per_variant = {}
            for variant in variants:
                res = train_if_absent(
                    variant,
                    seed_dir / f"{variant}.json",
                    train_dataset=full_dataset,
                    val_dataset=val_datasets[seed],
                    test_dataset=test_datasets[seed],
                    labeled_indices=lab_idx,
                    unlabeled_indices=unlab_idx,
                    in_dim=in_dim,
                    target_mean=tm,
                    target_std=ts,
                    epochs=EPOCHS_ABLATION,
                    seed=seed,
                )
                per_variant[variant] = {
                    "test_mae": res["test_mae"],
                    "best_val_mae": res["best_val_mae"],
                    "elapsed_sec": res["elapsed_sec"],
                }
            per_seed_frac[seed] = per_variant

        per_frac[frac] = {
            "variants": variants,
            "per_seed": per_seed_frac,
            "aggregate": aggregate(per_seed_frac, variants),
        }
        log(f"    {frac}: " + "  ".join(
            f"{v}={per_frac[frac]['aggregate'][v]['mean']:.3f}±{per_frac[frac]['aggregate'][v]['std']:.3f}"
            for v in variants
        ))

    dump_json(stage_dir / "summary.json", {
        "name": "C_ablation",
        "fractions": fractions,
        "seeds": SEEDS_ABLATION,
        "by_fraction": per_frac,
    })
    log("[Stage C] done")
    return per_frac


def stage_d_robustness(kept, out_root, in_dim):
    log("=" * 60)
    log("STAGE D: Robustness to corrupt unlabeled peaks (2 seeds)")
    log("=" * 60)
    stage_dir = out_root / "D_robustness"
    stage_dir.mkdir(parents=True, exist_ok=True)

    corruptions = [
        ("clean", 0.0, 0.0, 0.0),
        ("noise_1ppm", 1.0, 0.0, 0.0),
        ("drop_15", 0.0, 0.15, 0.0),
        ("noise_1ppm_drop_10_spurious_10", 1.0, 0.10, 0.10),
    ]

    full_dataset = NMRDataset(kept)
    per_corr = {}
    for corr_name, sigma, drop, spurious in corruptions:
        log(f"  corruption={corr_name}")
        per_seed_corr = {}
        for seed in SEEDS_ROBUSTNESS:
            train_idx, val_idx, test_idx = make_split(kept, "random", seed)
            tm, ts = compute_target_stats(kept, train_idx)
            lab_idx, unlab_idx = make_labeled_unlabeled(train_idx, kept, DEFAULT_FRAC, seed)
            val_dataset = NMRDataset([kept[i] for i in val_idx])
            test_dataset = NMRDataset([kept[i] for i in test_idx])

            out_path = stage_dir / corr_name / f"seed_{seed}.json"
            existing = load_json(out_path)
            if existing is not None:
                log(f"    [cached] seed {seed}: test MAE = {existing['test_mae']:.3f}")
                per_seed_corr[seed] = existing
                continue

            t0 = time.time()
            result = train_sort_match_with_corruption(
                noise_sigma=sigma,
                drop_frac=drop,
                spurious_frac=spurious,
                labeled_indices=lab_idx,
                unlabeled_indices=unlab_idx,
                full_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                in_dim=in_dim,
                target_mean=tm,
                target_std=ts,
                epochs=EPOCHS_MAIN,
                hidden=HIDDEN,
                n_layers=N_LAYERS,
                seed=seed,
            )
            result["elapsed_sec"] = time.time() - t0
            dump_json(out_path, result)
            log(f"    seed {seed}: test MAE = {result['test_mae']:.3f}  ({result['elapsed_sec']:.0f}s)")
            per_seed_corr[seed] = result

        vals = [per_seed_corr[s]["test_mae"] for s in per_seed_corr]
        per_corr[corr_name] = {
            "corruption": {"noise_sigma": sigma, "drop_frac": drop, "spurious_frac": spurious},
            "per_seed": per_seed_corr,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    # Also add the "clean supervised" baseline from stage A
    dump_json(stage_dir / "summary.json", {
        "name": "D_robustness",
        "seeds": SEEDS_ROBUSTNESS,
        "per_corruption": per_corr,
    })
    log("[Stage D] done")
    for k, v in per_corr.items():
        log(f"  {k:40s} {v['mean']:.3f} ± {v['std']:.3f}")
    return per_corr


def stage_e_solvent(kept, out_root, in_dim):
    log("=" * 60)
    log("STAGE E: Solvent conditioning (molecules with solvent metadata)")
    log("=" * 60)
    stage_dir = out_root / "E_solvent"
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Filter to molecules with non-None, non-Unreported solvent
    with_solvent = [m for m in kept if m.solvent and m.solvent not in ("Unreported", "Unknown")]
    log(f"  molecules with solvent metadata: {len(with_solvent)}")
    if len(with_solvent) < 500:
        log("  too few — stage E skipped")
        return None

    # Count solvent categories
    from collections import Counter
    counter = Counter(m.solvent for m in with_solvent)
    top_solvents = [s for s, _ in counter.most_common(8)]
    log(f"  top solvents: {top_solvents}")

    # Restrict to top 5 solvents for the conditioning experiment
    kept_e = [m for m in with_solvent if m.solvent in top_solvents[:5]]
    log(f"  restricted to top-5 solvents: {len(kept_e)}")

    full_dataset = NMRDataset(kept_e)
    variants = ["supervised", "sort_match_ssl"]
    per_seed = {}
    for seed in SEEDS_MAIN:  # solvent stage uses 3 seeds (matches main)
        train_idx, val_idx, test_idx = make_split(kept_e, "random", seed)
        tm, ts = compute_target_stats(kept_e, train_idx)
        val_dataset = NMRDataset([kept_e[i] for i in val_idx])
        test_dataset = NMRDataset([kept_e[i] for i in test_idx])
        lab_idx, unlab_idx = make_labeled_unlabeled(train_idx, kept_e, DEFAULT_FRAC, seed)

        seed_dir = stage_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        per_variant = {}
        for variant in variants:
            res = train_if_absent(
                variant,
                seed_dir / f"{variant}.json",
                train_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                labeled_indices=lab_idx,
                unlabeled_indices=unlab_idx,
                in_dim=in_dim,
                target_mean=tm,
                target_std=ts,
                epochs=EPOCHS_MAIN,
                seed=seed,
            )
            per_variant[variant] = {
                "test_mae": res["test_mae"],
                "best_val_mae": res["best_val_mae"],
                "elapsed_sec": res["elapsed_sec"],
            }
        per_seed[seed] = per_variant

    summary = {
        "name": "E_solvent",
        "n_molecules": len(kept_e),
        "solvent_counter": dict(counter),
        "top_solvents_used": top_solvents[:5],
        "seeds": SEEDS_MAIN,
        "variants": variants,
        "aggregate": aggregate(per_seed, variants),
        "per_seed": per_seed,
    }
    dump_json(stage_dir / "summary.json", summary)
    log("[Stage E] done")
    for v, a in summary["aggregate"].items():
        log(f"  {v:20s} {a['mean']:.3f} ± {a['std']:.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_overnight")
    parser.add_argument("--stages", type=str, default="ABCDE",
                        help="Substring of 'ABCDE' — which stages to run")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log(f"[overnight] starting; stages={args.stages}; max_records={args.max_records}")
    log(f"[overnight] output directory: {args.out}")
    log(f"[overnight] config: hidden={HIDDEN} n_layers={N_LAYERS} epochs_main={EPOCHS_MAIN} epochs_ablation={EPOCHS_ABLATION}")

    log("[overnight] parsing SDF (13C, all records)")
    t0 = time.time()
    raw = list(iter_nmrshiftdb2_sdf(args.sdf, nucleus="13C", max_records=args.max_records))
    kept = filter_valid(raw)
    log(f"[overnight] parsed {len(raw)} records, kept {len(kept)} clean molecules in {time.time()-t0:.0f}s")
    x0, _ = mol_to_graph_tensors(kept[0].mol)
    in_dim = x0.shape[1]
    log(f"[overnight] atom feature dim = {in_dim}")

    stage_summaries = {}

    if "A" in args.stages:
        stage_summaries["A"] = stage_a_main(kept, args.out, in_dim)
    if "B" in args.stages:
        stage_summaries["B"] = stage_b_scaffold(kept, args.out, in_dim)
    if "C" in args.stages:
        stage_summaries["C"] = stage_c_ablation(kept, args.out, in_dim)
    if "D" in args.stages:
        stage_summaries["D"] = stage_d_robustness(kept, args.out, in_dim)
    if "E" in args.stages:
        stage_summaries["E"] = stage_e_solvent(kept, args.out, in_dim)

    dump_json(args.out / "overnight_summary.json", {
        "args": vars(args) | {"sdf": str(args.sdf), "out": str(args.out)},
        "config": {
            "hidden": HIDDEN,
            "n_layers": N_LAYERS,
            "epochs_main": EPOCHS_MAIN,
            "epochs_ablation": EPOCHS_ABLATION,
            "max_records": args.max_records,
            "seeds": SEEDS_MAIN,
            "labeled_frac": DEFAULT_FRAC,
        },
        "n_clean_molecules": len(kept),
        "stages_run": list(stage_summaries.keys()),
    })

    log("=" * 60)
    log("OVERNIGHT RUN COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
