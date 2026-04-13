"""Add seeds 3 and 4 to Stage B scaffold split to address the DA reviewer's
CRITICAL issue on statistical power. Writes new per-seed files alongside the
existing seed_0/1/2, then recomputes the summary with n=5.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import NMRDataset, iter_nmrshiftdb2_sdf, mol_to_graph_tensors  # noqa: E402
from src.train import TrainConfig, set_labeled_cache, train_one_variant  # noqa: E402
from experiments.run_ssl_experiment import filter_valid  # noqa: E402
from experiments.run_overnight import (  # noqa: E402
    HIDDEN,
    N_LAYERS,
    EPOCHS_MAIN,
    DEFAULT_FRAC,
    MAX_RECORDS,
    make_split,
    compute_target_stats,
    make_labeled_unlabeled,
)


def main():
    new_seeds = [0, 1, 2, 3, 4]
    out_dir = ROOT / "experiments" / "results_overnight" / "B_scaffold"
    print(f"[extend] loading data")
    raw = list(iter_nmrshiftdb2_sdf(
        ROOT / "data" / "nmrshiftdb2withsignals.sd",
        nucleus="13C",
        max_records=MAX_RECORDS,
    ))
    kept = filter_valid(raw)
    print(f"[extend] {len(kept)} clean molecules")
    x0, _ = mol_to_graph_tensors(kept[0].mol)
    in_dim = x0.shape[1]

    variants = ["supervised", "sort_match_ssl"]
    full_dataset = NMRDataset(kept)

    for seed in new_seeds:
        print(f"\n[extend] seed {seed}")
        train_idx, val_idx, test_idx = make_split(kept, "scaffold", seed)
        tm, ts = compute_target_stats(kept, train_idx)
        val_dataset = NMRDataset([kept[i] for i in val_idx])
        test_dataset = NMRDataset([kept[i] for i in test_idx])
        lab_idx, unlab_idx = make_labeled_unlabeled(train_idx, kept, DEFAULT_FRAC, seed)
        labeled_ids = {kept[i].nmrshift_id for i in lab_idx}
        set_labeled_cache(labeled_ids)
        print(f"  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  labeled={len(lab_idx)}")

        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            log_path = seed_dir / f"{variant}.json"
            if log_path.exists():
                print(f"  [cached] {variant}")
                continue
            cfg = TrainConfig(
                variant=variant,
                hidden=HIDDEN,
                n_layers=N_LAYERS,
                epochs=EPOCHS_MAIN,
                batch_size=32,
                labeled_frac=DEFAULT_FRAC,
                ssl_weight=0.5,
                seed=seed,
                target_mean=tm,
                target_std=ts,
            )
            t0 = time.time()
            result = train_one_variant(
                cfg,
                train_dataset=full_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                labeled_indices=lab_idx,
                unlabeled_indices=unlab_idx,
                in_dim=in_dim,
                log_path=log_path,
            )
            print(f"  {variant}: test MAE = {result['test_mae']:.3f} ppm ({time.time()-t0:.0f}s)")

    # Recompute Stage B summary with all 5 seeds
    print("\n[extend] recomputing summary with 5 seeds")
    per_seed = {}
    for seed in [0, 1, 2, 3, 4]:
        seed_dir = out_dir / f"seed_{seed}"
        per_variant = {}
        for variant in variants:
            p = seed_dir / f"{variant}.json"
            if not p.exists():
                print(f"  missing {p}")
                continue
            with p.open() as f:
                r = json.load(f)
            per_variant[variant] = {
                "test_mae": r["test_mae"],
                "best_val_mae": r["best_val_mae"],
                "elapsed_sec": r["elapsed_sec"],
            }
        per_seed[seed] = per_variant

    def agg(vs):
        return {
            "mean": float(np.mean(vs)),
            "std": float(np.std(vs)),
            "n": len(vs),
            "values": vs,
        }

    aggregate = {}
    for v in variants:
        vals = [per_seed[s][v]["test_mae"] for s in per_seed if v in per_seed[s]]
        aggregate[v] = agg(vals)

    # Paired test: per-seed difference supervised - sort_match
    diffs = [
        per_seed[s]["supervised"]["test_mae"] - per_seed[s]["sort_match_ssl"]["test_mae"]
        for s in per_seed
        if "supervised" in per_seed[s] and "sort_match_ssl" in per_seed[s]
    ]
    paired_mean = float(np.mean(diffs))
    paired_std = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    paired_se = paired_std / np.sqrt(len(diffs)) if len(diffs) > 0 else 0.0
    # Wilcoxon signed-rank test (scipy)
    from scipy.stats import wilcoxon, ttest_rel
    sup_vals = [per_seed[s]["supervised"]["test_mae"] for s in per_seed]
    sm_vals = [per_seed[s]["sort_match_ssl"]["test_mae"] for s in per_seed]
    try:
        w_stat, w_p = wilcoxon(sup_vals, sm_vals, alternative="greater")
    except ValueError as e:
        w_stat, w_p = None, None
    t_stat, t_p = ttest_rel(sup_vals, sm_vals, alternative="greater")

    summary = {
        "name": "B_scaffold",
        "split_mode": "scaffold",
        "labeled_frac": DEFAULT_FRAC,
        "seeds": list(per_seed.keys()),
        "variants": variants,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "paired_diff_mean_ppm": paired_mean,
        "paired_diff_std_ppm": paired_std,
        "paired_diff_se_ppm": paired_se,
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_value_onesided": float(w_p) if w_p is not None else None,
        "paired_t_stat": float(t_stat),
        "paired_t_p_value_onesided": float(t_p),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[extend] FINAL STAGE B (n=5 seeds):")
    for v, a in aggregate.items():
        print(f"  {v:20s} {a['mean']:.3f} \u00b1 {a['std']:.3f}  (n={a['n']})")
    print(f"\n  paired diff (sup - sm)    : {paired_mean:.3f} \u00b1 {paired_std:.3f} ppm (SE {paired_se:.3f})")
    print(f"  paired t-test p (one-sided): {t_p:.4f}")
    if w_p is not None:
        print(f"  Wilcoxon p (one-sided)    : {w_p:.4f}")


if __name__ == "__main__":
    main()
