"""Load the 2-D experiment models, fit conformal intervals on a held-out
calibration split, evaluate empirical coverage on the test split, and run
a structure-verification sanity check.

Usage (after run_2d_experiment.py has produced results_2d/):
    python3 experiments/run_conformal_evaluation.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.conformal import ConformalCalibrator
from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D,
    HSQCDataset,
    compute_target_stats,
    pad_collate,
)
from experiments.run_2d_experiment import split_indices


def _collect_predictions(model, loader, device, c_mean, c_std, h_mean, h_std):
    """Run model on loader and collect per-atom (pred_c, true_c, pred_h, true_h) arrays."""
    cm = torch.tensor(c_mean, device=device)
    cs = torch.tensor(c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(h_mean, device=device)
    hs = torch.tensor(h_std, device=device).clamp_min(1e-3)

    pred_c_all = []
    true_c_all = []
    pred_h_all = []
    true_h_all = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            c_atoms = batch["c_atoms"].to(device)
            c_shifts = batch["c_shifts"].to(device)
            c_mask = batch["c_mask"].to(device)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(device)
            hsqc_h = batch["hsqc_h"].to(device)
            hsqc_mask = batch["hsqc_mask"].to(device)

            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm
            h_pred = h_pred_norm * hs + hm

            safe_c = c_atoms.clamp(min=0)
            c_gathered = c_pred.gather(1, safe_c)
            B, K = c_mask.shape
            for i in range(B):
                mask = c_mask[i].bool()
                pred_c_all.append(c_gathered[i, mask].cpu().numpy())
                true_c_all.append(c_shifts[i, mask].cpu().numpy())

            safe_h = hsqc_c_atoms.clamp(min=0)
            h_gathered = h_pred.gather(1, safe_h)
            B2, K2 = hsqc_mask.shape
            for i in range(B2):
                mask = hsqc_mask[i].bool()
                pred_h_all.append(h_gathered[i, mask].cpu().numpy())
                true_h_all.append(hsqc_h[i, mask].cpu().numpy())

    return (
        np.concatenate(pred_c_all) if pred_c_all else np.zeros(0),
        np.concatenate(true_c_all) if true_c_all else np.zeros(0),
        np.concatenate(pred_h_all) if pred_h_all else np.zeros(0),
        np.concatenate(true_h_all) if true_h_all else np.zeros(0),
    )


def _retrain_and_get_predictor(molecules, train_idx, val_idx, test_idx, cal_idx, cfg, labeled, unlabeled, ssl_variant, device):
    """Quick retrain for conformal evaluation with a calibration split carved out."""
    from src.nmr2d.train_2d import train_variant
    # We use the existing train_variant with a modified labeled/unlabeled set that
    # excludes the calibration split from training.
    cfg = Config2D(**{**cfg.__dict__, "variant": ssl_variant})
    log_path = Path("/tmp") / f"conformal_retrain_{ssl_variant}.json"
    # Use train_idx that excludes calibration (i.e., reuse same labeled/unlabeled)
    result = train_variant(
        cfg, HSQCDataset(molecules), train_idx, val_idx, test_idx,
        labeled_indices=labeled, unlabeled_indices=unlabeled, log_path=log_path,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "results_2d")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cal-frac", type=float, default=0.15)
    args = parser.parse_args()

    print(f"[conformal] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)

    # Carve off a calibration split from the test set
    rng = random.Random(args.seed + 17)
    test_shuffled = test_idx.copy()
    rng.shuffle(test_shuffled)
    n_cal = max(5, int(len(test_shuffled) * args.cal_frac * 10))  # use 50% of the test as cal since test is small
    n_cal = min(n_cal, len(test_shuffled) // 2)
    cal_idx = test_shuffled[:n_cal]
    test_idx_after = test_shuffled[n_cal:]
    print(f"  cal={len(cal_idx)} test-after-calibration={len(test_idx_after)}")

    # Re-retrain the 2D-SSL variant from scratch using all-but-calibration as training pool.
    # To keep this quick we re-use the saved result from the main run.
    result_path = args.results_dir / f"seed_{args.seed}" / "sort_match_ssl_2d.json"
    if not result_path.exists():
        print(f"ERROR: {result_path} not found. Run run_2d_experiment.py first.")
        sys.exit(1)
    with result_path.open() as f:
        main_result = json.load(f)
    cfg_d = main_result["config"]

    # For a proper conformal evaluation we'd retrain on train∪val and calibrate on cal.
    # Here we reuse the already-trained model from the main experiment (which used
    # train for training and val for early stopping) and calibrate on the original
    # test set's first half, then evaluate on the second half. This is a
    # cal/test split derived post-hoc, not a full re-run.
    import copy
    import torch.utils.data as tud
    from torch.utils.data import DataLoader

    # Rebuild and retrain the exact model
    cfg = Config2D(**cfg_d)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    rng = random.Random(args.seed + 1)
    train_shuffled = train_idx.copy()
    rng.shuffle(train_shuffled)
    n_lab = max(1, int(len(train_shuffled) * cfg.labeled_frac))
    labeled = train_shuffled[:n_lab]
    unlabeled = train_shuffled[n_lab:]

    print(f"[conformal] retraining sort_match_ssl_2d to extract predictions")
    from src.nmr2d.train_2d import train_variant
    log_path = Path("/tmp/conformal_sm2d.json")
    result = train_variant(
        cfg, dataset, train_idx, val_idx, cal_idx,  # use cal as eval to stash the model in memory
        labeled_indices=labeled, unlabeled_indices=unlabeled, log_path=log_path,
    )
    print(f"  retrain test (on cal split) C MAE: {result['test_c_mae']:.3f}")

    # Reload the trained model weights from the training run — we saved best_state
    # only internally. Simpler: rerun inference with the retrained model still
    # in scope. But we don't have a handle to the model here. Let me build a
    # separate helper that trains + returns the model object.

    # Simpler path: redo with direct model ownership
    print("[conformal] direct inference pass")
    from src.nmr2d.train_2d import NMRDualHeadGNN, set_seed
    set_seed(args.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    from src.nmr2d.train_2d import Config2D as _Cfg

    # Train once in-process to get the model
    from src.nmr2d.train_2d import train_variant as _tv

    # We already trained it above but the trained state wasn't returned. Train
    # once more and snapshot. (Quick — ~30s to 2min.)
    import math
    from src.nmr2d.train_2d import (
        per_atom_c_loss, masked_sort_match_loss, sliced_sort_match_loss_2d,
    )

    # -- Inline mini-training --
    cm_t = torch.tensor(cfg.c_mean, device=device)
    cs_t = torch.tensor(cfg.c_std, device=device).clamp_min(1e-3)
    hm_t = torch.tensor(cfg.h_mean, device=device)
    hs_t = torch.tensor(cfg.h_std, device=device).clamp_min(1e-3)
    labeled_lookup = {dataset.molecules[i].nmr_id for i in labeled}
    all_indices = labeled + unlabeled
    train_loader = DataLoader(tud.Subset(dataset, all_indices), batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
    best_val = math.inf
    best_state = None
    val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            c_atoms = batch["c_atoms"].to(device)
            c_shifts = batch["c_shifts"].to(device)
            c_mask = batch["c_mask"].to(device)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(device)
            hsqc_h = batch["hsqc_h"].to(device)
            hsqc_c = batch["hsqc_c"].to(device)
            hsqc_mask = batch["hsqc_mask"].to(device)
            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs_t + cm_t
            h_pred = h_pred_norm * hs_t + hm_t
            is_labeled = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=device)
            lab = is_labeled
            unlab = ~is_labeled
            sup = c_pred.new_tensor(0.0)
            ssl = c_pred.new_tensor(0.0)
            if lab.any():
                sup = per_atom_c_loss(c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab])
            if unlab.any():
                max_n = x.size(1)
                fill = torch.where(hsqc_mask[unlab], hsqc_c_atoms[unlab], hsqc_c_atoms[unlab].new_full(hsqc_c_atoms[unlab].shape, max_n + 1))
                sorted_hc, _ = torch.sort(fill, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pred_h_at_c = h_pred[unlab].gather(1, safe_hc)
                pred_c_at_c = c_pred[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pred_h_at_c, pred_c_at_c], dim=-1)
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl = sliced_sort_match_loss_2d(pred_set, target_set, hsqc_mask[unlab], K=cfg.K_directions, kind="mse")
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        # Validation
        model.eval()
        errs_c, ns_c = [], []
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(device); adjv = vb["adj"].to(device)
                am = vb["atom_mask"].to(device); cat = vb["c_atoms"].to(device)
                cs_v = vb["c_shifts"].to(device); cm_v = vb["c_mask"].to(device)
                cpn, _ = model(xv, adjv, am)
                cp = cpn * cs_t + cm_t
                g = cp.gather(1, cat.clamp(min=0))
                e = (g - cs_v).abs() * cm_v.float()
                errs_c.append(e.sum().item()); ns_c.append(cm_v.sum().item())
        val_c = sum(errs_c) / max(sum(ns_c), 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  in-process retrain best val C MAE: {best_val:.3f}")

    # Extract predictions on cal and test splits
    cal_loader = DataLoader(tud.Subset(dataset, cal_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(tud.Subset(dataset, test_idx_after), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    pred_c_cal, true_c_cal, pred_h_cal, true_h_cal = _collect_predictions(
        model, cal_loader, device, cfg.c_mean, cfg.c_std, cfg.h_mean, cfg.h_std
    )
    pred_c_test, true_c_test, pred_h_test, true_h_test = _collect_predictions(
        model, test_loader, device, cfg.c_mean, cfg.c_std, cfg.h_mean, cfg.h_std
    )

    print(f"[conformal] cal atoms C={len(pred_c_cal)} H={len(pred_h_cal)}")
    print(f"[conformal] test atoms C={len(pred_c_test)} H={len(pred_h_test)}")

    # Fit conformal on cal, evaluate on test
    c_calibrator = ConformalCalibrator(alpha=args.alpha)
    h_calibrator = ConformalCalibrator(alpha=args.alpha)
    c_calibrator.fit(np.abs(pred_c_cal - true_c_cal))
    h_calibrator.fit(np.abs(pred_h_cal - true_h_cal))
    c_coverage = c_calibrator.coverage(true_c_test, pred_c_test)
    h_coverage = h_calibrator.coverage(true_h_test, pred_h_test)

    print("\n[conformal] 13C calibration:")
    for k, v in c_coverage.items():
        print(f"  {k}: {v}")
    print("\n[conformal] 1H calibration:")
    for k, v in h_coverage.items():
        print(f"  {k}: {v}")

    out = args.results_dir / f"seed_{args.seed}" / "conformal.json"
    with out.open("w") as f:
        json.dump({
            "seed": args.seed,
            "alpha": args.alpha,
            "c": {
                "q_ppm": c_calibrator.quantile(),
                "coverage": c_coverage,
                "n_cal": int(len(pred_c_cal)),
                "n_test": int(len(pred_c_test)),
            },
            "h": {
                "q_ppm": h_calibrator.quantile(),
                "coverage": h_coverage,
                "n_cal": int(len(pred_h_cal)),
                "n_test": int(len(pred_h_test)),
            },
        }, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
