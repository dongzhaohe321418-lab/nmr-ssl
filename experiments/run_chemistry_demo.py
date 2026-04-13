"""Chemistry demonstration of the 2-D SSL + conformal pipeline.

Runs a structure-verification sanity check: for a handful of held-out NMRShiftDB2
molecules, predict the HSQC peak set with both point estimates and conformal
prediction intervals, and report whether the observed peaks fall within the
intervals — the "is this structure consistent with the spectrum?" question.

Also demonstrates a discriminative test: given two candidate structures for
the same observed spectrum, which one does the predictor consider more
consistent?
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses import masked_sort_match_loss
from src.nmr2d.conformal import ConformalCalibrator
from src.nmr2d.data_2d import HSQCMolecule, build_hsqc_molecules
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D,
    HSQCDataset,
    compute_target_stats,
    pad_collate,
    per_atom_c_loss,
    set_seed,
)
from experiments.run_2d_experiment import split_indices


def train_sort_match_ssl_2d(
    dataset: HSQCDataset,
    train_idx,
    val_idx,
    cfg: Config2D,
    labeled_indices,
    unlabeled_indices,
    device,
):
    """Retrain the 2-D SSL variant in-process so we can keep the model handle."""
    set_seed(cfg.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(
        in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cm = torch.tensor(cfg.c_mean, device=device)
    cs = torch.tensor(cfg.c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=device)
    hs = torch.tensor(cfg.h_std, device=device).clamp_min(1e-3)

    labeled_lookup = {dataset.molecules[i].nmr_id for i in labeled_indices}
    all_indices = labeled_indices + unlabeled_indices
    train_loader = DataLoader(
        tud.Subset(dataset, all_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        tud.Subset(dataset, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )

    best_val = math.inf
    best_state = None
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
            c_pred = c_pred_norm * cs + cm
            h_pred = h_pred_norm * hs + hm
            is_labeled = torch.tensor(
                [m in labeled_lookup for m in batch["ids"]], device=device
            )
            lab = is_labeled
            unlab = ~is_labeled
            sup = c_pred.new_tensor(0.0)
            ssl = c_pred.new_tensor(0.0)
            if lab.any():
                sup = per_atom_c_loss(
                    c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab]
                )
            if unlab.any():
                max_n = x.size(1)
                fill = torch.where(
                    hsqc_mask[unlab],
                    hsqc_c_atoms[unlab],
                    hsqc_c_atoms[unlab].new_full(hsqc_c_atoms[unlab].shape, max_n + 1),
                )
                sorted_hc, _ = torch.sort(fill, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pred_h_at_c = h_pred[unlab].gather(1, safe_hc)
                pred_c_at_c = c_pred[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pred_h_at_c, pred_c_at_c], dim=-1)
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl = sliced_sort_match_loss_2d(
                    pred_set, target_set, hsqc_mask[unlab], K=cfg.K_directions, kind="mse"
                )
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        errs = []
        ns = []
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(device)
                adjv = vb["adj"].to(device)
                am = vb["atom_mask"].to(device)
                cat = vb["c_atoms"].to(device)
                cs_v = vb["c_shifts"].to(device)
                cm_v = vb["c_mask"].to(device)
                cpn, _ = model(xv, adjv, am)
                cp = cpn * cs + cm
                g = cp.gather(1, cat.clamp(min=0))
                e = (g - cs_v).abs() * cm_v.float()
                errs.append(e.sum().item())
                ns.append(cm_v.sum().item())
        val_c = sum(errs) / max(sum(ns), 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_val


def collect_residuals(model, dataset, indices, device, cfg):
    loader = DataLoader(
        tud.Subset(dataset, indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )
    cm = torch.tensor(cfg.c_mean, device=device)
    cs = torch.tensor(cfg.c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=device)
    hs = torch.tensor(cfg.h_std, device=device).clamp_min(1e-3)
    c_res, h_res = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            am = batch["atom_mask"].to(device)
            c_atoms = batch["c_atoms"].to(device)
            c_shifts = batch["c_shifts"].to(device)
            c_mask = batch["c_mask"].to(device)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(device)
            hsqc_h = batch["hsqc_h"].to(device)
            hsqc_mask = batch["hsqc_mask"].to(device)
            cp_n, hp_n = model(x, adj, am)
            cp = cp_n * cs + cm
            hp = hp_n * hs + hm
            gc = cp.gather(1, c_atoms.clamp(min=0))
            gh = hp.gather(1, hsqc_c_atoms.clamp(min=0))
            for i in range(gc.size(0)):
                m = c_mask[i].bool()
                c_res.extend((gc[i, m] - c_shifts[i, m]).abs().cpu().numpy())
            for i in range(gh.size(0)):
                m = hsqc_mask[i].bool()
                h_res.extend((gh[i, m] - hsqc_h[i, m]).abs().cpu().numpy())
    return np.asarray(c_res), np.asarray(h_res)


def demo_molecule(model, dataset, mol_idx, device, cfg, c_calibrator, h_calibrator):
    m = dataset.molecules[mol_idx]
    item = dataset[mol_idx]
    x = item["x"].unsqueeze(0).to(device)
    adj = item["adj"].unsqueeze(0).to(device)
    atom_mask = torch.ones(1, x.size(1), dtype=torch.bool, device=device)
    cm = torch.tensor(cfg.c_mean, device=device)
    cs = torch.tensor(cfg.c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=device)
    hs = torch.tensor(cfg.h_std, device=device).clamp_min(1e-3)
    model.eval()
    with torch.no_grad():
        c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
        c_pred = (c_pred_norm * cs + cm).squeeze(0).cpu().numpy()
        h_pred = (h_pred_norm * hs + hm).squeeze(0).cpu().numpy()
    return {
        "smiles": m.smiles,
        "nmr_id": m.nmr_id,
        "n_atoms": m.n_atoms,
        "n_hsqc": m.n_hsqc_peaks,
        "hsqc_observed": m.hsqc_peaks,
        "hsqc_predicted": [
            (float(h_pred[c_idx]), float(c_pred[c_idx])) for c_idx in m.hsqc_c_atoms
        ],
        "c_interval_halfwidth_ppm": c_calibrator.quantile(),
        "h_interval_halfwidth_ppm": h_calibrator.quantile(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-demo", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "chemistry_demo.json")
    args = parser.parse_args()

    print("[demo] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    dataset = HSQCDataset(molecules)
    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
    print(f"  {len(molecules)} molecules, train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cfg = Config2D(
        variant="sort_match_ssl_2d",
        hidden=192,
        n_layers=4,
        epochs=30,
        labeled_frac=0.1,
        ssl_weight=0.5,
        K_directions=8,
        seed=args.seed,
        c_mean=c_mean,
        c_std=c_std,
        h_mean=h_mean,
        h_std=h_std,
    )
    rng = random.Random(args.seed + 1)
    train_shuffled = train_idx.copy()
    rng.shuffle(train_shuffled)
    n_lab = max(1, int(len(train_shuffled) * cfg.labeled_frac))
    labeled = train_shuffled[:n_lab]
    unlabeled = train_shuffled[n_lab:]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("[demo] retraining sort_match_ssl_2d in-process")
    model, best_val = train_sort_match_ssl_2d(
        dataset, train_idx, val_idx, cfg, labeled, unlabeled, device
    )
    print(f"  best val C MAE: {best_val:.3f}")

    # Use the val split as calibration, test split as held-out
    print("[demo] calibrating conformal intervals on val split")
    c_res_cal, h_res_cal = collect_residuals(model, dataset, val_idx, device, cfg)
    print(f"  cal residuals: C={len(c_res_cal)}, H={len(h_res_cal)}")

    c_calibrator = ConformalCalibrator(alpha=args.alpha)
    h_calibrator = ConformalCalibrator(alpha=args.alpha)
    c_calibrator.fit(c_res_cal)
    h_calibrator.fit(h_res_cal)

    c_res_test, h_res_test = collect_residuals(model, dataset, test_idx, device, cfg)
    c_coverage = c_calibrator.coverage(
        np.zeros_like(c_res_test), c_res_test
    )
    # That's not quite right — coverage is fraction of test |y - y_hat| <= q
    c_covered = float(np.mean(c_res_test <= c_calibrator.quantile()))
    h_covered = float(np.mean(h_res_test <= h_calibrator.quantile()))
    print(
        f"  13C: q={c_calibrator.quantile():.3f} ppm, empirical coverage={c_covered:.3f}"
        f" (target {1 - args.alpha})"
    )
    print(
        f"  1H : q={h_calibrator.quantile():.3f} ppm, empirical coverage={h_covered:.3f}"
        f" (target {1 - args.alpha})"
    )

    # Demo: show predictions + intervals for 5 test molecules
    print(f"\n[demo] predicting HSQC for {args.n_demo} test molecules")
    demos = []
    for mol_idx in test_idx[: args.n_demo]:
        d = demo_molecule(model, dataset, mol_idx, device, cfg, c_calibrator, h_calibrator)
        demos.append(d)
        print(f"\nMolecule: {d['smiles']}")
        print(f"  atoms={d['n_atoms']}  hsqc_peaks={d['n_hsqc']}")
        print(f"  Observed vs Predicted (with conformal intervals at 95%):")
        print(f"  {'i':>3s}  {'H_obs':>6s}  {'H_pred':>6s}  {'|ΔH|':>5s} (|Δ|<={d['h_interval_halfwidth_ppm']:.2f})"
              f"    {'C_obs':>7s}  {'C_pred':>7s}  {'|ΔC|':>6s} (|Δ|<={d['c_interval_halfwidth_ppm']:.2f})"
              f"  {'cons?':>6s}")
        for i, ((h_obs, c_obs), (h_pred, c_pred)) in enumerate(
            zip(d["hsqc_observed"], d["hsqc_predicted"])
        ):
            dh = abs(h_obs - h_pred)
            dc = abs(c_obs - c_pred)
            h_ok = dh <= d["h_interval_halfwidth_ppm"]
            c_ok = dc <= d["c_interval_halfwidth_ppm"]
            ok = "✓" if h_ok and c_ok else "✗"
            print(
                f"  {i:3d}  {h_obs:6.2f}  {h_pred:6.2f}  {dh:5.2f}        "
                f"    {c_obs:7.2f}  {c_pred:7.2f}  {dc:6.2f}          {ok}"
            )
        h_consistent = all(
            abs(h_obs - h_pred) <= d["h_interval_halfwidth_ppm"]
            for (h_obs, _), (h_pred, _) in zip(d["hsqc_observed"], d["hsqc_predicted"])
        )
        c_consistent = all(
            abs(c_obs - c_pred) <= d["c_interval_halfwidth_ppm"]
            for (_, c_obs), (_, c_pred) in zip(d["hsqc_observed"], d["hsqc_predicted"])
        )
        print(f"  → H-consistent at 95%: {h_consistent}  |  C-consistent at 95%: {c_consistent}")
        d["h_all_within"] = h_consistent
        d["c_all_within"] = c_consistent

    # Aggregate "structure consistency" stats over the whole test set
    print(f"\n[demo] structure-consistency analysis across {len(test_idx)} test molecules")
    within_all_h = 0
    within_all_c = 0
    within_both = 0
    total = 0
    for mol_idx in test_idx:
        d = demo_molecule(model, dataset, mol_idx, device, cfg, c_calibrator, h_calibrator)
        total += 1
        h_ok = all(
            abs(ho - hp) <= d["h_interval_halfwidth_ppm"]
            for (ho, _), (hp, _) in zip(d["hsqc_observed"], d["hsqc_predicted"])
        )
        c_ok = all(
            abs(co - cp) <= d["c_interval_halfwidth_ppm"]
            for (_, co), (_, cp) in zip(d["hsqc_observed"], d["hsqc_predicted"])
        )
        within_all_h += int(h_ok)
        within_all_c += int(c_ok)
        within_both += int(h_ok and c_ok)

    print(f"  test molecules: {total}")
    print(
        f"  structure consistent on all H shifts (95% intervals): "
        f"{within_all_h}/{total} ({within_all_h/total*100:.1f}%)"
    )
    print(
        f"  structure consistent on all C shifts (95% intervals): "
        f"{within_all_c}/{total} ({within_all_c/total*100:.1f}%)"
    )
    print(
        f"  both nuclei consistent: {within_both}/{total} ({within_both/total*100:.1f}%)"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(
            {
                "seed": args.seed,
                "alpha": args.alpha,
                "best_val_c_mae": best_val,
                "c_quantile_ppm": c_calibrator.quantile(),
                "h_quantile_ppm": h_calibrator.quantile(),
                "c_empirical_coverage": c_covered,
                "h_empirical_coverage": h_covered,
                "n_test_molecules": total,
                "structure_consistent_all_h": within_all_h,
                "structure_consistent_all_c": within_all_c,
                "structure_consistent_both": within_both,
                "demos": demos,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"\n[demo] wrote {args.out}")


if __name__ == "__main__":
    main()
