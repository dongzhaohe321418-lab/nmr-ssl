"""Stereoisomer-focused chemistry demo.

Picks 5 diastereomer-rich molecules from the held-out test split, re-trains
the 2-D SSL variant, and checks per-peak conformal consistency. Reviewers
specifically flagged that the original demo molecules were "too simple" —
this addresses that by showing HSQC verification on natural-product-like
polycyclic stereochemically complex molecules.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
from rdkit import Chem
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.conformal import ConformalCalibrator
from src.nmr2d.data_2d import build_hsqc_molecules
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


def stereo_complexity(m):
    n_chi = sum(1 for a in m.mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    n_ez = sum(1 for b in m.mol.GetBonds() if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ))
    return n_chi + n_ez


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "stereo_demo.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-demo", type=int, default=5)
    args = parser.parse_args()

    print(f"[stereo-demo] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)

    # Pick stereo-rich molecules from the TEST split
    test_with_score = [(stereo_complexity(molecules[i]), molecules[i].n_hsqc_peaks, i) for i in test_idx]
    # Require at least 3 stereo elements AND at least 4 HSQC peaks
    candidates = [(s, h, i) for (s, h, i) in test_with_score if s >= 3 and h >= 4]
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    picks = [i for _, _, i in candidates[: args.n_demo]]
    print(f"  stereo candidates: {len(candidates)}   chose {len(picks)} demo molecules")

    # Retrain 2-D SSL once
    rng = random.Random(args.seed + 1)
    train_shuffled = train_idx.copy()
    rng.shuffle(train_shuffled)
    n_lab = max(1, int(len(train_shuffled) * 0.1))
    labeled = train_shuffled[:n_lab]
    unlabeled = train_shuffled[n_lab:]

    cfg = Config2D(
        variant="sort_match_ssl_2d", hidden=192, n_layers=4, dropout=0.1,
        lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=30, ssl_weight=0.5,
        labeled_frac=0.1, K_directions=8, seed=args.seed,
        c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
    )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    set_seed(args.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cm = torch.tensor(c_mean, device=device); cs = torch.tensor(c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(h_mean, device=device); hs = torch.tensor(h_std, device=device).clamp_min(1e-3)

    labeled_lookup = {dataset.molecules[i].nmr_id for i in labeled}
    train_loader = DataLoader(tud.Subset(dataset, labeled + unlabeled), batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    best_val = math.inf
    best_state = None
    print(f"[stereo-demo] training")
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device); adj = batch["adj"].to(device); atom_mask = batch["atom_mask"].to(device)
            c_atoms = batch["c_atoms"].to(device); c_shifts = batch["c_shifts"].to(device); c_mask = batch["c_mask"].to(device)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(device); hsqc_h = batch["hsqc_h"].to(device); hsqc_c = batch["hsqc_c"].to(device); hsqc_mask = batch["hsqc_mask"].to(device)
            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm; h_pred = h_pred_norm * hs + hm
            is_labeled = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=device)
            lab = is_labeled; unlab = ~is_labeled
            sup = c_pred.new_tensor(0.0); ssl = c_pred.new_tensor(0.0)
            if lab.any():
                sup = per_atom_c_loss(c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab])
            if unlab.any():
                max_n = x.size(1)
                fill = torch.where(hsqc_mask[unlab], hsqc_c_atoms[unlab], hsqc_c_atoms[unlab].new_full(hsqc_c_atoms[unlab].shape, max_n + 1))
                sorted_hc, _ = torch.sort(fill, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pred_h = h_pred[unlab].gather(1, safe_hc); pred_c = c_pred[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pred_h, pred_c], dim=-1)
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl = sliced_sort_match_loss_2d(pred_set, target_set, hsqc_mask[unlab], K=cfg.K_directions, kind="mse")
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        e, n = 0.0, 0.0
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(device); adjv = vb["adj"].to(device); am = vb["atom_mask"].to(device)
                cat = vb["c_atoms"].to(device); cs_v = vb["c_shifts"].to(device); cmsk = vb["c_mask"].to(device)
                cpn, _ = model(xv, adjv, am); cp = cpn * cs + cm
                g = cp.gather(1, cat.clamp(min=0)); err = (g - cs_v).abs() * cmsk.float()
                e += err.sum().item(); n += cmsk.sum().item()
        val_c = e / max(n, 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  best val C MAE: {best_val:.3f}")

    # Calibrate conformal on val split
    print(f"[stereo-demo] calibrating conformal on val")
    cal_c_res, cal_h_res = [], []
    model.eval()
    with torch.no_grad():
        for vb in val_loader:
            xv = vb["x"].to(device); adjv = vb["adj"].to(device); am = vb["atom_mask"].to(device)
            cat = vb["c_atoms"].to(device); cs_v = vb["c_shifts"].to(device); cmsk = vb["c_mask"].to(device)
            hcat = vb["hsqc_c_atoms"].to(device); hh = vb["hsqc_h"].to(device); hmsk = vb["hsqc_mask"].to(device)
            cpn, hpn = model(xv, adjv, am); cp = cpn * cs + cm; hp = hpn * hs + hm
            gc = cp.gather(1, cat.clamp(min=0)); gh = hp.gather(1, hcat.clamp(min=0))
            for i in range(cmsk.shape[0]):
                cal_c_res.extend((gc[i] - cs_v[i]).abs()[cmsk[i].bool()].cpu().tolist())
                cal_h_res.extend((gh[i] - hh[i]).abs()[hmsk[i].bool()].cpu().tolist())
    cal_c = ConformalCalibrator(alpha=0.05); cal_c.fit(np.array(cal_c_res))
    cal_h = ConformalCalibrator(alpha=0.05); cal_h.fit(np.array(cal_h_res))
    q_c = cal_c.quantile(); q_h = cal_h.quantile()
    print(f"  q_C={q_c:.3f}  q_H={q_h:.3f}")

    # Predict + consistency for each stereo demo
    demos = []
    print(f"\n[stereo-demo] verifying {len(picks)} diastereomer-rich test molecules\n")
    for idx in picks:
        m = molecules[idx]
        n_chi = sum(1 for a in m.mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
        n_ez = sum(1 for b in m.mol.GetBonds() if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ))
        item = dataset[idx]
        x1 = item["x"].unsqueeze(0).to(device)
        n = x1.shape[1]
        adj_full = torch.zeros(n, n, device=device)
        from src.data import mol_to_graph_tensors
        _, adj_np = mol_to_graph_tensors(m.mol)
        adj_full = adj_np.unsqueeze(0).to(device)
        am_full = torch.ones(1, n, dtype=torch.bool, device=device)
        with torch.no_grad():
            cpn, hpn = model(x1, adj_full, am_full)
            cp = (cpn * cs + cm)[0]; hp = (hpn * hs + hm)[0]
        obs = m.hsqc_peaks
        pred = [(hp[c_idx].item(), cp[c_idx].item()) for c_idx in m.hsqc_c_atoms]
        h_within = [abs(oh - ph_) <= q_h for (oh, _), (ph_, _) in zip(obs, pred)]
        c_within = [abs(oc - pc_) <= q_c for (_, oc), (_, pc_) in zip(obs, pred)]
        worst_h = max(abs(oh - ph_) for (oh, _), (ph_, _) in zip(obs, pred))
        worst_c = max(abs(oc - pc_) for (_, oc), (_, pc_) in zip(obs, pred))
        consistent = all(h_within) and all(c_within)
        demos.append({
            "smiles": m.smiles,
            "nmr_id": m.nmr_id,
            "n_chiral_atoms": n_chi,
            "n_ez_bonds": n_ez,
            "n_atoms": m.n_atoms,
            "n_hsqc_peaks": m.n_hsqc_peaks,
            "worst_h_residual": worst_h,
            "worst_c_residual": worst_c,
            "h_all_within": all(h_within),
            "c_all_within": all(c_within),
            "consistent_at_95": consistent,
        })
        print(f"  {m.smiles[:60]}")
        print(f"    chi={n_chi} ez={n_ez}  hsqc={m.n_hsqc_peaks}  worst ΔH={worst_h:.2f}  worst ΔC={worst_c:.2f}  consistent={consistent}")

    out = {
        "seed": args.seed,
        "best_val_c_mae": best_val,
        "q_c": q_c,
        "q_h": q_h,
        "n_stereo_candidates_in_test": len(candidates),
        "demos": demos,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
