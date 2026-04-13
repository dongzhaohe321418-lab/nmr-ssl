"""Per-carbon-type error decomposition.

Reuses the chemistry-demo trained model (or retrains) and evaluates test
MAE separately for aromatic carbons, carbonyl/heteroatom-adjacent carbons,
sp3 CH/CH2/CH3, and olefinic carbons. Also per-proton-type for 1H.

Output: error_decomposition.json + a Markdown fragment for the preprint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
from rdkit import Chem
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D, HSQCDataset, compute_target_stats, pad_collate, per_atom_c_loss, set_seed,
)
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from experiments.run_2d_experiment import split_indices


def classify_c(atom: Chem.Atom) -> str:
    """Return a coarse carbon type for a C atom."""
    if atom.GetIsAromatic():
        return "aromatic"
    # Count double bonds to N or O
    dbl_to_het = 0
    dbl_to_c = 0
    for b in atom.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            other = b.GetOtherAtom(atom)
            if other.GetSymbol() in ("O", "N", "S"):
                dbl_to_het += 1
            elif other.GetSymbol() == "C":
                dbl_to_c += 1
    if dbl_to_het > 0:
        return "carbonyl/imino"
    if dbl_to_c > 0:
        return "olefinic"
    # sp3: count H's to distinguish CH3/CH2/CH/quat
    n_h = atom.GetTotalNumHs()
    return {3: "sp3_CH3", 2: "sp3_CH2", 1: "sp3_CH", 0: "sp3_quat"}.get(n_h, "sp3_other")


def classify_h(atom: Chem.Atom) -> str:
    """Return a coarse proton type for a HEAVY atom (C usually). Based on the carbon type."""
    if atom.GetSymbol() != "C":
        return "non_C_H"
    return classify_c(atom).replace("sp3_", "sp3_H_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "error_decomposition.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print(f"[err-decomp] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)
    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)

    rng = random.Random(args.seed + 1)
    shuf = train_idx.copy(); rng.shuffle(shuf)
    n_lab = max(1, int(len(shuf) * 0.1))
    labeled, unlabeled = shuf[:n_lab], shuf[n_lab:]

    cfg = Config2D(
        variant="sort_match_ssl_2d", hidden=192, n_layers=4, dropout=0.1,
        lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=args.epochs, ssl_weight=0.5,
        labeled_frac=0.1, K_directions=8, seed=args.seed,
        c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
    )
    import math
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
    print(f"[err-decomp] training")
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

    # Decompose test errors by carbon type and proton type
    test_loader = DataLoader(tud.Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    c_errs_by_type: dict[str, list[float]] = {}
    h_errs_by_type: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        # iterate by molecule via Subset to access mol object
        for idx in test_idx:
            m = molecules[idx]
            item = dataset[idx]
            x = item["x"].unsqueeze(0).to(device)
            from src.data import mol_to_graph_tensors
            _, adj_np = mol_to_graph_tensors(m.mol)
            adj = adj_np.unsqueeze(0).to(device)
            am = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            cpn, hpn = model(x, adj, am)
            cp = (cpn * cs + cm)[0]; hp = (hpn * hs + hm)[0]
            for c_idx, true_c in m.c_shift_by_atom.items():
                atom = m.mol.GetAtomWithIdx(c_idx)
                ct = classify_c(atom)
                err = abs(cp[c_idx].item() - true_c)
                c_errs_by_type.setdefault(ct, []).append(err)
            for c_idx in m.hsqc_c_atoms:
                atom = m.mol.GetAtomWithIdx(c_idx)
                ht = classify_h(atom)
                true_h = m.h_mean_by_heavy_atom[c_idx]
                err = abs(hp[c_idx].item() - true_h)
                h_errs_by_type.setdefault(ht, []).append(err)

    c_summary = {t: {"n": len(v), "mae": float(np.mean(v)), "p90": float(np.percentile(v, 90))} for t, v in c_errs_by_type.items()}
    h_summary = {t: {"n": len(v), "mae": float(np.mean(v)), "p90": float(np.percentile(v, 90))} for t, v in h_errs_by_type.items()}

    print("\n[err-decomp] 13C MAE by carbon type:")
    for t in sorted(c_summary, key=lambda k: -c_summary[k]["n"]):
        s = c_summary[t]
        print(f"  {t:18s}  n={s['n']:5d}  MAE={s['mae']:.3f}  p90={s['p90']:.3f}")
    print("\n[err-decomp] 1H MAE by carbon type:")
    for t in sorted(h_summary, key=lambda k: -h_summary[k]["n"]):
        s = h_summary[t]
        print(f"  {t:18s}  n={s['n']:5d}  MAE={s['mae']:.3f}  p90={s['p90']:.3f}")

    out = {
        "seed": args.seed,
        "best_val_c_mae": best_val,
        "c_by_type": c_summary,
        "h_by_type": h_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
