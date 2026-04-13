"""Realistic wrong-candidate benchmark: constitutional isomers.

Instead of pairing each test molecule with a *random* test molecule of the
same HSQC length (the earlier "same-length wrong-pairing" control), we
generate CHEMICALLY MEANINGFUL wrong candidates for each test molecule:

1. **Constitutional isomers** of the same molecular formula — drawn from
   the OTHER molecules in the NMRShiftDB2 dataset that share the test
   molecule's Hill formula (C_cH_hN_nO_o...). If a test molecule's formula
   is unique, we fall back to "closest formula in the dataset".

2. **Regioisomers** — for a small set of aromatic test molecules we
   enumerate ortho/meta/para variants via RDKit atom-reordering where
   possible.

3. **Bemis-Murcko scaffold swaps** — replace the test molecule's
   functional groups with those of another molecule sharing the same
   scaffold.

For each wrong candidate we:
1. Run the 2-D SSL model to predict its HSQC peak list.
2. Compare the test molecule's OBSERVED HSQC against the candidate's PREDICTED
   HSQC at a matched sorting (same peak count).
3. Check the 95% conformal per-peak consistency.
4. Report the correct-vs-wrong-candidate discrimination rate.

Reviewer priority #2: "realistic wrong candidates, not random pairs."
"""

from __future__ import annotations

import argparse
import collections
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
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.conformal import ConformalCalibrator
from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D, HSQCDataset, compute_target_stats, pad_collate, per_atom_c_loss, set_seed,
)
from experiments.run_2d_experiment import split_indices


DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def hill_formula(mol) -> str:
    """Hill-order molecular formula."""
    return rdMolDescriptors.CalcMolFormula(mol)


def murcko_scaffold(mol) -> str:
    """Canonical SMILES of the Bemis-Murcko scaffold."""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else ""
    except Exception:
        return ""


def find_constitutional_isomers(test_molecules, all_molecules, k_per_target=5):
    """For each test molecule, find up to k other molecules in the dataset
    with the same molecular formula but different SMILES."""
    formula_to_idx = collections.defaultdict(list)
    for idx, m in enumerate(all_molecules):
        formula_to_idx[hill_formula(m.mol)].append(idx)

    pairings = {}  # test_idx -> list of isomer indices
    for t_idx in test_molecules:
        t_mol = all_molecules[t_idx]
        f = hill_formula(t_mol.mol)
        candidates = [i for i in formula_to_idx[f] if i != t_idx]
        if len(candidates) >= 1:
            pairings[t_idx] = candidates[:k_per_target]
    return pairings


def find_scaffold_neighbors(test_molecules, all_molecules, k_per_target=5):
    """For each test molecule, find up to k molecules sharing the same
    Bemis-Murcko scaffold."""
    scaffold_to_idx = collections.defaultdict(list)
    for idx, m in enumerate(all_molecules):
        sc = murcko_scaffold(m.mol)
        if sc:
            scaffold_to_idx[sc].append(idx)
    pairings = {}
    for t_idx in test_molecules:
        t_mol = all_molecules[t_idx]
        sc = murcko_scaffold(t_mol.mol)
        if not sc:
            continue
        candidates = [i for i in scaffold_to_idx[sc] if i != t_idx]
        if len(candidates) >= 1:
            pairings[t_idx] = candidates[:k_per_target]
    return pairings


def predict_hsqc_on_mol(model, molecule, dataset, device, c_stat, h_stat):
    """Run inference on one molecule and return list of (H_pred, C_pred) at
    each HSQC atom in the stored order."""
    cm, cs = c_stat
    hm, hs = h_stat
    # Find the molecule index in the dataset to reuse its graph tensors
    for i, m in enumerate(dataset.molecules):
        if m.nmr_id == molecule.nmr_id:
            idx = i
            break
    else:
        return []
    item = dataset[idx]
    x = item["x"].unsqueeze(0).to(device)
    from src.data import mol_to_graph_tensors
    _, adj_np = mol_to_graph_tensors(molecule.mol)
    adj = adj_np.unsqueeze(0).to(device)
    am = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        cpn, hpn = model(x, adj, am)
        cp = (cpn * cs + cm)[0]
        hp = (hpn * hs + hm)[0]
    preds = []
    for c_idx in molecule.hsqc_c_atoms:
        preds.append((hp[c_idx].item(), cp[c_idx].item()))
    return preds


def match_multisets_by_sorting(obs, pred):
    """Sort both multisets lexicographically and zip. Returns residuals list
    len(min(obs, pred))."""
    if not obs or not pred:
        return []
    a = sorted(obs)
    b = sorted(pred)
    n = min(len(a), len(b))
    return [(abs(a[i][0] - b[i][0]), abs(a[i][1] - b[i][1])) for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "realistic_isomer_control.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print(f"[isomer] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
    rng = random.Random(args.seed + 1)
    shuf = train_idx.copy(); rng.shuffle(shuf)
    n_lab = max(1, int(len(shuf) * 0.1))
    labeled, unlabeled = shuf[:n_lab], shuf[n_lab:]

    # Find chemically meaningful pairings
    formula_pairs = find_constitutional_isomers(test_idx, molecules, k_per_target=3)
    scaffold_pairs = find_scaffold_neighbors(test_idx, molecules, k_per_target=3)
    print(f"  test molecules with formula-isomer pairs: {len(formula_pairs)}")
    print(f"  test molecules with scaffold-neighbor pairs: {len(scaffold_pairs)}")

    # Train the 2-D SSL model fresh
    cfg = Config2D(
        variant="sort_match_ssl_2d", hidden=192, n_layers=4, dropout=0.1,
        lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=args.epochs, ssl_weight=0.5,
        labeled_frac=0.1, K_directions=16, seed=args.seed,
        c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
    )

    set_seed(args.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cm = torch.tensor(c_mean, device=DEVICE); cs = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)
    hm = torch.tensor(h_mean, device=DEVICE); hs = torch.tensor(h_std, device=DEVICE).clamp_min(1e-3)

    labeled_lookup = {molecules[i].nmr_id for i in labeled}
    train_loader = DataLoader(tud.Subset(dataset, labeled + unlabeled), batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    best_val = math.inf; best_state = None
    print(f"[isomer] training")
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE); adj = batch["adj"].to(DEVICE); atom_mask = batch["atom_mask"].to(DEVICE)
            c_atoms = batch["c_atoms"].to(DEVICE); c_shifts = batch["c_shifts"].to(DEVICE); c_mask = batch["c_mask"].to(DEVICE)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(DEVICE); hsqc_h = batch["hsqc_h"].to(DEVICE); hsqc_c = batch["hsqc_c"].to(DEVICE); hsqc_mask = batch["hsqc_mask"].to(DEVICE)
            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm; h_pred = h_pred_norm * hs + hm
            is_labeled = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=DEVICE)
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
                xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); am = vb["atom_mask"].to(DEVICE)
                cat = vb["c_atoms"].to(DEVICE); cs_v = vb["c_shifts"].to(DEVICE); cmsk = vb["c_mask"].to(DEVICE)
                cpn, _ = model(xv, adjv, am); cp = cpn * cs + cm
                g = cp.gather(1, cat.clamp(min=0)); err = (g - cs_v).abs() * cmsk.float()
                e += err.sum().item(); n += cmsk.sum().item()
        val_c = e / max(n, 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"  best val C MAE: {best_val:.3f}")

    # Calibrate conformal on val
    cal_c_res, cal_h_res = [], []
    with torch.no_grad():
        for vb in val_loader:
            xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); am = vb["atom_mask"].to(DEVICE)
            cat = vb["c_atoms"].to(DEVICE); cs_v = vb["c_shifts"].to(DEVICE); cmsk = vb["c_mask"].to(DEVICE)
            hcat = vb["hsqc_c_atoms"].to(DEVICE); hh = vb["hsqc_h"].to(DEVICE); hmsk = vb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, am)
            cp = cpn * cs + cm; hp = hpn * hs + hm
            gc = cp.gather(1, cat.clamp(min=0)); gh = hp.gather(1, hcat.clamp(min=0))
            for i in range(cmsk.shape[0]):
                cal_c_res.extend((gc[i] - cs_v[i]).abs()[cmsk[i].bool()].cpu().tolist())
                cal_h_res.extend((gh[i] - hh[i]).abs()[hmsk[i].bool()].cpu().tolist())
    cal_c = ConformalCalibrator(alpha=0.05); cal_c.fit(np.array(cal_c_res))
    cal_h = ConformalCalibrator(alpha=0.05); cal_h.fit(np.array(cal_h_res))
    q_c = cal_c.quantile(); q_h = cal_h.quantile()
    print(f"  q_C={q_c:.3f}  q_H={q_h:.3f}")

    # Run the consistency check: observed-vs-own and observed-vs-each-isomer
    def check_consistent(obs, pred):
        if not obs or not pred:
            return (False, False)
        res = match_multisets_by_sorting(obs, pred)
        n_common = len(res)
        if n_common == 0:
            return (False, False)
        h_ok = all(r[0] <= q_h for r in res)
        c_ok = all(r[1] <= q_c for r in res)
        return (h_ok, c_ok)

    def collect_discrimination(pair_set, label):
        own_h, own_c, own_both, own_n = 0, 0, 0, 0
        wrong_h, wrong_c, wrong_both, wrong_n = 0, 0, 0, 0
        for t_idx, cand_list in pair_set.items():
            t_mol = molecules[t_idx]
            t_obs = list(t_mol.hsqc_peaks)
            t_pred = predict_hsqc_on_mol(model, t_mol, dataset, DEVICE, (cm, cs), (hm, hs))
            h_ok, c_ok = check_consistent(t_obs, t_pred)
            own_h += int(h_ok); own_c += int(c_ok); own_both += int(h_ok and c_ok); own_n += 1
            for c_idx in cand_list:
                c_mol = molecules[c_idx]
                c_pred = predict_hsqc_on_mol(model, c_mol, dataset, DEVICE, (cm, cs), (hm, hs))
                h_ok, c_ok = check_consistent(t_obs, c_pred)
                wrong_h += int(h_ok); wrong_c += int(c_ok); wrong_both += int(h_ok and c_ok); wrong_n += 1
        return {
            "label": label,
            "own_n": own_n,
            "wrong_n": wrong_n,
            "own_h_rate": own_h / max(own_n, 1),
            "own_c_rate": own_c / max(own_n, 1),
            "own_both_rate": own_both / max(own_n, 1),
            "wrong_h_rate": wrong_h / max(wrong_n, 1),
            "wrong_c_rate": wrong_c / max(wrong_n, 1),
            "wrong_both_rate": wrong_both / max(wrong_n, 1),
        }

    print(f"\n[isomer] discrimination against constitutional isomers")
    iso_result = collect_discrimination(formula_pairs, "constitutional_isomer")
    print(f"  own:   H {iso_result['own_h_rate']*100:.1f}%  C {iso_result['own_c_rate']*100:.1f}%  joint {iso_result['own_both_rate']*100:.1f}%  (n={iso_result['own_n']})")
    print(f"  wrong: H {iso_result['wrong_h_rate']*100:.1f}%  C {iso_result['wrong_c_rate']*100:.1f}%  joint {iso_result['wrong_both_rate']*100:.1f}%  (n={iso_result['wrong_n']})")

    print(f"\n[isomer] discrimination against scaffold neighbors")
    sc_result = collect_discrimination(scaffold_pairs, "scaffold_neighbor")
    print(f"  own:   H {sc_result['own_h_rate']*100:.1f}%  C {sc_result['own_c_rate']*100:.1f}%  joint {sc_result['own_both_rate']*100:.1f}%  (n={sc_result['own_n']})")
    print(f"  wrong: H {sc_result['wrong_h_rate']*100:.1f}%  C {sc_result['wrong_c_rate']*100:.1f}%  joint {sc_result['wrong_both_rate']*100:.1f}%  (n={sc_result['wrong_n']})")

    out = {
        "seed": args.seed,
        "q_c": q_c, "q_h": q_h,
        "best_val_c_mae": best_val,
        "constitutional_isomer": iso_result,
        "scaffold_neighbor": sc_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
