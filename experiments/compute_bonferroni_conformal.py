"""P1.3 — Compute Bonferroni-corrected conformal intervals and molecule-level
empirical coverage.

For a molecule with k HSQC peaks at target molecule-level α = 0.05:
  α_atom = α_mol / (2k)       # Bonferroni: 2 because ¹H and ¹³C both checked
  q_C_atom, q_H_atom = conformal quantiles at those α_atom values

The empirical molecule-level "all peaks within the corrected intervals" rate
is then a valid lower bound on 1 − α_mol (by the Bonferroni inequality
applied over per-atom marginal conformal guarantees).

We retrain the 2-D SSL model fresh (same config as chemistry demo), then
compute the corrected intervals from the val-set residuals, and evaluate
on the test set.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D, HSQCDataset, compute_target_stats, pad_collate, per_atom_c_loss, set_seed,
)
from experiments.run_2d_experiment import split_indices

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def finite_sample_quantile(residuals, alpha):
    """Split-conformal quantile at level 1 - alpha with finite-sample correction."""
    r = np.sort(np.asarray(residuals, dtype=np.float64))
    n = r.size
    if n == 0:
        return float("inf")
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(k, n)
    return float(r[k - 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "bonferroni_conformal.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--alpha-mol", type=float, default=0.05)
    args = parser.parse_args()

    print("[bonferroni] loading dataset", flush=True)
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
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
        labeled_frac=0.1, K_directions=16, seed=args.seed,
        c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
    )

    # Train
    set_seed(args.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cm = torch.tensor(c_mean, device=DEVICE); cs = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)
    hm = torch.tensor(h_mean, device=DEVICE); hs = torch.tensor(h_std, device=DEVICE).clamp_min(1e-3)

    labeled_lookup = {molecules[i].nmr_id for i in labeled}
    train_loader = DataLoader(tud.Subset(dataset, labeled + unlabeled), batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(tud.Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    best_val = math.inf; best_state = None
    print(f"[bonferroni] training {args.epochs} epochs", flush=True)
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE); adj = batch["adj"].to(DEVICE); am = batch["atom_mask"].to(DEVICE)
            cat = batch["c_atoms"].to(DEVICE); cshf = batch["c_shifts"].to(DEVICE); cmsk = batch["c_mask"].to(DEVICE)
            hcat = batch["hsqc_c_atoms"].to(DEVICE); hh = batch["hsqc_h"].to(DEVICE); hc = batch["hsqc_c"].to(DEVICE); hmk = batch["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(x, adj, am)
            cp = cpn * cs + cm; hp = hpn * hs + hm
            is_lab = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=DEVICE)
            sup = cp.new_tensor(0.0); ssl = cp.new_tensor(0.0)
            if is_lab.any():
                sup = per_atom_c_loss(cp[is_lab], cat[is_lab], cshf[is_lab], cmsk[is_lab])
            unlab = ~is_lab
            if unlab.any():
                max_n = x.size(1)
                fill = torch.where(hmk[unlab], hcat[unlab], hcat[unlab].new_full(hcat[unlab].shape, max_n + 1))
                sorted_hc, _ = torch.sort(fill, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pr_h = hp[unlab].gather(1, safe_hc); pr_c = cp[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pr_h, pr_c], dim=-1)
                target_set = torch.stack([hh[unlab], hc[unlab]], dim=-1)
                ssl = sliced_sort_match_loss_2d(pred_set, target_set, hmk[unlab], K=cfg.K_directions, kind="mse")
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        e, n = 0.0, 0.0
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); amv = vb["atom_mask"].to(DEVICE)
                cv = vb["c_atoms"].to(DEVICE); cvs = vb["c_shifts"].to(DEVICE); cm2 = vb["c_mask"].to(DEVICE)
                cpn, _ = model(xv, adjv, amv); cpred = cpn * cs + cm
                g = cpred.gather(1, cv.clamp(min=0)); err = (g - cvs).abs() * cm2.float()
                e += err.sum().item(); n += cm2.sum().item()
        val_c = e / max(n, 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"  best val C MAE: {best_val:.3f}", flush=True)

    # Collect val residuals per atom
    print(f"[bonferroni] collecting val residuals", flush=True)
    c_res_val, h_res_val = [], []
    with torch.no_grad():
        for vb in val_loader:
            xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); amv = vb["atom_mask"].to(DEVICE)
            cv = vb["c_atoms"].to(DEVICE); cvs = vb["c_shifts"].to(DEVICE); cm2 = vb["c_mask"].to(DEVICE)
            hcat = vb["hsqc_c_atoms"].to(DEVICE); hh = vb["hsqc_h"].to(DEVICE); hmk = vb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, amv); cp = cpn * cs + cm; hp = hpn * hs + hm
            gc = cp.gather(1, cv.clamp(min=0)); gh = hp.gather(1, hcat.clamp(min=0))
            for i in range(cm2.shape[0]):
                c_res_val.extend((gc[i] - cvs[i]).abs()[cm2[i].bool()].cpu().tolist())
                h_res_val.extend((gh[i] - hh[i]).abs()[hmk[i].bool()].cpu().tolist())
    c_res_val = np.array(c_res_val); h_res_val = np.array(h_res_val)
    print(f"  val atoms: C {len(c_res_val)}  H {len(h_res_val)}", flush=True)

    # Compute Bonferroni-corrected atom-level alpha per molecule, varying k
    # We precompute a lookup table of quantiles at alpha_atom = alpha_mol / (2k)
    # for k in [1, max_k]
    peaks_per_mol = [molecules[i].n_hsqc_peaks for i in test_idx]
    k_values = sorted(set(peaks_per_mol))
    print(f"  test peaks-per-molecule: min={min(peaks_per_mol)}  median={int(np.median(peaks_per_mol))}  max={max(peaks_per_mol)}", flush=True)

    q_lookup = {}  # k → (q_C, q_H)
    for k in k_values:
        alpha_atom = args.alpha_mol / (2 * k)
        q_c = finite_sample_quantile(c_res_val, alpha_atom)
        q_h = finite_sample_quantile(h_res_val, alpha_atom)
        q_lookup[k] = (q_c, q_h)
    print(f"\n  Bonferroni-corrected quantiles per molecule-k (α_mol={args.alpha_mol}):", flush=True)
    for k in [min(k_values), int(np.median(peaks_per_mol)), max(k_values)]:
        qc, qh = q_lookup[k]
        alpha_atom = args.alpha_mol / (2 * k)
        print(f"    k={k:2d}: α_atom={alpha_atom:.4f}  q_C={qc:.2f} ppm  q_H={qh:.3f} ppm", flush=True)

    # Also baseline (uncorrected) quantile at α_mol
    q_c_uncorr = finite_sample_quantile(c_res_val, args.alpha_mol)
    q_h_uncorr = finite_sample_quantile(h_res_val, args.alpha_mol)
    print(f"\n  Uncorrected (α=0.05): q_C={q_c_uncorr:.2f}  q_H={q_h_uncorr:.3f}", flush=True)

    # Test molecule-level coverage
    n_pass_uncorr = 0
    n_pass_corr = 0
    n_total = 0
    for idx in test_idx:
        m = molecules[idx]
        if not m.hsqc_peaks:
            continue
        item = dataset[idx]
        x = item["x"].unsqueeze(0).to(DEVICE)
        from src.data import mol_to_graph_tensors
        _, adj_np = mol_to_graph_tensors(m.mol)
        adj = adj_np.unsqueeze(0).to(DEVICE)
        amv = torch.ones(1, x.shape[1], dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            cpn, hpn = model(x, adj, amv)
            cp = (cpn * cs + cm)[0]; hp = (hpn * hs + hm)[0]
        preds = [(hp[c_idx].item(), cp[c_idx].item()) for c_idx in m.hsqc_c_atoms]
        obs = m.hsqc_peaks

        # Uncorrected check
        h_ok_u = all(abs(oh - ph_) <= q_h_uncorr for (oh, _), (ph_, _) in zip(obs, preds))
        c_ok_u = all(abs(oc - pc_) <= q_c_uncorr for (_, oc), (_, pc_) in zip(obs, preds))
        if h_ok_u and c_ok_u:
            n_pass_uncorr += 1

        # Corrected check with per-molecule q based on its k
        k = m.n_hsqc_peaks
        qc_k, qh_k = q_lookup.get(k, (q_c_uncorr, q_h_uncorr))
        h_ok_c = all(abs(oh - ph_) <= qh_k for (oh, _), (ph_, _) in zip(obs, preds))
        c_ok_c = all(abs(oc - pc_) <= qc_k for (_, oc), (_, pc_) in zip(obs, preds))
        if h_ok_c and c_ok_c:
            n_pass_corr += 1
        n_total += 1

    print(f"\n  Test molecule-level joint-pass rates:", flush=True)
    print(f"    uncorrected (α_atom=0.05):        {n_pass_uncorr}/{n_total} = {n_pass_uncorr/n_total*100:.1f}%", flush=True)
    print(f"    Bonferroni-corrected (α_mol=0.05): {n_pass_corr}/{n_total} = {n_pass_corr/n_total*100:.1f}%", flush=True)
    print(f"    theory guarantees ≥ {100*(1-args.alpha_mol):.0f}% for corrected version", flush=True)

    out = {
        "seed": args.seed,
        "epochs": args.epochs,
        "alpha_mol": args.alpha_mol,
        "q_uncorrected": {"c": q_c_uncorr, "h": q_h_uncorr},
        "bonferroni_quantiles": {
            str(k): {"alpha_atom": args.alpha_mol / (2 * k), "q_c": q_lookup[k][0], "q_h": q_lookup[k][1]}
            for k in k_values
        },
        "test_rates": {
            "n_total": n_total,
            "n_pass_uncorrected": n_pass_uncorr,
            "rate_uncorrected": n_pass_uncorr / n_total,
            "n_pass_bonferroni": n_pass_corr,
            "rate_bonferroni": n_pass_corr / n_total,
        },
        "val_atoms": {"c": len(c_res_val), "h": len(h_res_val)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
