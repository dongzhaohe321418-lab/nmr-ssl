"""Four additional experiments requested by peer review:

1. K-directions sweep: K in {2, 4, 8, 16, 32} at seed 0; show saturation.
2. Noise-injection ablation: add realistic Gaussian noise to the HSQC targets
   (sigma_H in {0, 0.03, 0.10} ppm and sigma_C in {0, 0.5, 2.0} ppm), retrain
   2-D SSL, measure test MAE. Demonstrates robustness to literature-grade
   peak-position uncertainty.
3. Wrong-structure negative control: take the already-trained 2-D SSL model
   and, for each test molecule, pair its observed HSQC with the predicted
   HSQC of a DIFFERENT (randomly chosen) molecule of similar size. Show the
   per-peak conformal consistency check FAILS in those wrong cases.
4. Per-head encoder ablation: duplicate the GIN encoder (one per head) and
   retrain to test whether the 13C <-> 2D gap is a capacity artifact.

Writes results_2d/reviewer_experiments.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tud
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _make_labeled_split(train_idx, labeled_frac, seed):
    rng = random.Random(seed + 1)
    train_shuffled = train_idx.copy()
    rng.shuffle(train_shuffled)
    n_lab = max(1, int(len(train_shuffled) * labeled_frac))
    return train_shuffled[:n_lab], train_shuffled[n_lab:]


def _train_once(
    dataset,
    train_idx,
    val_idx,
    test_idx,
    labeled,
    unlabeled,
    *,
    cfg,
    noise_h=0.0,
    noise_c=0.0,
    separate_heads=False,
):
    set_seed(cfg.seed)
    in_dim = dataset[0]["x"].shape[1]
    if separate_heads:
        model = SeparateHeadNMRGNN(
            in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout
        ).to(DEVICE)
    else:
        model = NMRDualHeadGNN(
            in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout
        ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cm = torch.tensor(cfg.c_mean, device=DEVICE)
    cs = torch.tensor(cfg.c_std, device=DEVICE).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=DEVICE)
    hs = torch.tensor(cfg.h_std, device=DEVICE).clamp_min(1e-3)

    labeled_lookup = {dataset.molecules[i].nmr_id for i in labeled}
    all_indices = labeled + unlabeled

    train_loader = DataLoader(
        tud.Subset(dataset, all_indices),
        batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        tud.Subset(dataset, val_idx),
        batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        tud.Subset(dataset, test_idx),
        batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate,
    )

    best_val = math.inf
    best_state = None
    noise_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 777)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            adj = batch["adj"].to(DEVICE)
            atom_mask = batch["atom_mask"].to(DEVICE)
            c_atoms = batch["c_atoms"].to(DEVICE)
            c_shifts = batch["c_shifts"].to(DEVICE)
            c_mask = batch["c_mask"].to(DEVICE)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(DEVICE)
            hsqc_h = batch["hsqc_h"].to(DEVICE)
            hsqc_c = batch["hsqc_c"].to(DEVICE)
            hsqc_mask = batch["hsqc_mask"].to(DEVICE)

            # Inject noise into HSQC TARGETS (not labels), simulating literature
            # peak-position uncertainty.
            if noise_h > 0:
                eh = torch.randn(hsqc_h.shape, generator=noise_gen).to(DEVICE) * noise_h
                hsqc_h = hsqc_h + eh * hsqc_mask.float()
            if noise_c > 0:
                ec = torch.randn(hsqc_c.shape, generator=noise_gen).to(DEVICE) * noise_c
                hsqc_c = hsqc_c + ec * hsqc_mask.float()

            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm
            h_pred = h_pred_norm * hs + hm

            is_labeled = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=DEVICE)
            lab = is_labeled
            unlab = ~is_labeled

            sup = c_pred.new_tensor(0.0)
            ssl = c_pred.new_tensor(0.0)
            if lab.any():
                sup = per_atom_c_loss(c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab])
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
                    pred_set, target_set, hsqc_mask[unlab],
                    K=cfg.K_directions, kind="mse",
                )
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation
        model.eval()
        errs_c, ns_c = 0.0, 0.0
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE)
                am = vb["atom_mask"].to(DEVICE); cat = vb["c_atoms"].to(DEVICE)
                cs_v = vb["c_shifts"].to(DEVICE); cmsk = vb["c_mask"].to(DEVICE)
                cpn, _ = model(xv, adjv, am)
                cp = cpn * cs + cm
                g = cp.gather(1, cat.clamp(min=0))
                e = (g - cs_v).abs() * cmsk.float()
                errs_c += e.sum().item()
                ns_c += cmsk.sum().item()
        val_c = errs_c / max(ns_c, 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Test MAE
    model.eval()
    errs_c_sum, ns_c_sum = 0.0, 0.0
    errs_h_sum, ns_h_sum = 0.0, 0.0
    with torch.no_grad():
        for tb in test_loader:
            xv = tb["x"].to(DEVICE); adjv = tb["adj"].to(DEVICE)
            am = tb["atom_mask"].to(DEVICE); cat = tb["c_atoms"].to(DEVICE)
            cs_v = tb["c_shifts"].to(DEVICE); cmsk = tb["c_mask"].to(DEVICE)
            hcat = tb["hsqc_c_atoms"].to(DEVICE)
            hh = tb["hsqc_h"].to(DEVICE); hmsk = tb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, am)
            cp = cpn * cs + cm
            hp = hpn * hs + hm
            gc = cp.gather(1, cat.clamp(min=0))
            gh = hp.gather(1, hcat.clamp(min=0))
            errs_c_sum += ((gc - cs_v).abs() * cmsk.float()).sum().item()
            ns_c_sum += cmsk.sum().item()
            errs_h_sum += ((gh - hh).abs() * hmsk.float()).sum().item()
            ns_h_sum += hmsk.sum().item()
    return {
        "test_c_mae": errs_c_sum / max(ns_c_sum, 1),
        "test_h_mae": errs_h_sum / max(ns_h_sum, 1),
        "best_val_c_mae": best_val,
        "model": model,
    }


class SeparateHeadNMRGNN(nn.Module):
    """Two separate GIN encoders — one per readout head — so the heads share
    no representation. Used to test the capacity-split hypothesis."""

    def __init__(self, in_dim, hidden=192, n_layers=4, dropout=0.1):
        super().__init__()
        self.enc_c = NMRDualHeadGNN(in_dim=in_dim, hidden=hidden, n_layers=n_layers, dropout=dropout)
        self.enc_h = NMRDualHeadGNN(in_dim=in_dim, hidden=hidden, n_layers=n_layers, dropout=dropout)

    def forward(self, x, adj, mask):
        c_pred, _ = self.enc_c(x, adj, mask)
        _, h_pred = self.enc_h(x, adj, mask)
        return c_pred, h_pred


def _base_cfg(K=8, epochs=30):
    return Config2D(
        variant="sort_match_ssl_2d",
        hidden=192,
        n_layers=4,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        epochs=epochs,
        ssl_weight=0.5,
        labeled_frac=0.1,
        K_directions=K,
        seed=0,
    )


def run_k_sweep(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats):
    c_mean, c_std, h_mean, h_std = stats
    results = {}
    for K in [2, 4, 8, 16, 32]:
        print(f"\n[k-sweep] K={K}")
        cfg = _base_cfg(K=K, epochs=20)
        cfg.c_mean, cfg.c_std, cfg.h_mean, cfg.h_std = c_mean, c_std, h_mean, h_std
        t0 = time.time()
        r = _train_once(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, cfg=cfg)
        dt = time.time() - t0
        print(f"  K={K}: C MAE = {r['test_c_mae']:.3f}  H MAE = {r['test_h_mae']:.3f}  ({dt:.0f}s)")
        results[K] = {"c_mae": r["test_c_mae"], "h_mae": r["test_h_mae"], "elapsed": dt}
    return results


def run_noise_sweep(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats):
    c_mean, c_std, h_mean, h_std = stats
    noise_levels = [
        ("clean",   0.0,  0.0),
        ("low",     0.03, 0.5),
        ("medium",  0.10, 2.0),
        ("high",    0.20, 4.0),
    ]
    results = {}
    for name, nh, nc in noise_levels:
        print(f"\n[noise] {name}: sigma_H={nh} sigma_C={nc}")
        cfg = _base_cfg(K=8, epochs=20)
        cfg.c_mean, cfg.c_std, cfg.h_mean, cfg.h_std = c_mean, c_std, h_mean, h_std
        t0 = time.time()
        r = _train_once(
            dataset, train_idx, val_idx, test_idx, labeled, unlabeled,
            cfg=cfg, noise_h=nh, noise_c=nc,
        )
        dt = time.time() - t0
        print(f"  {name}: C MAE = {r['test_c_mae']:.3f}  H MAE = {r['test_h_mae']:.3f}  ({dt:.0f}s)")
        results[name] = {
            "sigma_h_ppm": nh, "sigma_c_ppm": nc,
            "c_mae": r["test_c_mae"], "h_mae": r["test_h_mae"], "elapsed": dt,
        }
    return results


def run_separate_heads(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats):
    c_mean, c_std, h_mean, h_std = stats
    cfg = _base_cfg(K=8, epochs=20)
    cfg.c_mean, cfg.c_std, cfg.h_mean, cfg.h_std = c_mean, c_std, h_mean, h_std
    print(f"\n[separate-heads] per-head encoders")
    t0 = time.time()
    r = _train_once(
        dataset, train_idx, val_idx, test_idx, labeled, unlabeled,
        cfg=cfg, separate_heads=True,
    )
    dt = time.time() - t0
    print(f"  separate: C MAE = {r['test_c_mae']:.3f}  H MAE = {r['test_h_mae']:.3f}  ({dt:.0f}s)")
    return {"c_mae": r["test_c_mae"], "h_mae": r["test_h_mae"], "elapsed": dt}


def run_wrong_structure_control(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats):
    """Train the 2D SSL model once, predict HSQC on all test molecules, then
    for each test molecule compare its OBSERVED HSQC against (a) its own
    prediction and (b) the prediction of a different test molecule of similar
    size. Report the fraction of molecules whose per-peak consistency check
    PASSES/FAILS in each case."""
    c_mean, c_std, h_mean, h_std = stats
    cfg = _base_cfg(K=8, epochs=30)
    cfg.c_mean, cfg.c_std, cfg.h_mean, cfg.h_std = c_mean, c_std, h_mean, h_std

    print(f"\n[wrong-struct] training model")
    r = _train_once(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, cfg=cfg)
    model = r["model"]

    cm = torch.tensor(c_mean, device=DEVICE)
    cs = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)
    hm = torch.tensor(h_mean, device=DEVICE)
    hs = torch.tensor(h_std, device=DEVICE).clamp_min(1e-3)

    # First, compute conformal quantiles on val split
    print(f"[wrong-struct] calibrating conformal on val split")
    cal_loader = DataLoader(
        tud.Subset(dataset, val_idx),
        batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate,
    )
    cal_c_res, cal_h_res = [], []
    model.eval()
    with torch.no_grad():
        for vb in cal_loader:
            xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE)
            am = vb["atom_mask"].to(DEVICE)
            cat = vb["c_atoms"].to(DEVICE); cs_v = vb["c_shifts"].to(DEVICE); cmsk = vb["c_mask"].to(DEVICE)
            hcat = vb["hsqc_c_atoms"].to(DEVICE); hh = vb["hsqc_h"].to(DEVICE); hmsk = vb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, am)
            cp = cpn * cs + cm
            hp = hpn * hs + hm
            gc = cp.gather(1, cat.clamp(min=0))
            gh = hp.gather(1, hcat.clamp(min=0))
            for i in range(cmsk.shape[0]):
                cal_c_res.extend((gc[i] - cs_v[i]).abs()[cmsk[i].bool()].cpu().tolist())
                cal_h_res.extend((gh[i] - hh[i]).abs()[hmsk[i].bool()].cpu().tolist())
    cal_c = ConformalCalibrator(alpha=0.05); cal_c.fit(np.array(cal_c_res))
    cal_h = ConformalCalibrator(alpha=0.05); cal_h.fit(np.array(cal_h_res))
    q_c = cal_c.quantile(); q_h = cal_h.quantile()
    print(f"  q_C={q_c:.3f}  q_H={q_h:.3f}")

    # Predict per-test-molecule HSQC (as atom-index tuples) + observed HSQC
    test_loader = DataLoader(
        tud.Subset(dataset, test_idx),
        batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate,
    )
    observed = []   # list of lists of (h, c) per molecule
    predicted = []  # same
    with torch.no_grad():
        for tb in test_loader:
            xv = tb["x"].to(DEVICE); adjv = tb["adj"].to(DEVICE)
            am = tb["atom_mask"].to(DEVICE)
            hcat = tb["hsqc_c_atoms"].to(DEVICE)
            hh = tb["hsqc_h"].to(DEVICE); hc = tb["hsqc_c"].to(DEVICE)
            hmsk = tb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, am)
            cp = cpn * cs + cm
            hp = hpn * hs + hm
            ph = hp.gather(1, hcat.clamp(min=0))
            pc = cp.gather(1, hcat.clamp(min=0))
            B = hh.shape[0]
            for i in range(B):
                mask = hmsk[i].bool()
                obs = list(zip(hh[i][mask].cpu().tolist(), hc[i][mask].cpu().tolist()))
                prd = list(zip(ph[i][mask].cpu().tolist(), pc[i][mask].cpu().tolist()))
                observed.append(obs)
                predicted.append(prd)

    # Own-structure: each molecule's obs vs its own pred
    own_h_pass, own_c_pass, own_both = 0, 0, 0
    n_mol = len(observed)
    for obs, prd in zip(observed, predicted):
        if not obs or not prd:
            continue
        h_ok = all(abs(oh - ph_) <= q_h for (oh, _), (ph_, _) in zip(obs, prd))
        c_ok = all(abs(oc - pc_) <= q_c for (_, oc), (_, pc_) in zip(obs, prd))
        own_h_pass += int(h_ok)
        own_c_pass += int(c_ok)
        own_both += int(h_ok and c_ok)

    # Wrong-structure: pair each molecule with a DIFFERENT test molecule of
    # matching HSQC length (forces the check to be non-trivially comparable).
    rng = random.Random(123)
    wrong_h_pass, wrong_c_pass, wrong_both = 0, 0, 0
    attempts = 0
    for i, obs in enumerate(observed):
        if not obs:
            continue
        # Find another test molecule with the same HSQC length
        candidates = [j for j in range(n_mol) if j != i and len(predicted[j]) == len(obs) and len(predicted[j]) > 0]
        if not candidates:
            continue
        j = rng.choice(candidates)
        prd = predicted[j]
        h_ok = all(abs(oh - ph_) <= q_h for (oh, _), (ph_, _) in zip(obs, prd))
        c_ok = all(abs(oc - pc_) <= q_c for (_, oc), (_, pc_) in zip(obs, prd))
        wrong_h_pass += int(h_ok)
        wrong_c_pass += int(c_ok)
        wrong_both += int(h_ok and c_ok)
        attempts += 1

    print(f"  own:   H pass {own_h_pass}/{n_mol}  C pass {own_c_pass}/{n_mol}  both {own_both}/{n_mol}")
    print(f"  wrong: H pass {wrong_h_pass}/{attempts}  C pass {wrong_c_pass}/{attempts}  both {wrong_both}/{attempts}")
    return {
        "q_c": q_c,
        "q_h": q_h,
        "n_test": n_mol,
        "n_wrong_attempts": attempts,
        "own": {
            "h_pass": own_h_pass, "c_pass": own_c_pass, "both": own_both,
            "h_rate": own_h_pass / n_mol, "c_rate": own_c_pass / n_mol, "both_rate": own_both / n_mol,
        },
        "wrong": {
            "h_pass": wrong_h_pass, "c_pass": wrong_c_pass, "both": wrong_both,
            "h_rate": wrong_h_pass / attempts if attempts else 0,
            "c_rate": wrong_c_pass / attempts if attempts else 0,
            "both_rate": wrong_both / attempts if attempts else 0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "reviewer_experiments.json")
    parser.add_argument("--skip", nargs="+", default=[])
    args = parser.parse_args()

    print(f"[reviewer] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed=0)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
    labeled, unlabeled = _make_labeled_split(train_idx, 0.1, seed=0)
    stats = (c_mean, c_std, h_mean, h_std)

    out = {"n_molecules": len(molecules), "seed": 0}

    if "k" not in args.skip:
        out["k_sweep"] = run_k_sweep(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats)
    if "noise" not in args.skip:
        out["noise_sweep"] = run_noise_sweep(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats)
    if "heads" not in args.skip:
        out["separate_heads"] = run_separate_heads(dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats)
    if "wrong" not in args.skip:
        out["wrong_structure"] = run_wrong_structure_control(
            dataset, train_idx, val_idx, test_idx, labeled, unlabeled, stats
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
