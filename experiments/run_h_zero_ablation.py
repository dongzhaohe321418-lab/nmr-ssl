"""P0.1 — H-zero ablation (the central causal-claim audit).

If the paper's headline claim "the ¹H head learns from the 2-D HSQC target
without ever seeing an atom-assigned ¹H label" is actually true, then:

    ZEROING OUT the ¹H column of every unlabeled-split HSQC target
    ===> the ¹H head should COLLAPSE to the random floor (~2.5 ppm).

If instead the ¹H head only modestly degrades (e.g. 0.5 → 1.0 ppm), that would
prove the ¹H head is being trained predominantly by gradient signal leaking
through the shared encoder from the ¹³C supervised loss, and the "unassigned
2-D HSQC unlocks ¹H" story is mostly an artifact.

Protocol:
- Same dataset, splits, hyperparameters, K=16, 3 seeds, 30 epochs as the
  current main experiment.
- The ONLY change: in the 2-D SSL loss, replace every unlabeled-molecule
  HSQC ¹H target with 0.0 before standardization. The ¹³C column is kept.

Reports: 3-seed test ¹H MAE, 3-seed test ¹³C MAE, compared against the main
experiment's K=16 numbers.
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


def _labeled_split(train_idx, frac, seed):
    rng = random.Random(seed + 1)
    s = train_idx.copy(); rng.shuffle(s)
    n = max(1, int(len(s) * frac))
    return s[:n], s[n:]


def train_h_zero(dataset, molecules, seed, epochs=30, K=16):
    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
    labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)

    cfg = Config2D(
        variant="sort_match_ssl_2d", hidden=192, n_layers=4, dropout=0.1,
        lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=epochs, ssl_weight=0.5,
        labeled_frac=0.1, K_directions=K, seed=seed,
        c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
    )
    set_seed(seed)
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
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(DEVICE); adj = batch["adj"].to(DEVICE); atom_mask = batch["atom_mask"].to(DEVICE)
            c_atoms = batch["c_atoms"].to(DEVICE); c_shifts = batch["c_shifts"].to(DEVICE); c_mask = batch["c_mask"].to(DEVICE)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(DEVICE)
            hsqc_h = batch["hsqc_h"].to(DEVICE).clone()  # will zero below
            hsqc_c = batch["hsqc_c"].to(DEVICE)
            hsqc_mask = batch["hsqc_mask"].to(DEVICE)

            # *** THE ABLATION ***: zero out ¹H targets
            # (keep the ¹³C column as the "supervision the model DOES see")
            hsqc_h.zero_()

            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm
            h_pred = h_pred_norm * hs + hm

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
                pred_h_at_c = h_pred[unlab].gather(1, safe_hc)
                pred_c_at_c = c_pred[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pred_h_at_c, pred_c_at_c], dim=-1)
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl = sliced_sort_match_loss_2d(pred_set, target_set, hsqc_mask[unlab], K=cfg.K_directions, kind="mse")
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation (on CLEAN test targets — we only zeroed training)
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

    # Test evaluation on CLEAN targets
    model.eval()
    e_c, n_c, e_h, n_h = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for tb in test_loader:
            xv = tb["x"].to(DEVICE); adjv = tb["adj"].to(DEVICE); am = tb["atom_mask"].to(DEVICE)
            cat = tb["c_atoms"].to(DEVICE); cs_v = tb["c_shifts"].to(DEVICE); cmsk = tb["c_mask"].to(DEVICE)
            hcat = tb["hsqc_c_atoms"].to(DEVICE); hh = tb["hsqc_h"].to(DEVICE); hmsk = tb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, am)
            cp = cpn * cs + cm; hp = hpn * hs + hm
            gc = cp.gather(1, cat.clamp(min=0)); gh = hp.gather(1, hcat.clamp(min=0))
            e_c += ((gc - cs_v).abs() * cmsk.float()).sum().item(); n_c += cmsk.sum().item()
            e_h += ((gh - hh).abs() * hmsk.float()).sum().item(); n_h += hmsk.sum().item()
    return {
        "seed": seed,
        "test_c_mae": e_c / max(n_c, 1),
        "test_h_mae": e_h / max(n_h, 1),
        "best_val_c_mae": best_val,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "h_zero_ablation.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print(f"[h-zero] loading dataset", flush=True)
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules", flush=True)
    dataset = HSQCDataset(molecules)

    per_seed = {}
    for seed in args.seeds:
        print(f"\n---- h-zero seed {seed} ----", flush=True)
        t0 = time.time()
        r = train_h_zero(dataset, molecules, seed, epochs=args.epochs, K=16)
        dt = time.time() - t0
        print(f"  seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}  ({dt:.0f}s)", flush=True)
        per_seed[seed] = r

    c_vals = [r["test_c_mae"] for r in per_seed.values()]
    h_vals = [r["test_h_mae"] for r in per_seed.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    out = {"per_seed": per_seed, "aggregate": agg, "epochs": args.epochs, "K": 16}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\n========== H-ZERO ABLATION SUMMARY ==========", flush=True)
    print(f"  C: {agg['c_mean']:.3f} ± {agg['c_std']:.3f}", flush=True)
    print(f"  H: {agg['h_mean']:.3f} ± {agg['h_std']:.3f}", flush=True)
    print(f"", flush=True)
    print(f"  Compared to baseline K=16 (C 4.87 ± 0.07, H 0.46 ± 0.14)", flush=True)
    print(f"  If H >> 1.5 ppm: central claim HOLDS", flush=True)
    print(f"  If H close to 0.46 ppm: central claim FAILS", flush=True)


if __name__ == "__main__":
    main()
