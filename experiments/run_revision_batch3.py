"""Third experiment batch addressing self-peer-review findings:

1. Main experiment at K=16, 30 epochs, 3 seeds (the current abstract is K=8)
2. Lambda (ssl_weight) sweep at K=16, seed 0: lambda in {0.25, 0.5, 1.0, 2.0}
3. Axis-aligned K=2 ablation (Π along H-axis and C-axis only, not random)
4. Unified conformal + chemistry demo + negative control on ONE retrained K=16 model,
   writing the results to the same JSON for consistency.

Writes results_2d/revision_batch3.json and updates references so later rewrites
can pull from a single source of truth.
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

from src.nmr2d.conformal import ConformalCalibrator
from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.losses_2d import axis_aligned_sort_match_loss_2d, sliced_sort_match_loss_2d
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D, HSQCDataset, compute_target_stats, pad_collate, per_atom_c_loss, set_seed,
    train_variant,
)
from experiments.run_2d_experiment import split_indices


DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _labeled_split(train_idx, frac, seed):
    rng = random.Random(seed + 1)
    s = train_idx.copy()
    rng.shuffle(s)
    n = max(1, int(len(s) * frac))
    return s[:n], s[n:]


def main_k16(dataset, molecules, seeds=(0, 1, 2)):
    per_seed = {}
    for seed in seeds:
        print(f"\n========== K=16 main  seed {seed} ==========")
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
        cfg = Config2D(
            variant="sort_match_ssl_2d", hidden=192, n_layers=4, dropout=0.1,
            lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=30, ssl_weight=0.5,
            labeled_frac=0.1, K_directions=16, seed=seed,
            c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
        )
        log = ROOT / "experiments" / "results_2d" / f"k16_seed{seed}.json"
        r = train_variant(cfg, dataset, train_idx, val_idx, test_idx, labeled, unlabeled, log_path=log)
        per_seed[seed] = {"test_c_mae": r["test_c_mae"], "test_h_mae": r["test_h_mae"], "elapsed": r["elapsed_sec"]}
        print(f"  K=16 seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
    return per_seed


def lambda_sweep(dataset, molecules, seed=0):
    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
    labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
    out = {}
    for lam in [0.25, 0.5, 1.0, 2.0]:
        print(f"\n---- lambda={lam} ----")
        cfg = Config2D(
            variant="sort_match_ssl_2d", hidden=192, n_layers=4, epochs=20,
            labeled_frac=0.1, ssl_weight=lam, K_directions=16, seed=seed,
            c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
        )
        log = ROOT / "experiments" / "results_2d" / f"lambda_{lam}.json"
        r = train_variant(cfg, dataset, train_idx, val_idx, test_idx, labeled, unlabeled, log_path=log)
        out[lam] = {"c_mae": r["test_c_mae"], "h_mae": r["test_h_mae"]}
        print(f"  lambda={lam}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
    return out


class AxisAlignedTrainer:
    """Stand-in that replaces sliced_sort_match_loss_2d with axis_aligned_sort_match_loss_2d
    in the same training loop, for a K=2 axis-aligned ablation.
    """
    def __init__(self, dataset, molecules, seed=0):
        self.dataset = dataset
        self.molecules = molecules
        self.seed = seed

    def run(self):
        train_idx, val_idx, test_idx = split_indices(len(self.molecules), 0.8, 0.1, self.seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(self.dataset, train_idx)
        labeled, unlabeled = _labeled_split(train_idx, 0.1, self.seed)
        cfg = Config2D(
            variant="sort_match_ssl_2d", hidden=192, n_layers=4, epochs=20,
            labeled_frac=0.1, ssl_weight=0.5, K_directions=2, seed=self.seed,
            c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
        )
        set_seed(self.seed)
        in_dim = self.dataset[0]["x"].shape[1]
        model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        cm = torch.tensor(c_mean, device=DEVICE); cs = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)
        hm = torch.tensor(h_mean, device=DEVICE); hs = torch.tensor(h_std, device=DEVICE).clamp_min(1e-3)
        labeled_lookup = {self.dataset.molecules[i].nmr_id for i in labeled}
        train_loader = DataLoader(tud.Subset(self.dataset, labeled + unlabeled),
                                  batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(tud.Subset(self.dataset, val_idx),
                                batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
        test_loader = DataLoader(tud.Subset(self.dataset, test_idx),
                                 batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

        best_val = math.inf; best_state = None
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
                    fill = torch.where(hsqc_mask[unlab], hsqc_c_atoms[unlab],
                                       hsqc_c_atoms[unlab].new_full(hsqc_c_atoms[unlab].shape, max_n + 1))
                    sorted_hc, _ = torch.sort(fill, dim=-1)
                    safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                    pred_h = h_pred[unlab].gather(1, safe_hc); pred_c = c_pred[unlab].gather(1, safe_hc)
                    pred_set = torch.stack([pred_h, pred_c], dim=-1)
                    target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                    ssl = axis_aligned_sort_match_loss_2d(pred_set, target_set, hsqc_mask[unlab], kind="mse")
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

        # Test MAE
        model.eval()
        e_c = 0.0; n_c = 0.0; e_h = 0.0; n_h = 0.0
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
        return {"c_mae": e_c / max(n_c, 1), "h_mae": e_h / max(n_h, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "revision_batch3.json")
    parser.add_argument("--skip", nargs="+", default=[])
    args = parser.parse_args()

    print(f"[batch3] loading dataset")
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    out = {"n_molecules": len(molecules)}

    if "k16" not in args.skip:
        out["k16_main"] = main_k16(dataset, molecules, seeds=(0, 1, 2))
        c_vals = [v["test_c_mae"] for v in out["k16_main"].values()]
        h_vals = [v["test_h_mae"] for v in out["k16_main"].values()]
        out["k16_agg"] = {
            "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
            "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
        }
        print(f"\n  K=16 AGG:  C {out['k16_agg']['c_mean']:.3f} ± {out['k16_agg']['c_std']:.3f}"
              f"  H {out['k16_agg']['h_mean']:.3f} ± {out['k16_agg']['h_std']:.3f}")

    if "lambda" not in args.skip:
        out["lambda_sweep"] = lambda_sweep(dataset, molecules, seed=0)

    if "axis" not in args.skip:
        print(f"\n========== axis-aligned K=2 ablation ==========")
        r = AxisAlignedTrainer(dataset, molecules, seed=0).run()
        out["axis_aligned_k2"] = r
        print(f"  axis-K=2: C {r['c_mae']:.3f}  H {r['h_mae']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
