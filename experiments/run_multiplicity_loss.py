"""P_NEW1 — Multiplicity-augmented HSQC loss (novel contribution).

Motivation. In published HSQC peak lists, each cross-peak typically carries a
multiplicity tag (CH, CH₂, CH₃) from multiplicity-edited HSQC or DEPT. This
information is free extra supervision that the raw sort-match loss discards.
We add a per-peak classification head that predicts, for each H-bearing
carbon, its multiplicity class {CH, CH₂, CH₃}. The total loss becomes:

    L = sup_¹³C_MSE
      + λ_ssl · sliced_sort_match_on_(¹H, ¹³C) targets
      + λ_mul · cross_entropy(predicted_multiplicity, target_multiplicity)

where the multiplicity target is derived from the unassigned HSQC as a
PERMUTATION-INVARIANT MULTISET: for each molecule the loss compares the
predicted multiplicity-class COUNTS against the observed counts. This
respects the unassigned-data constraint (no atom-to-peak mapping needed).

Because the multiplicity classes are discrete, we use a differentiable
relaxation: predicted multiplicity is a soft class probability via a small
MLP on the encoder output, and the loss is the L1 distance between the
histogram of predicted class probabilities (summed over H-bearing C atoms,
then renormalized) and the observed one-hot class histogram.

This is the first loss in this paper that uses information beyond the raw
(¹H, ¹³C) shift tuples — it exploits the structural tag that a real HSQC
ALREADY provides for free.
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
import torch.nn as nn
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


def _multiplicity_class(atom):
    """Return 0 for non-H-bearing, 1 for CH, 2 for CH₂, 3 for CH₃."""
    n_h = atom.GetTotalNumHs()
    if n_h == 0:
        return 0
    return min(n_h, 3)


class MultiplicityHead(nn.Module):
    """Small MLP predicting multiplicity class logits from encoder output."""

    def __init__(self, hidden, n_classes=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, features):
        return self.mlp(features)


class DualHeadWithMultiplicity(nn.Module):
    def __init__(self, base, hidden):
        super().__init__()
        self.base = base
        self.mul_head = MultiplicityHead(hidden)

    def forward(self, x, adj, mask):
        # Reproduce the forward pass of NMRDualHeadGNN but capture the encoder output
        z = self.base.embed(x)
        for layer in self.base.layers:
            z = layer(z, adj, mask)
        c_pred = self.base.readout_c(z).squeeze(-1)
        h_pred = self.base.readout_h(z).squeeze(-1)
        mul_logits = self.mul_head(z)  # (B, N, 4)
        return c_pred, h_pred, mul_logits


def histogram_soft_l1_loss(mul_logits, target_counts, mask, n_classes=4):
    """Compute L1 between predicted class histogram and target histogram.

    mul_logits: (B, N, n_classes) — soft class probabilities (pre-softmax)
    target_counts: (B, n_classes) — observed counts of each class in HSQC
    mask: (B, N) — valid atom positions
    """
    probs = torch.softmax(mul_logits, dim=-1)  # (B, N, C)
    mask_f = mask.float().unsqueeze(-1)  # (B, N, 1)
    pred_counts = (probs * mask_f).sum(dim=1)  # (B, C)
    # Normalize both sides to be histograms that sum to 1
    pred_sum = pred_counts.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pred_hist = pred_counts / pred_sum
    tgt_sum = target_counts.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    tgt_hist = target_counts / tgt_sum
    return torch.abs(pred_hist - tgt_hist).sum(dim=-1).mean()


def build_target_counts(molecules_batch, n_classes=4):
    """For each molecule in batch, count HSQC peaks in each multiplicity class."""
    counts = []
    for m in molecules_batch:
        c = [0] * n_classes
        for c_idx in m.hsqc_c_atoms:
            atom = m.mol.GetAtomWithIdx(c_idx)
            cls = _multiplicity_class(atom)
            c[cls] += 1
        counts.append(c)
    return torch.tensor(counts, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "multiplicity_loss.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lambda-mul", type=float, default=1.0, help="Multiplicity loss weight")
    args = parser.parse_args()

    print("[mul-loss] loading dataset", flush=True)
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules", flush=True)
    dataset = HSQCDataset(molecules)

    per_seed = {}
    for seed in args.seeds:
        print(f"\n---- seed {seed} ----", flush=True)
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        rng = random.Random(seed + 1)
        shuf = train_idx.copy(); rng.shuffle(shuf)
        n_lab = max(1, int(len(shuf) * 0.1))
        labeled, unlabeled = shuf[:n_lab], shuf[n_lab:]

        cfg = Config2D(
            variant="sort_match_ssl_2d", hidden=192, n_layers=4, dropout=0.1,
            lr=1e-3, weight_decay=1e-5, batch_size=32, epochs=args.epochs, ssl_weight=2.0,
            labeled_frac=0.1, K_directions=16, seed=seed,
            c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
        )
        set_seed(seed)
        in_dim = dataset[0]["x"].shape[1]
        base = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(DEVICE)
        model = DualHeadWithMultiplicity(base, hidden=cfg.hidden).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        cm = torch.tensor(c_mean, device=DEVICE); cs = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)
        hm = torch.tensor(h_mean, device=DEVICE); hs = torch.tensor(h_std, device=DEVICE).clamp_min(1e-3)

        labeled_lookup = {molecules[i].nmr_id for i in labeled}
        # Need to access molecule objects during training for target_counts
        id2mol = {m.nmr_id: m for m in molecules}

        train_loader = DataLoader(tud.Subset(dataset, labeled + unlabeled), batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
        test_loader = DataLoader(tud.Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

        best_val = math.inf; best_state = None
        t0 = time.time()
        for epoch in range(cfg.epochs):
            model.train()
            for batch in train_loader:
                x = batch["x"].to(DEVICE); adj = batch["adj"].to(DEVICE); am = batch["atom_mask"].to(DEVICE)
                c_atoms = batch["c_atoms"].to(DEVICE); c_shifts = batch["c_shifts"].to(DEVICE); c_mask = batch["c_mask"].to(DEVICE)
                hsqc_c_atoms = batch["hsqc_c_atoms"].to(DEVICE); hsqc_h = batch["hsqc_h"].to(DEVICE); hsqc_c = batch["hsqc_c"].to(DEVICE); hsqc_mask = batch["hsqc_mask"].to(DEVICE)

                c_pred_norm, h_pred_norm, mul_logits = model(x, adj, am)
                c_pred = c_pred_norm * cs + cm; h_pred = h_pred_norm * hs + hm

                is_labeled = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=DEVICE)
                lab = is_labeled; unlab = ~is_labeled
                sup = c_pred.new_tensor(0.0); ssl = c_pred.new_tensor(0.0); mul_loss = c_pred.new_tensor(0.0)

                if lab.any():
                    sup = per_atom_c_loss(c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab])

                # Multiplicity histogram loss on ALL molecules (labeled+unlabeled)
                batch_mols = [id2mol[i] for i in batch["ids"]]
                target_counts = build_target_counts(batch_mols).to(DEVICE)
                mul_loss = histogram_soft_l1_loss(mul_logits, target_counts, am)

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

                total = sup + cfg.ssl_weight * ssl + args.lambda_mul * mul_loss
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            model.eval()
            e, n = 0.0, 0.0
            with torch.no_grad():
                for vb in val_loader:
                    xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); amv = vb["atom_mask"].to(DEVICE)
                    cv = vb["c_atoms"].to(DEVICE); cvs = vb["c_shifts"].to(DEVICE); cm2 = vb["c_mask"].to(DEVICE)
                    cpn, _, _ = model(xv, adjv, amv); cpred = cpn * cs + cm
                    g = cpred.gather(1, cv.clamp(min=0)); err = (g - cvs).abs() * cm2.float()
                    e += err.sum().item(); n += cm2.sum().item()
            val_c = e / max(n, 1)
            if val_c < best_val:
                best_val = val_c
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        # Test
        model.eval()
        e_c, n_c, e_h, n_h = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for tb in test_loader:
                xv = tb["x"].to(DEVICE); adjv = tb["adj"].to(DEVICE); amv = tb["atom_mask"].to(DEVICE)
                cv = tb["c_atoms"].to(DEVICE); cvs = tb["c_shifts"].to(DEVICE); cm2 = tb["c_mask"].to(DEVICE)
                hcat = tb["hsqc_c_atoms"].to(DEVICE); hh = tb["hsqc_h"].to(DEVICE); hmk = tb["hsqc_mask"].to(DEVICE)
                cpn, hpn, _ = model(xv, adjv, amv)
                cp = cpn * cs + cm; hp = hpn * hs + hm
                gc = cp.gather(1, cv.clamp(min=0)); gh = hp.gather(1, hcat.clamp(min=0))
                e_c += ((gc - cvs).abs() * cm2.float()).sum().item(); n_c += cm2.sum().item()
                e_h += ((gh - hh).abs() * hmk.float()).sum().item(); n_h += hmk.sum().item()
        r = {
            "test_c_mae": e_c / max(n_c, 1),
            "test_h_mae": e_h / max(n_h, 1),
            "best_val_c_mae": best_val,
            "elapsed": time.time() - t0,
        }
        per_seed[seed] = r
        print(f"  seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}  ({r['elapsed']:.0f}s)", flush=True)

    c_vals = [r["test_c_mae"] for r in per_seed.values()]
    h_vals = [r["test_h_mae"] for r in per_seed.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    print(f"\n========== MULTIPLICITY-LOSS SUMMARY ==========", flush=True)
    print(f"  C: {agg['c_mean']:.3f} ± {agg['c_std']:.3f}", flush=True)
    print(f"  H: {agg['h_mean']:.3f} ± {agg['h_std']:.3f}", flush=True)
    print(f"  compared to λ=2.0 baseline C 4.535 H 0.353", flush=True)

    out = {"per_seed": per_seed, "aggregate": agg, "lambda_mul": args.lambda_mul, "epochs": args.epochs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
