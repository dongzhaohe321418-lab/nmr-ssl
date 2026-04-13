"""Realistic HSQC degradation experiment (FAST version).

Trains the 2-D SSL model with realistic HSQC degradation applied to the
training-set HSQC targets. Uses a PRECOMPUTED degradation pass at the start
of each epoch so the existing fast training loop can be reused — no per-sample
Python iteration in the inner loop.

Degradation modes (combined into four recipes):
- clean:     no perturbation (baseline)
- realistic: per-peak noise + per-molecule solvent offset + 10% peak dropout
             (preserves atom alignment)
- merge:     adds peak-merging when (|ΔH| <= 0.05, |ΔC| <= 1.0)
- worst:     aggressive noise + 25% dropout + large solvent offsets

For the degradations that BREAK atom alignment (merging), we pad the target
HSQC multiset and fall back to the standard sliced-sort-match loss which is
already permutation-invariant over the multiset. The predicted multiset is
read at the H-bearing C atoms as before. When the degraded target has fewer
peaks than the predicted multiset, we right-pad with zeros + mask.
"""

from __future__ import annotations

import argparse
import copy
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

from src.nmr2d.data_2d import HSQCMolecule, build_hsqc_molecules
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.realistic_hsqc import RealisticHSQCDegrader
from src.nmr2d.train_2d import (
    Config2D, HSQCDataset, compute_target_stats, pad_collate, per_atom_c_loss, set_seed,
)
from experiments.run_2d_experiment import split_indices


DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


DEGRADER_RECIPES = {
    "clean":     RealisticHSQCDegrader(sigma_h=0.0,  sigma_c=0.0, p_drop=0.0,  merge_h=0.0,  merge_c=0.0, offset_std_h=0.0, offset_std_c=0.0),
    "realistic": RealisticHSQCDegrader(sigma_h=0.03, sigma_c=0.5, p_drop=0.10, merge_h=0.0,  merge_c=0.0, offset_std_h=0.05, offset_std_c=1.0),
    "merge":     RealisticHSQCDegrader(sigma_h=0.03, sigma_c=0.5, p_drop=0.10, merge_h=0.05, merge_c=1.0, offset_std_h=0.05, offset_std_c=1.0),
    "worst":     RealisticHSQCDegrader(sigma_h=0.08, sigma_c=1.5, p_drop=0.25, merge_h=0.10, merge_c=2.0, offset_std_h=0.15, offset_std_c=3.0),
}


def _degrade_molecule(orig_peaks, degrader, rng):
    return degrader(list(orig_peaks), rng)


class PrebuiltDegradedDataset(HSQCDataset):
    """At construction, this dataset holds the ORIGINAL molecules. Before each
    epoch call `rebuild(epoch_seed)` to precompute a degraded HSQC peak list
    for every molecule and cache it as overrides. __getitem__ returns the
    overrides for molecules in the degrade_set.
    """

    def __init__(self, molecules, degrader, degrade_indices, seed=0):
        super().__init__(molecules)
        self.degrader = degrader
        self.degrade_set = set(degrade_indices)
        self._base_seed = seed
        self._overrides: dict[int, dict] = {}

    def rebuild(self, epoch: int):
        rng = random.Random(hash((self._base_seed, epoch)) & 0x7FFFFFFF)
        self._overrides.clear()
        for idx in self.degrade_set:
            m = self.molecules[idx]
            orig = list(m.hsqc_peaks)
            degraded = self.degrader(orig, rng)
            if not degraded:
                continue
            h = torch.tensor([p[0] for p in degraded], dtype=torch.float32)
            c = torch.tensor([p[1] for p in degraded], dtype=torch.float32)
            # After degradation (with merging) the atom alignment is destroyed,
            # so we store a placeholder atom-idx tensor of zeros that will be
            # masked out of the supervised C loss. The SSL sliced loss only
            # needs the multiset and a valid mask.
            c_atoms_placeholder = torch.zeros(len(degraded), dtype=torch.long)
            self._overrides[idx] = {
                "hsqc_h": h, "hsqc_c": c, "hsqc_c_atoms": c_atoms_placeholder,
            }

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        o = self._overrides.get(idx)
        if o is not None:
            item["hsqc_h"] = o["hsqc_h"]
            item["hsqc_c"] = o["hsqc_c"]
            item["hsqc_c_atoms"] = o["hsqc_c_atoms"]
        return item


def train_with_degrader(
    molecules, dataset, train_idx, val_idx, test_idx, labeled, unlabeled,
    cfg, degrader, recipe_name,
):
    # Build a wrapper dataset that applies per-epoch degradation ONLY on the
    # unlabeled training portion. Validation and test use the original dataset.
    degraded_ds = PrebuiltDegradedDataset(molecules, degrader, degrade_indices=unlabeled, seed=cfg.seed)

    set_seed(cfg.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cm = torch.tensor(cfg.c_mean, device=DEVICE); cs = torch.tensor(cfg.c_std, device=DEVICE).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=DEVICE); hs = torch.tensor(cfg.h_std, device=DEVICE).clamp_min(1e-3)

    labeled_lookup = {molecules[i].nmr_id for i in labeled}
    val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(tud.Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    best_val = math.inf; best_state = None
    t0 = time.time()
    for epoch in range(cfg.epochs):
        degraded_ds.rebuild(epoch)
        train_loader = DataLoader(tud.Subset(degraded_ds, labeled + unlabeled),
                                  batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
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
                # Labeled molecules use the ORIGINAL C assignment (not degraded)
                sup = per_atom_c_loss(c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab])
            if unlab.any():
                # For unlabeled molecules, the HSQC multiset has been degraded.
                # Use the model's predicted HSQC multiset at H-bearing carbons
                # (selected by original hsqc_c_atoms from the CLEAN dataset) and
                # compare against the degraded target multiset via sliced loss.
                # Because the degraded dataset overrode hsqc_c_atoms to zeros,
                # we need the CLEAN ones. We re-fetch them via the molecule list.
                max_n = x.size(1)
                # Build a per-row tensor of clean hsqc_c_atoms from the
                # molecule objects using the ids in the batch
                clean_hc = torch.full((unlab.sum().item(), hsqc_c_atoms.size(1)), max_n + 1,
                                       dtype=torch.long, device=DEVICE)
                clean_mask = torch.zeros_like(clean_hc, dtype=torch.bool)
                ids_unlab = [batch["ids"][i] for i in range(len(batch["ids"])) if not is_labeled[i]]
                id2mol = {m.nmr_id: m for m in molecules}
                for i, mid in enumerate(ids_unlab):
                    m = id2mol.get(mid)
                    if m is None or not m.hsqc_c_atoms:
                        continue
                    k = min(len(m.hsqc_c_atoms), clean_hc.size(1))
                    clean_hc[i, :k] = torch.tensor(m.hsqc_c_atoms[:k], device=DEVICE)
                    clean_mask[i, :k] = True
                sorted_hc, _ = torch.sort(clean_hc, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pred_h_at_c = h_pred[unlab].gather(1, safe_hc)
                pred_c_at_c = c_pred[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pred_h_at_c, pred_c_at_c], dim=-1)

                # Degraded target has its own shape; pad pred_set or target_set to match
                target_h = hsqc_h[unlab]; target_c = hsqc_c[unlab]; target_mask = hsqc_mask[unlab]
                # pred_set shape (U, Np, 2); target shape (U, Nt, 2)
                Np = pred_set.size(1); Nt = target_h.size(1)
                N = max(Np, Nt)
                if Np < N:
                    pad = torch.zeros(pred_set.size(0), N - Np, 2, device=DEVICE)
                    pred_set = torch.cat([pred_set, pad], dim=1)
                    pad_mask = torch.zeros(pred_set.size(0), N - Np, dtype=torch.bool, device=DEVICE)
                    pred_mask_full = torch.cat([clean_mask[:, :Np], pad_mask], dim=1)
                else:
                    pred_mask_full = clean_mask[:, :N]
                if Nt < N:
                    pad_h = torch.zeros(target_h.size(0), N - Nt, device=DEVICE)
                    pad_c = torch.zeros(target_c.size(0), N - Nt, device=DEVICE)
                    target_h = torch.cat([target_h, pad_h], dim=1)
                    target_c = torch.cat([target_c, pad_c], dim=1)
                    pad_tmask = torch.zeros(target_mask.size(0), N - Nt, dtype=torch.bool, device=DEVICE)
                    target_mask_full = torch.cat([target_mask, pad_tmask], dim=1)
                else:
                    target_mask_full = target_mask[:, :N]
                target_set = torch.stack([target_h, target_c], dim=-1)
                # Combined mask: positions valid in BOTH pred and target
                combined_mask = pred_mask_full & target_mask_full
                if combined_mask.any():
                    ssl = sliced_sort_match_loss_2d(
                        pred_set, target_set, combined_mask,
                        K=cfg.K_directions, kind="mse",
                    )
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation (clean data)
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
        print(f"  [{recipe_name}] epoch {epoch+1}/{cfg.epochs} val C MAE {val_c:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Test MAE on clean test
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

    elapsed = time.time() - t0
    return {
        "recipe": recipe_name,
        "test_c_mae": e_c / max(n_c, 1),
        "test_h_mae": e_h / max(n_h, 1),
        "best_val_c_mae": best_val,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_2d" / "realistic_hsqc.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    print(f"[realistic-hsqc] loading dataset", flush=True)
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules", flush=True)
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, args.seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
    rng = random.Random(args.seed + 1)
    shuf = train_idx.copy(); rng.shuffle(shuf)
    n_lab = max(1, int(len(shuf) * 0.1))
    labeled, unlabeled = shuf[:n_lab], shuf[n_lab:]

    results = {}
    for recipe_name, degrader in DEGRADER_RECIPES.items():
        print(f"\n---- recipe: {recipe_name} ----", flush=True)
        cfg = Config2D(
            variant="sort_match_ssl_2d", hidden=192, n_layers=4, epochs=args.epochs,
            labeled_frac=0.1, ssl_weight=0.5, K_directions=16, seed=args.seed,
            c_mean=c_mean, c_std=c_std, h_mean=h_mean, h_std=h_std,
        )
        r = train_with_degrader(
            molecules, dataset, train_idx, val_idx, test_idx, labeled, unlabeled,
            cfg, degrader, recipe_name,
        )
        print(f"  {recipe_name}: test C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}  ({r['elapsed']:.0f}s)", flush=True)
        results[recipe_name] = r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"seed": args.seed, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\n============ REALISTIC HSQC SUMMARY ============", flush=True)
    for k, v in results.items():
        print(f"  {k:10s}  C {v['test_c_mae']:.3f}  H {v['test_h_mae']:.3f}", flush=True)


if __name__ == "__main__":
    main()
