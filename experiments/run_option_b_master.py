"""Master orchestrator — Option B revision pipeline for Nature Communications.

Runs S2 + S3 of the revision plan sequentially in one background process:

  S2:
    P1.1  main headline rerun at lambda=2.0, K=16, 3 seeds, 30 epochs
    P1.2a 3-seed K-sweep {2, 4, 8, 16, 32}, 20 epochs
    P1.2b 3-seed lambda-sweep {0.25, 0.5, 1.0, 2.0}, 20 epochs
    P1.5  scaffold-OOD split with Bemis-Murcko stratification (3 seeds)

  S3:
    P2.2  combined-supervision: full ¹³C labels on ALL training molecules
          PLUS 2D SSL loss on unassigned HSQC targets
    P2.5  stop-gradient ablation (SSL gradient does NOT flow into ¹³C head
          through shared encoder)
    P2.3  pretrain-then-finetune transfer baseline:
          (a) pretrain on full ¹³C labels of 1542 mols,
          (b) finetune on 10% labeled + SSL 2D on remainder

All results go to experiments/results_2d/option_b_*.json with flush=True
logging.
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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    train_variant,
)
from experiments.run_2d_experiment import split_indices


DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def log(msg):
    print(msg, flush=True)


def _labeled_split(train_idx, frac, seed):
    rng = random.Random(seed + 1)
    s = train_idx.copy()
    rng.shuffle(s)
    n = max(1, int(len(s) * frac))
    return s[:n], s[n:]


# ---------------------------------------------------------------------------
# Generic training utility (used by S3 variants where train_variant isn't
# flexible enough)
# ---------------------------------------------------------------------------

def train_flexible(
    molecules,
    dataset,
    train_idx,
    val_idx,
    test_idx,
    labeled_frac,
    labeled_override,
    unlabeled_override,
    cfg,
    *,
    use_full_c_labels: bool = False,
    stop_grad_ssl_to_c: bool = False,
):
    """Train a dual-head 2-D SSL model with flexible options.

    use_full_c_labels: when True, compute the ¹³C supervised loss on EVERY
      molecule in the train split (not only the 10% labeled subset). The SSL
      loss is still computed on the unlabeled_override subset.

    stop_grad_ssl_to_c: when True, detach the ¹³C prediction tensor used by
      the SSL sliced loss so the SSL gradient only flows into the ¹H head
      and the shared encoder (but not the ¹³C head).
    """
    set_seed(cfg.seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cm = torch.tensor(cfg.c_mean, device=DEVICE)
    cs = torch.tensor(cfg.c_std, device=DEVICE).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=DEVICE)
    hs = torch.tensor(cfg.h_std, device=DEVICE).clamp_min(1e-3)

    labeled = labeled_override
    unlabeled = unlabeled_override
    labeled_lookup = {molecules[i].nmr_id for i in labeled}
    all_train = labeled + unlabeled

    train_loader = DataLoader(tud.Subset(dataset, all_train), batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(tud.Subset(dataset, test_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)

    best_val = math.inf
    best_state = None
    t0 = time.time()
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

            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm
            h_pred = h_pred_norm * hs + hm

            is_labeled = torch.tensor([m in labeled_lookup for m in batch["ids"]], device=DEVICE)

            sup_loss = c_pred.new_tensor(0.0)
            ssl_loss = c_pred.new_tensor(0.0)

            if use_full_c_labels:
                sup_loss = per_atom_c_loss(c_pred, c_atoms, c_shifts, c_mask)
            else:
                lab = is_labeled
                if lab.any():
                    sup_loss = per_atom_c_loss(c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab])

            unlab = ~is_labeled
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
                if stop_grad_ssl_to_c:
                    pred_c_at_c = pred_c_at_c.detach()
                pred_set = torch.stack([pred_h_at_c, pred_c_at_c], dim=-1)
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl_loss = sliced_sort_match_loss_2d(
                    pred_set, target_set, hsqc_mask[unlab], K=cfg.K_directions, kind="mse"
                )

            total = sup_loss + cfg.ssl_weight * ssl_loss
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # Validation
        model.eval()
        e, n = 0.0, 0.0
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(DEVICE)
                adjv = vb["adj"].to(DEVICE)
                am = vb["atom_mask"].to(DEVICE)
                cat = vb["c_atoms"].to(DEVICE)
                cs_v = vb["c_shifts"].to(DEVICE)
                cmsk = vb["c_mask"].to(DEVICE)
                cpn, _ = model(xv, adjv, am)
                cp = cpn * cs + cm
                g = cp.gather(1, cat.clamp(min=0))
                err = (g - cs_v).abs() * cmsk.float()
                e += err.sum().item()
                n += cmsk.sum().item()
        val_c = e / max(n, 1)
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Test evaluation
    model.eval()
    e_c, n_c, e_h, n_h = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for tb in test_loader:
            xv = tb["x"].to(DEVICE)
            adjv = tb["adj"].to(DEVICE)
            am = tb["atom_mask"].to(DEVICE)
            cat = tb["c_atoms"].to(DEVICE)
            cs_v = tb["c_shifts"].to(DEVICE)
            cmsk = tb["c_mask"].to(DEVICE)
            hcat = tb["hsqc_c_atoms"].to(DEVICE)
            hh = tb["hsqc_h"].to(DEVICE)
            hmsk = tb["hsqc_mask"].to(DEVICE)
            cpn, hpn = model(xv, adjv, am)
            cp = cpn * cs + cm
            hp = hpn * hs + hm
            gc = cp.gather(1, cat.clamp(min=0))
            gh = hp.gather(1, hcat.clamp(min=0))
            e_c += ((gc - cs_v).abs() * cmsk.float()).sum().item()
            n_c += cmsk.sum().item()
            e_h += ((gh - hh).abs() * hmsk.float()).sum().item()
            n_h += hmsk.sum().item()

    return {
        "test_c_mae": e_c / max(n_c, 1),
        "test_h_mae": e_h / max(n_h, 1),
        "best_val_c_mae": best_val,
        "elapsed": time.time() - t0,
    }


def _base_cfg(K=16, epochs=30, ssl_weight=0.5, seed=0, stats=None):
    cm, cs, hm, hs = stats
    return Config2D(
        variant="sort_match_ssl_2d",
        hidden=192, n_layers=4, dropout=0.1,
        lr=1e-3, weight_decay=1e-5, batch_size=32,
        epochs=epochs, ssl_weight=ssl_weight,
        labeled_frac=0.1, K_directions=K, seed=seed,
        c_mean=cm, c_std=cs, h_mean=hm, h_std=hs,
    )


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------

def scaffold_split_indices(molecules, seed, train_frac=0.8, val_frac=0.1):
    """Bemis-Murcko scaffold-based split.

    Compute canonical scaffold SMILES for each molecule, group by scaffold,
    sort scaffolds by size DESC, put the largest into train until train is
    full, then val, then test. Within each group, shuffle by seed.
    """
    scaffold_groups = {}
    for i, m in enumerate(molecules):
        try:
            sc = MurckoScaffold.GetScaffoldForMol(m.mol)
            key = Chem.MolToSmiles(sc) if sc is not None else ""
        except Exception:
            key = ""
        scaffold_groups.setdefault(key, []).append(i)

    groups = sorted(scaffold_groups.values(), key=lambda g: -len(g))
    rng = random.Random(seed)
    # Shuffle scaffolds OTHER than the largest
    if len(groups) > 1:
        head = groups[0]
        tail = groups[1:]
        rng.shuffle(tail)
        groups = [head] + tail

    n_total = len(molecules)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train, val, test = [], [], []
    for g in groups:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)
    return train, val, test


# ---------------------------------------------------------------------------
# S2 experiments
# ---------------------------------------------------------------------------

def run_p11_lambda2_headline(molecules, dataset, seeds):
    log("\n========== P1.1 λ=2.0 headline (3 seeds × 30 epochs, K=16) ==========")
    out = {}
    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
        stats = (c_mean, c_std, h_mean, h_std)

        cfg = _base_cfg(K=16, epochs=30, ssl_weight=2.0, seed=seed, stats=stats)
        log(f"  seed {seed}: training...")
        r = train_flexible(molecules, dataset, train_idx, val_idx, test_idx,
                           0.1, labeled, unlabeled, cfg)
        log(f"    seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}  ({r['elapsed']:.0f}s)")
        out[seed] = r
    c_vals = [v["test_c_mae"] for v in out.values()]
    h_vals = [v["test_h_mae"] for v in out.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    log(f"  AGG λ=2.0 K=16: C {agg['c_mean']:.3f} ± {agg['c_std']:.3f}  H {agg['h_mean']:.3f} ± {agg['h_std']:.3f}")
    return {"per_seed": out, "aggregate": agg}


def run_p12a_k_sweep(molecules, dataset, seeds):
    log("\n========== P1.2a 3-seed K-sweep {2,4,8,16,32} × 20 epochs ==========")
    out = {}
    for K in [2, 4, 8, 16, 32]:
        per_seed = {}
        for seed in seeds:
            train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
            c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
            labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
            stats = (c_mean, c_std, h_mean, h_std)
            cfg = _base_cfg(K=K, epochs=20, ssl_weight=0.5, seed=seed, stats=stats)
            r = train_flexible(molecules, dataset, train_idx, val_idx, test_idx,
                               0.1, labeled, unlabeled, cfg)
            per_seed[seed] = r
            log(f"  K={K} seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
        c_vals = [v["test_c_mae"] for v in per_seed.values()]
        h_vals = [v["test_h_mae"] for v in per_seed.values()]
        out[K] = {
            "per_seed": per_seed,
            "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
            "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
        }
        log(f"  K={K}: C {out[K]['c_mean']:.3f} ± {out[K]['c_std']:.3f}  H {out[K]['h_mean']:.3f} ± {out[K]['h_std']:.3f}")
    return out


def run_p12b_lambda_sweep(molecules, dataset, seeds):
    log("\n========== P1.2b 3-seed λ-sweep {0.25, 0.5, 1.0, 2.0} × 20 epochs ==========")
    out = {}
    for lam in [0.25, 0.5, 1.0, 2.0]:
        per_seed = {}
        for seed in seeds:
            train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
            c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
            labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
            stats = (c_mean, c_std, h_mean, h_std)
            cfg = _base_cfg(K=16, epochs=20, ssl_weight=lam, seed=seed, stats=stats)
            r = train_flexible(molecules, dataset, train_idx, val_idx, test_idx,
                               0.1, labeled, unlabeled, cfg)
            per_seed[seed] = r
            log(f"  λ={lam} seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
        c_vals = [v["test_c_mae"] for v in per_seed.values()]
        h_vals = [v["test_h_mae"] for v in per_seed.values()]
        out[lam] = {
            "per_seed": per_seed,
            "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
            "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
        }
        log(f"  λ={lam}: C {out[lam]['c_mean']:.3f} ± {out[lam]['c_std']:.3f}  H {out[lam]['h_mean']:.3f} ± {out[lam]['h_std']:.3f}")
    return out


def run_p15_scaffold_ood(molecules, dataset, seeds):
    log("\n========== P1.5 scaffold-OOD split (3 seeds × 30 epochs, K=16) ==========")
    out = {}
    for seed in seeds:
        train_idx, val_idx, test_idx = scaffold_split_indices(molecules, seed)
        log(f"  seed {seed}: scaffold train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
        stats = (c_mean, c_std, h_mean, h_std)
        cfg = _base_cfg(K=16, epochs=30, ssl_weight=0.5, seed=seed, stats=stats)
        r = train_flexible(molecules, dataset, train_idx, val_idx, test_idx,
                           0.1, labeled, unlabeled, cfg)
        out[seed] = r
        log(f"  seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
    c_vals = [v["test_c_mae"] for v in out.values()]
    h_vals = [v["test_h_mae"] for v in out.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    log(f"  AGG scaffold-OOD: C {agg['c_mean']:.3f} ± {agg['c_std']:.3f}  H {agg['h_mean']:.3f} ± {agg['h_std']:.3f}")
    return {"per_seed": out, "aggregate": agg}


# ---------------------------------------------------------------------------
# S3 experiments
# ---------------------------------------------------------------------------

def run_p22_combined(molecules, dataset, seeds):
    log("\n========== P2.2 combined: full ¹³C labels + unassigned HSQC SSL ==========")
    out = {}
    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        # Labeled = train_idx (everyone), unlabeled = train_idx (for SSL pass)
        # This gives the ¹³C sup loss on 100% of training mols AND the SSL
        # loss on 100% of training mols simultaneously.
        stats = (c_mean, c_std, h_mean, h_std)
        cfg = _base_cfg(K=16, epochs=30, ssl_weight=0.5, seed=seed, stats=stats)

        # Use labeled=[] unlabeled=train_idx with use_full_c_labels=True
        r = train_flexible(molecules, dataset, train_idx, val_idx, test_idx,
                           1.0, [], list(train_idx), cfg,
                           use_full_c_labels=True)
        out[seed] = r
        log(f"  seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
    c_vals = [v["test_c_mae"] for v in out.values()]
    h_vals = [v["test_h_mae"] for v in out.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    log(f"  AGG combined: C {agg['c_mean']:.3f} ± {agg['c_std']:.3f}  H {agg['h_mean']:.3f} ± {agg['h_std']:.3f}")
    return {"per_seed": out, "aggregate": agg}


def run_p25_stopgrad(molecules, dataset, seeds):
    log("\n========== P2.5 stop-gradient: SSL grad NOT flowing into ¹³C head ==========")
    out = {}
    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
        stats = (c_mean, c_std, h_mean, h_std)
        cfg = _base_cfg(K=16, epochs=30, ssl_weight=0.5, seed=seed, stats=stats)
        r = train_flexible(molecules, dataset, train_idx, val_idx, test_idx,
                           0.1, labeled, unlabeled, cfg,
                           stop_grad_ssl_to_c=True)
        out[seed] = r
        log(f"  seed {seed}: C {r['test_c_mae']:.3f}  H {r['test_h_mae']:.3f}")
    c_vals = [v["test_c_mae"] for v in out.values()]
    h_vals = [v["test_h_mae"] for v in out.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    log(f"  AGG stop-grad: C {agg['c_mean']:.3f} ± {agg['c_std']:.3f}  H {agg['h_mean']:.3f} ± {agg['h_std']:.3f}")
    return {"per_seed": out, "aggregate": agg}


def run_p23_pretrain_finetune(molecules, dataset, seeds):
    log("\n========== P2.3 pretrain-then-finetune transfer baseline ==========")
    # Phase 1: pretrain a ¹³C-only predictor on ALL 1542 molecules' ¹³C labels
    # for 30 epochs. Phase 2: load the weights, throw away the ¹³C classifier
    # head (keep encoder), add a fresh 2-D SSL dual-head on top, fine-tune on
    # 10% labeled + 90% unlabeled HSQC for 30 epochs.
    out = {}
    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
        c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)
        labeled, unlabeled = _labeled_split(train_idx, 0.1, seed)
        stats = (c_mean, c_std, h_mean, h_std)

        # --- Phase 1: pretrain on full C labels ---
        set_seed(seed)
        in_dim = dataset[0]["x"].shape[1]
        pre_model = NMRDualHeadGNN(in_dim=in_dim, hidden=192, n_layers=4, dropout=0.1).to(DEVICE)
        opt = torch.optim.AdamW(pre_model.parameters(), lr=1e-3, weight_decay=1e-5)
        cm = torch.tensor(c_mean, device=DEVICE)
        cs = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)

        train_loader = DataLoader(tud.Subset(dataset, list(train_idx)), batch_size=32, shuffle=True, collate_fn=pad_collate)
        val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=32, shuffle=False, collate_fn=pad_collate)
        pre_best = math.inf
        pre_state = None
        for epoch in range(30):
            pre_model.train()
            for batch in train_loader:
                x = batch["x"].to(DEVICE); adj = batch["adj"].to(DEVICE); am = batch["atom_mask"].to(DEVICE)
                cat = batch["c_atoms"].to(DEVICE); cshf = batch["c_shifts"].to(DEVICE); cmsk = batch["c_mask"].to(DEVICE)
                cpn, _ = pre_model(x, adj, am)
                cp = cpn * cs + cm
                loss = per_atom_c_loss(cp, cat, cshf, cmsk)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pre_model.parameters(), 5.0)
                opt.step()
            pre_model.eval()
            e, n = 0.0, 0.0
            with torch.no_grad():
                for vb in val_loader:
                    xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); amv = vb["atom_mask"].to(DEVICE)
                    cv = vb["c_atoms"].to(DEVICE); cvs = vb["c_shifts"].to(DEVICE); cm2 = vb["c_mask"].to(DEVICE)
                    cpn, _ = pre_model(xv, adjv, amv)
                    cpred = cpn * cs + cm
                    g = cpred.gather(1, cv.clamp(min=0))
                    err = (g - cvs).abs() * cm2.float()
                    e += err.sum().item(); n += cm2.sum().item()
            v = e / max(n, 1)
            if v < pre_best:
                pre_best = v
                pre_state = {k: vv.cpu().clone() for k, vv in pre_model.state_dict().items()}

        # --- Phase 2: finetune 2-D SSL from pretrained weights ---
        set_seed(seed + 777)
        ft_model = NMRDualHeadGNN(in_dim=in_dim, hidden=192, n_layers=4, dropout=0.1).to(DEVICE)
        if pre_state is not None:
            ft_model.load_state_dict({k: v.to(DEVICE) for k, v in pre_state.items()})
        opt = torch.optim.AdamW(ft_model.parameters(), lr=5e-4, weight_decay=1e-5)
        cm = torch.tensor(c_mean, device=DEVICE)
        cs_t = torch.tensor(c_std, device=DEVICE).clamp_min(1e-3)
        hm = torch.tensor(h_mean, device=DEVICE)
        hs_t = torch.tensor(h_std, device=DEVICE).clamp_min(1e-3)

        labeled_lookup = {molecules[i].nmr_id for i in labeled}
        ft_loader = DataLoader(tud.Subset(dataset, labeled + unlabeled), batch_size=32, shuffle=True, collate_fn=pad_collate)
        ft_val_loader = DataLoader(tud.Subset(dataset, val_idx), batch_size=32, shuffle=False, collate_fn=pad_collate)
        test_loader = DataLoader(tud.Subset(dataset, test_idx), batch_size=32, shuffle=False, collate_fn=pad_collate)
        ft_best = math.inf
        ft_state = None
        for epoch in range(30):
            ft_model.train()
            for batch in ft_loader:
                x = batch["x"].to(DEVICE); adj = batch["adj"].to(DEVICE); am = batch["atom_mask"].to(DEVICE)
                cat = batch["c_atoms"].to(DEVICE); cshf = batch["c_shifts"].to(DEVICE); cmsk = batch["c_mask"].to(DEVICE)
                hcat = batch["hsqc_c_atoms"].to(DEVICE); hh = batch["hsqc_h"].to(DEVICE); hc = batch["hsqc_c"].to(DEVICE); hmk = batch["hsqc_mask"].to(DEVICE)
                cpn, hpn = ft_model(x, adj, am)
                cp = cpn * cs_t + cm
                hp = hpn * hs_t + hm
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
                    ssl = sliced_sort_match_loss_2d(pred_set, target_set, hmk[unlab], K=16, kind="mse")
                total = sup + 0.5 * ssl
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 5.0)
                opt.step()
            ft_model.eval()
            e, n = 0.0, 0.0
            with torch.no_grad():
                for vb in ft_val_loader:
                    xv = vb["x"].to(DEVICE); adjv = vb["adj"].to(DEVICE); amv = vb["atom_mask"].to(DEVICE)
                    cv = vb["c_atoms"].to(DEVICE); cvs = vb["c_shifts"].to(DEVICE); cm2 = vb["c_mask"].to(DEVICE)
                    cpn, _ = ft_model(xv, adjv, amv)
                    cpred = cpn * cs_t + cm
                    g = cpred.gather(1, cv.clamp(min=0))
                    err = (g - cvs).abs() * cm2.float()
                    e += err.sum().item(); n += cm2.sum().item()
            v = e / max(n, 1)
            if v < ft_best:
                ft_best = v
                ft_state = {k: vv.cpu().clone() for k, vv in ft_model.state_dict().items()}
        if ft_state is not None:
            ft_model.load_state_dict({k: v.to(DEVICE) for k, v in ft_state.items()})

        # Test
        ft_model.eval()
        ec, nc, eh, nh = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for tb in test_loader:
                xv = tb["x"].to(DEVICE); adjv = tb["adj"].to(DEVICE); amv = tb["atom_mask"].to(DEVICE)
                cv = tb["c_atoms"].to(DEVICE); cvs = tb["c_shifts"].to(DEVICE); cm2 = tb["c_mask"].to(DEVICE)
                hcat = tb["hsqc_c_atoms"].to(DEVICE); hh = tb["hsqc_h"].to(DEVICE); hmk = tb["hsqc_mask"].to(DEVICE)
                cpn, hpn = ft_model(xv, adjv, amv)
                cp = cpn * cs_t + cm; hp = hpn * hs_t + hm
                gc = cp.gather(1, cv.clamp(min=0)); gh = hp.gather(1, hcat.clamp(min=0))
                ec += ((gc - cvs).abs() * cm2.float()).sum().item(); nc += cm2.sum().item()
                eh += ((gh - hh).abs() * hmk.float()).sum().item(); nh += hmk.sum().item()
        out[seed] = {
            "pretrain_best_val_c_mae": pre_best,
            "finetune_best_val_c_mae": ft_best,
            "test_c_mae": ec / max(nc, 1),
            "test_h_mae": eh / max(nh, 1),
        }
        log(f"  seed {seed}: pretrain C={pre_best:.3f}  finetune val C={ft_best:.3f}  test C {out[seed]['test_c_mae']:.3f}  H {out[seed]['test_h_mae']:.3f}")
    c_vals = [v["test_c_mae"] for v in out.values()]
    h_vals = [v["test_h_mae"] for v in out.values()]
    agg = {
        "c_mean": float(np.mean(c_vals)), "c_std": float(np.std(c_vals)),
        "h_mean": float(np.mean(h_vals)), "h_std": float(np.std(h_vals)),
    }
    log(f"  AGG pretrain-finetune: C {agg['c_mean']:.3f} ± {agg['c_std']:.3f}  H {agg['h_mean']:.3f} ± {agg['h_std']:.3f}")
    return {"per_seed": out, "aggregate": agg}


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "results_2d")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Comma-separated list of stage names to skip")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[master] loading dataset")
    t0 = time.time()
    molecules = build_hsqc_molecules(args.sdf, max_records=20000, max_atoms=60)
    log(f"  {len(molecules)} molecules in {time.time()-t0:.1f}s")
    dataset = HSQCDataset(molecules)

    all_results = {}
    stages = {
        "p11_lambda2_headline": lambda: run_p11_lambda2_headline(molecules, dataset, args.seeds),
        "p12a_k_sweep": lambda: run_p12a_k_sweep(molecules, dataset, args.seeds),
        "p12b_lambda_sweep": lambda: run_p12b_lambda_sweep(molecules, dataset, args.seeds),
        "p15_scaffold_ood": lambda: run_p15_scaffold_ood(molecules, dataset, args.seeds),
        "p22_combined": lambda: run_p22_combined(molecules, dataset, args.seeds),
        "p25_stopgrad": lambda: run_p25_stopgrad(molecules, dataset, args.seeds),
        "p23_pretrain_finetune": lambda: run_p23_pretrain_finetune(molecules, dataset, args.seeds),
    }

    for name, fn in stages.items():
        if name in args.skip:
            log(f"[master] skipping {name}")
            continue
        try:
            t_stage = time.time()
            all_results[name] = fn()
            dt = time.time() - t_stage
            log(f"[master] {name} done in {dt:.0f}s")
            # Incremental write so partial progress is saved
            with (args.out_dir / "option_b_master.json").open("w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            import traceback
            log(f"[master] {name} FAILED: {e}")
            log(traceback.format_exc())
            all_results[name] = {"error": str(e)}

    with (args.out_dir / "option_b_master.json").open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\n[master] wrote {args.out_dir / 'option_b_master.json'}")
    log(f"[master] total time {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
