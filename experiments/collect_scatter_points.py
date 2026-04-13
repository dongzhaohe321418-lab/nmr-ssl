"""Retrain the 2-D SSL model once and collect per-atom (pred, true) pairs
on the test split for ¹³C and ¹H.

Saves to experiments/results_2d/scatter_points.npz. This is a companion data
dump used by docs/2d/figures/fig_scatter_r2.* — intentionally small and
self-contained. Follows the training boilerplate from run_error_decomposition.py
but runs on CPU at reduced epochs (GPU is busy with the label-sweep batch).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nmr2d.data_2d import build_hsqc_molecules
from src.nmr2d.model_2d import NMRDualHeadGNN
from src.nmr2d.train_2d import (
    Config2D,
    HSQCDataset,
    compute_target_stats,
    pad_collate,
    per_atom_c_loss,
    set_seed,
)
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from experiments.run_2d_experiment import split_indices
from src.data import mol_to_graph_tensors


def main():
    sdf = ROOT / "data" / "nmrshiftdb2withsignals.sd"
    out = ROOT / "experiments" / "results_2d" / "scatter_points.npz"
    seed = 0
    # GPU (MPS) busy with label sweep -> force CPU with reduced epochs,
    # per figure-artist brief: "we only need the scatter, not SOTA MAE".
    device_str = "cpu"
    epochs = 10

    print(f"[scatter] device={device_str} epochs={epochs}")
    print("[scatter] loading dataset")
    molecules = build_hsqc_molecules(sdf, max_records=20000, max_atoms=60)
    print(f"  {len(molecules)} molecules")
    dataset = HSQCDataset(molecules)

    train_idx, val_idx, test_idx = split_indices(len(molecules), 0.8, 0.1, seed)
    c_mean, c_std, h_mean, h_std = compute_target_stats(dataset, train_idx)

    rng = random.Random(seed + 1)
    shuf = train_idx.copy()
    rng.shuffle(shuf)
    n_lab = max(1, int(len(shuf) * 0.1))
    labeled, unlabeled = shuf[:n_lab], shuf[n_lab:]

    cfg = Config2D(
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
        K_directions=8,
        seed=seed,
        c_mean=c_mean,
        c_std=c_std,
        h_mean=h_mean,
        h_std=h_std,
    )

    device = torch.device(device_str)
    set_seed(seed)
    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(
        in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cm = torch.tensor(c_mean, device=device)
    cs = torch.tensor(c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(h_mean, device=device)
    hs = torch.tensor(h_std, device=device).clamp_min(1e-3)

    labeled_lookup = {dataset.molecules[i].nmr_id for i in labeled}
    train_loader = DataLoader(
        tud.Subset(dataset, labeled + unlabeled),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        tud.Subset(dataset, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )

    best_val = math.inf
    best_state = None
    print("[scatter] training")
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            c_atoms = batch["c_atoms"].to(device)
            c_shifts = batch["c_shifts"].to(device)
            c_mask = batch["c_mask"].to(device)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(device)
            hsqc_h = batch["hsqc_h"].to(device)
            hsqc_c = batch["hsqc_c"].to(device)
            hsqc_mask = batch["hsqc_mask"].to(device)
            c_pred_norm, h_pred_norm = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm
            h_pred = h_pred_norm * hs + hm
            is_labeled = torch.tensor(
                [m in labeled_lookup for m in batch["ids"]], device=device
            )
            lab = is_labeled
            unlab = ~is_labeled
            sup = c_pred.new_tensor(0.0)
            ssl = c_pred.new_tensor(0.0)
            if lab.any():
                sup = per_atom_c_loss(
                    c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab]
                )
            if unlab.any():
                max_n = x.size(1)
                fill = torch.where(
                    hsqc_mask[unlab],
                    hsqc_c_atoms[unlab],
                    hsqc_c_atoms[unlab].new_full(hsqc_c_atoms[unlab].shape, max_n + 1),
                )
                sorted_hc, _ = torch.sort(fill, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pred_h = h_pred[unlab].gather(1, safe_hc)
                pred_c = c_pred[unlab].gather(1, safe_hc)
                pred_set = torch.stack([pred_h, pred_c], dim=-1)
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl = sliced_sort_match_loss_2d(
                    pred_set,
                    target_set,
                    hsqc_mask[unlab],
                    K=cfg.K_directions,
                    kind="mse",
                )
            total = sup + cfg.ssl_weight * ssl
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        e, n = 0.0, 0.0
        with torch.no_grad():
            for vb in val_loader:
                xv = vb["x"].to(device)
                adjv = vb["adj"].to(device)
                am = vb["atom_mask"].to(device)
                cat = vb["c_atoms"].to(device)
                cs_v = vb["c_shifts"].to(device)
                cmsk = vb["c_mask"].to(device)
                cpn, _ = model(xv, adjv, am)
                cp = cpn * cs + cm
                g = cp.gather(1, cat.clamp(min=0))
                err = (g - cs_v).abs() * cmsk.float()
                e += err.sum().item()
                n += cmsk.sum().item()
        val_c = e / max(n, 1)
        print(f"  epoch {epoch+1:2d}/{cfg.epochs}  val_c_mae={val_c:.3f}")
        if val_c < best_val:
            best_val = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"[scatter] best val C MAE: {best_val:.3f}")

    # Collect per-atom (pred, true) pairs on test split.
    c_pred_list: list[float] = []
    c_true_list: list[float] = []
    h_pred_list: list[float] = []
    h_true_list: list[float] = []

    model.eval()
    with torch.no_grad():
        for idx in test_idx:
            m = molecules[idx]
            item = dataset[idx]
            x = item["x"].unsqueeze(0).to(device)
            _, adj_np = mol_to_graph_tensors(m.mol)
            adj = adj_np.unsqueeze(0).to(device)
            am = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            cpn, hpn = model(x, adj, am)
            cp = (cpn * cs + cm)[0]
            hp = (hpn * hs + hm)[0]
            for c_idx, true_c in m.c_shift_by_atom.items():
                c_pred_list.append(float(cp[c_idx].item()))
                c_true_list.append(float(true_c))
            for c_idx in m.hsqc_c_atoms:
                true_h = m.h_mean_by_heavy_atom[c_idx]
                h_pred_list.append(float(hp[c_idx].item()))
                h_true_list.append(float(true_h))

    c_pred_arr = np.asarray(c_pred_list, dtype=np.float32)
    c_true_arr = np.asarray(c_true_list, dtype=np.float32)
    h_pred_arr = np.asarray(h_pred_list, dtype=np.float32)
    h_true_arr = np.asarray(h_true_list, dtype=np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        c_pred=c_pred_arr,
        c_true=c_true_arr,
        h_pred=h_pred_arr,
        h_true=h_true_arr,
        best_val_c_mae=np.float32(best_val),
        epochs=np.int32(cfg.epochs),
        device=np.array(device_str),
    )
    print(
        f"[scatter] wrote {out}  "
        f"C n={len(c_pred_arr)}  H n={len(h_pred_arr)}"
    )


if __name__ == "__main__":
    main()
