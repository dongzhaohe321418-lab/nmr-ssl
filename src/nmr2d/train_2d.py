"""Training loop for the 2-D NMR experiment.

Three variants share an identical dual-head GNN, identical optimizer, identical
data splits and labeled/unlabeled partition. The only difference is the loss
applied on the unlabeled portion of the training split.

Variants
--------
supervised_1d
    Train on the 10% labeled subset only. Per-atom MSE between predicted 13-C
    shift and the assigned target; no 1-H loss at all.

sort_match_ssl_1d
    Same labeled loss, PLUS a masked 1-D sort-match MSE loss on the 13-C
    predictions at C atoms of the 90% unlabeled molecules. (Same as the main
    paper's method.) No 1-H supervision.

sort_match_ssl_2d  (our new method)
    Same labeled loss, PLUS a sliced-sort-match 2-D loss on the HSQC peak set
    (H_mean_at_C_k, delta_C_k) of the 90% unlabeled molecules. The 2-D loss
    is computed via sliced-Wasserstein with K random directions. This is the
    first method to consume 2-D HSQC peak lists as unassigned SSL supervision.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data import mol_to_graph_tensors
from src.losses import masked_sort_match_loss
from src.nmr2d.data_2d import HSQCMolecule
from src.nmr2d.losses_2d import sliced_sort_match_loss_2d
from src.nmr2d.model_2d import NMRDualHeadGNN


# ------------------------------ dataset ---------------------------------


class HSQCDataset(Dataset):
    def __init__(self, molecules: list[HSQCMolecule]):
        self.molecules = molecules

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> dict:
        m = self.molecules[idx]
        x, adj = mol_to_graph_tensors(m.mol)
        # 13-C atom assignments
        c_atoms = sorted(m.c_shift_by_atom.keys())
        c_shifts = [m.c_shift_by_atom[i] for i in c_atoms]
        # HSQC cross-peaks (carbon atom indices retained for atom-accuracy metric)
        h_mean = [m.h_mean_by_heavy_atom[i] for i in m.hsqc_c_atoms]
        c_at_h = [m.c_shift_by_atom[i] for i in m.hsqc_c_atoms]
        return {
            "x": x,
            "adj": adj,
            "c_atoms": torch.tensor(c_atoms, dtype=torch.long),
            "c_shifts": torch.tensor(c_shifts, dtype=torch.float32),
            "hsqc_c_atoms": torch.tensor(m.hsqc_c_atoms, dtype=torch.long),
            "hsqc_h": torch.tensor(h_mean, dtype=torch.float32),
            "hsqc_c": torch.tensor(c_at_h, dtype=torch.float32),
            "nmr_id": m.nmr_id,
        }


def pad_collate(batch: list[dict]) -> dict:
    B = len(batch)
    n_max = max(item["x"].shape[0] for item in batch)
    k_max = max(item["c_shifts"].shape[0] for item in batch)
    h_max = max(max(item["hsqc_h"].shape[0], 1) for item in batch)
    feat_dim = batch[0]["x"].shape[1]

    x = torch.zeros(B, n_max, feat_dim)
    adj = torch.zeros(B, n_max, n_max)
    atom_mask = torch.zeros(B, n_max, dtype=torch.bool)

    c_atoms = torch.zeros(B, k_max, dtype=torch.long)
    c_shifts = torch.zeros(B, k_max)
    c_mask = torch.zeros(B, k_max, dtype=torch.bool)

    hsqc_c_atoms = torch.zeros(B, h_max, dtype=torch.long)
    hsqc_h = torch.zeros(B, h_max)
    hsqc_c = torch.zeros(B, h_max)
    hsqc_mask = torch.zeros(B, h_max, dtype=torch.bool)

    ids = []
    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        k = item["c_shifts"].shape[0]
        h = item["hsqc_h"].shape[0]
        x[i, :n] = item["x"]
        adj[i, :n, :n] = item["adj"]
        atom_mask[i, :n] = True
        c_atoms[i, :k] = item["c_atoms"]
        c_shifts[i, :k] = item["c_shifts"]
        c_mask[i, :k] = True
        if h > 0:
            hsqc_c_atoms[i, :h] = item["hsqc_c_atoms"]
            hsqc_h[i, :h] = item["hsqc_h"]
            hsqc_c[i, :h] = item["hsqc_c"]
            hsqc_mask[i, :h] = True
        ids.append(item["nmr_id"])

    return {
        "x": x,
        "adj": adj,
        "atom_mask": atom_mask,
        "c_atoms": c_atoms,
        "c_shifts": c_shifts,
        "c_mask": c_mask,
        "hsqc_c_atoms": hsqc_c_atoms,
        "hsqc_h": hsqc_h,
        "hsqc_c": hsqc_c,
        "hsqc_mask": hsqc_mask,
        "ids": ids,
    }


# ------------------------------ training ---------------------------------


@dataclass
class Config2D:
    variant: str  # "supervised_1d" | "sort_match_ssl_1d" | "sort_match_ssl_2d"
    hidden: int = 192
    n_layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 30
    ssl_weight: float = 0.5
    labeled_frac: float = 0.1
    K_directions: int = 8
    seed: int = 0
    c_mean: float = 0.0
    c_std: float = 1.0
    h_mean: float = 0.0
    h_std: float = 1.0


def _select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def per_atom_c_loss(pred: Tensor, target_atom: Tensor, target_shift: Tensor, mask: Tensor) -> Tensor:
    safe = target_atom.clamp(min=0)
    gathered = pred.gather(1, safe)
    sq = (gathered - target_shift) ** 2
    sq = sq * mask.float()
    denom = mask.sum().clamp(min=1).float()
    return sq.sum() / denom


def per_atom_c_mae(pred: Tensor, target_atom: Tensor, target_shift: Tensor, mask: Tensor) -> float:
    safe = target_atom.clamp(min=0)
    gathered = pred.gather(1, safe)
    err = (gathered - target_shift).abs() * mask.float()
    return (err.sum() / mask.sum().clamp(min=1).float()).item()


def compute_target_stats(dataset: HSQCDataset, indices: list[int]) -> tuple[float, float, float, float]:
    c_vals, h_vals = [], []
    for i in indices:
        item = dataset[i]
        c_vals.extend(item["c_shifts"].tolist())
        h_vals.extend(item["hsqc_h"].tolist())
    c_mean = float(np.mean(c_vals)) if c_vals else 0.0
    c_std = float(np.std(c_vals)) if c_vals else 1.0
    h_mean = float(np.mean(h_vals)) if h_vals else 0.0
    h_std = float(np.std(h_vals)) if h_vals else 1.0
    return c_mean, c_std, h_mean, h_std


def evaluate_c_mae(
    model: NMRDualHeadGNN,
    loader: DataLoader,
    device: torch.device,
    c_mean: float,
    c_std: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    cm = torch.tensor(c_mean, device=device)
    cs = torch.tensor(c_std, device=device).clamp_min(1e-3)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            c_atoms = batch["c_atoms"].to(device)
            c_shifts = batch["c_shifts"].to(device)
            c_mask = batch["c_mask"].to(device)
            c_pred_norm, _ = model(x, adj, atom_mask)
            c_pred = c_pred_norm * cs + cm
            safe = c_atoms.clamp(min=0)
            gathered = c_pred.gather(1, safe)
            err = (gathered - c_shifts).abs() * c_mask.float()
            total += err.sum().item()
            count += c_mask.sum().item()
    model.train()
    return total / max(count, 1)


def evaluate_h_mae(
    model: NMRDualHeadGNN,
    loader: DataLoader,
    device: torch.device,
    h_mean: float,
    h_std: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    hm = torch.tensor(h_mean, device=device)
    hs = torch.tensor(h_std, device=device).clamp_min(1e-3)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            hsqc_c_atoms = batch["hsqc_c_atoms"].to(device)
            hsqc_h = batch["hsqc_h"].to(device)
            hsqc_mask = batch["hsqc_mask"].to(device)
            _, h_pred_norm = model(x, adj, atom_mask)
            h_pred = h_pred_norm * hs + hm
            safe = hsqc_c_atoms.clamp(min=0)
            gathered = h_pred.gather(1, safe)
            err = (gathered - hsqc_h).abs() * hsqc_mask.float()
            total += err.sum().item()
            count += hsqc_mask.sum().item()
    model.train()
    return total / max(count, 1)


def train_variant(
    cfg: Config2D,
    dataset: HSQCDataset,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    labeled_indices: list[int],
    unlabeled_indices: list[int],
    log_path: Path,
) -> dict:
    set_seed(cfg.seed)
    device = _select_device()

    in_dim = dataset[0]["x"].shape[1]
    model = NMRDualHeadGNN(
        in_dim=in_dim, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=cfg.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cm = torch.tensor(cfg.c_mean, device=device)
    cs = torch.tensor(cfg.c_std, device=device).clamp_min(1e-3)
    hm = torch.tensor(cfg.h_mean, device=device)
    hs = torch.tensor(cfg.h_std, device=device).clamp_min(1e-3)

    labeled_set = set(labeled_indices)

    def make_loader(indices, shuffle):
        subset = torch.utils.data.Subset(dataset, indices)
        return DataLoader(subset, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=pad_collate)

    if cfg.variant == "supervised_1d":
        train_loader = make_loader(labeled_indices, shuffle=True)
    else:
        train_loader = make_loader(labeled_indices + unlabeled_indices, shuffle=True)

    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    # Labeled-id lookup for batch-level supervision masking
    labeled_id_lookup = {dataset.molecules[i].nmr_id for i in labeled_indices}

    best_val_c = math.inf
    best_state = None
    history = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_metrics = {}
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
                [mid in labeled_id_lookup for mid in batch["ids"]], device=device
            )
            lab = is_labeled
            unlab = ~is_labeled

            sup_loss = c_pred.new_tensor(0.0)
            ssl_loss = c_pred.new_tensor(0.0)

            if lab.any():
                sup_loss = per_atom_c_loss(
                    c_pred[lab], c_atoms[lab], c_shifts[lab], c_mask[lab]
                )

            if cfg.variant == "sort_match_ssl_1d" and unlab.any():
                # 1-D 13-C sort-match at C atoms of the molecule (in RDKit index order)
                sorted_c_atoms, _ = torch.sort(c_atoms[unlab].masked_fill(~c_mask[unlab], c_atoms.size(1) + 1), dim=-1)
                safe_c = sorted_c_atoms.clamp(min=0, max=c_pred.size(1) - 1)
                pred_c_rdorder = c_pred[unlab].gather(1, safe_c)
                ssl_loss = masked_sort_match_loss(
                    pred_c_rdorder, c_shifts[unlab], c_mask[unlab], kind="mse"
                )

            elif cfg.variant == "sort_match_ssl_2d" and unlab.any():
                # 2-D sort-match on HSQC cross-peaks. Gather predicted (H, C)
                # at the carbon atoms that carry an H, in RDKit atom-index
                # order, and compare the resulting 2-D set to the observed
                # HSQC peak multiset via the sliced Wasserstein loss.
                hc_unlab = hsqc_c_atoms[unlab]
                hm_unlab = hsqc_mask[unlab]
                # Sort by atom idx so we can gather predictions canonically
                max_n = x.size(1)
                fill = torch.where(hm_unlab, hc_unlab, hc_unlab.new_full(hc_unlab.shape, max_n + 1))
                sorted_hc, _ = torch.sort(fill, dim=-1)
                safe_hc = sorted_hc.clamp(min=0, max=max_n - 1)
                pred_h_at_c = h_pred[unlab].gather(1, safe_hc)
                pred_c_at_c = c_pred[unlab].gather(1, safe_hc)
                # Predicted 2-D HSQC set: (B, K_h, 2)
                pred_set = torch.stack([pred_h_at_c, pred_c_at_c], dim=-1)
                # Target set in SDF order (H_mean, C) per row
                target_set = torch.stack([hsqc_h[unlab], hsqc_c[unlab]], dim=-1)
                ssl_loss = sliced_sort_match_loss_2d(
                    pred_set, target_set, hm_unlab, K=cfg.K_directions, kind="mse"
                )

            total = sup_loss + cfg.ssl_weight * ssl_loss
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_metrics.setdefault("total", []).append(total.item())
            epoch_metrics.setdefault("sup", []).append(sup_loss.item())
            if isinstance(ssl_loss, Tensor) and ssl_loss.numel() == 1:
                epoch_metrics.setdefault("ssl", []).append(ssl_loss.item())

        val_c = evaluate_c_mae(model, val_loader, device, cfg.c_mean, cfg.c_std)
        val_h = evaluate_h_mae(model, val_loader, device, cfg.h_mean, cfg.h_std)
        history.append(
            {
                "epoch": epoch,
                "val_c_mae": val_c,
                "val_h_mae": val_h,
                **{k: float(np.mean(v)) for k, v in epoch_metrics.items() if v},
            }
        )
        if val_c < best_val_c:
            best_val_c = val_c
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_c = evaluate_c_mae(model, test_loader, device, cfg.c_mean, cfg.c_std)
    test_h = evaluate_h_mae(model, test_loader, device, cfg.h_mean, cfg.h_std)
    elapsed = time.time() - t0

    result = {
        "config": asdict(cfg),
        "best_val_c_mae": best_val_c,
        "test_c_mae": test_c,
        "test_h_mae": test_h,
        "history": history,
        "elapsed_sec": elapsed,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        json.dump(result, f, indent=2)
    return result
