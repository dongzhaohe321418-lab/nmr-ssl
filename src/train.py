"""Training loops for the three experimental variants.

Variant A — supervised
    Train only on atom-assigned data using per-atom MSE on the correct atom.

Variant B — naive "unassigned" pretrain then supervised finetune
    Pretrain on all data using an arbitrary assignment (e.g., first-k atoms
    of the predicted set matched to the target list in given order). This
    is the strawman: it captures the "use unassigned data" motivation but
    does NOT use the sort-match loss.

Variant C — sort-match semi-supervised (our method)
    Joint training: supervised loss on labeled examples + sort-match loss on
    "unassigned" examples (where atom indices have been dropped). This is the
    contribution of the paper.

All three variants use the same GNN architecture and the same test set.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data import NMRDataset, pad_batch
from src.losses import masked_sort_match_loss, sort_match_loss
from src.model import NMRShiftGNN


@dataclass
class TrainConfig:
    variant: str  # "supervised" | "naive_ssl" | "sort_match_ssl"
    hidden: int = 128
    n_layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 30
    ssl_weight: float = 1.0
    labeled_frac: float = 0.1
    seed: int = 0
    device: str = "mps"
    target_mean: float = 0.0
    target_std: float = 1.0


def select_device(requested: str) -> torch.device:
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def per_atom_mse_loss(
    pred: Tensor,
    target_atom: Tensor,
    target_shift: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Standard supervised loss: MSE between predicted shift at target atom indices
    and the ground-truth shift.

    pred         : (B, N) predicted per-atom shift
    target_atom  : (B, K) atom indices
    target_shift : (B, K) ground-truth shifts
    target_mask  : (B, K) valid-target mask
    """
    B, K = target_atom.shape
    safe_idx = target_atom.clamp(min=0)
    gathered = pred.gather(1, safe_idx)  # (B, K)
    sq = (gathered - target_shift) ** 2
    sq = sq * target_mask.float()
    denom = target_mask.sum().clamp(min=1).float()
    return sq.sum() / denom


def per_atom_mae(
    pred: Tensor,
    target_atom: Tensor,
    target_shift: Tensor,
    target_mask: Tensor,
) -> float:
    safe_idx = target_atom.clamp(min=0)
    gathered = pred.gather(1, safe_idx)
    err = (gathered - target_shift).abs() * target_mask.float()
    return (err.sum() / target_mask.sum().clamp(min=1).float()).item()


def extract_predicted_sets(
    pred: Tensor,
    target_atom: Tensor,
    target_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather predictions at the target atom positions, padded with the mean.

    This is used for the sort-match loss: we extract the predicted shifts at
    exactly the positions where the target set lives (by atom index), so that
    predicted and target sets have the same cardinality per molecule.

    The key insight: during SSL training we "don't know" which predicted-atom
    corresponds to which target peak, but we DO know WHICH ATOMS produce peaks
    (e.g., all ``C`` atoms). So we still gather at those atom indices; we just
    do not use the *assignment* between them.

    For a cleaner formulation (no need to know which atoms), see ``pool_atoms``
    below — it assumes all atoms of the target nucleus contribute.
    """
    safe_idx = target_atom.clamp(min=0)
    gathered = pred.gather(1, safe_idx)  # (B, K)
    # Pad invalid positions with the per-row mean so sorting is unaffected.
    row_mean = (gathered * target_mask.float()).sum(dim=1, keepdim=True) / target_mask.sum(
        dim=1, keepdim=True
    ).clamp(min=1).float()
    gathered = torch.where(target_mask, gathered, row_mean)
    return gathered, target_mask


def _gather_c_atoms_rdkit_order(
    pred: Tensor, target_atom: Tensor, target_mask: Tensor
) -> Tensor:
    """Return predictions at C-atom positions in RDKit atom-index order.

    ``target_atom`` lists C atom indices in SDF order (which happens to be
    shift-sorted). Sorting gives us RDKit atom-index order. Padded positions
    in target_atom are -1; we sort them to the *end* by temporarily replacing
    them with a large sentinel value greater than any valid atom index.
    """
    n_max = pred.size(1)
    LARGE_IDX = n_max + 1
    sortable = torch.where(target_mask, target_atom, torch.full_like(target_atom, LARGE_IDX))
    sorted_atoms, _ = torch.sort(sortable, dim=-1)
    safe = sorted_atoms.clamp(min=0, max=n_max - 1)
    return pred.gather(1, safe)


def variant_loss(
    variant: str,
    pred: Tensor,
    target_atom: Tensor,
    target_shift: Tensor,
    target_mask: Tensor,
    labeled_mask: Tensor,
    ssl_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    """Return (loss, metrics dict) for a single training mini-batch.

    labeled_mask is (B,) bool indicating which samples are "labeled" this epoch.
    """
    metrics: dict[str, float] = {}
    B = pred.size(0)
    lab = labeled_mask
    unlab = ~labeled_mask

    sup_loss = pred.new_tensor(0.0)
    ssl_loss = pred.new_tensor(0.0)

    if lab.any():
        sup_loss = per_atom_mse_loss(
            pred[lab], target_atom[lab], target_shift[lab], target_mask[lab]
        )
        metrics["sup_loss"] = sup_loss.item()

    if variant == "supervised":
        total = sup_loss
    elif variant == "naive_ssl":
        # Strawman: pretend we don't know the peak→atom assignment. Gather
        # predictions at C atoms in RDKit atom-index order, and naively match
        # them to target_shift in its original (shift-sorted) order. Wrong
        # assignment in general → the loss penalizes a bogus mapping. Still
        # weakly informative because the set of shifts is preserved.
        if unlab.any():
            unlab_target_atom = target_atom[unlab]
            unlab_target_shift = target_shift[unlab]
            unlab_mask = target_mask[unlab]
            pred_c = _gather_c_atoms_rdkit_order(
                pred[unlab], unlab_target_atom, unlab_mask
            )
            sq = ((pred_c - unlab_target_shift) ** 2) * unlab_mask.float()
            denom = unlab_mask.sum().clamp(min=1).float()
            ssl_loss = sq.sum() / denom
            metrics["ssl_loss"] = ssl_loss.item()
        total = sup_loss + ssl_weight * ssl_loss
    elif variant == "sort_match_ssl":
        # Our method: gather predictions at C atoms (in RDKit atom order, so
        # no leakage of the true assignment), sort both predictions and target
        # shifts, match them element-wise. This is the optimal permutation-
        # invariant matching (Theorem 1 in docs/theorem.md).
        if unlab.any():
            unlab_target_atom = target_atom[unlab]
            unlab_target_shift = target_shift[unlab]
            unlab_mask = target_mask[unlab]
            pred_c = _gather_c_atoms_rdkit_order(
                pred[unlab], unlab_target_atom, unlab_mask
            )
            ssl_loss = masked_sort_match_loss(
                pred_c, unlab_target_shift, unlab_mask, kind="mse"
            )
            metrics["ssl_loss"] = ssl_loss.item()
        total = sup_loss + ssl_weight * ssl_loss
    else:
        raise ValueError(variant)

    metrics["total_loss"] = total.item()
    return total, metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    dataset: NMRDataset,
    indices: list[int] | None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = dataset if indices is None else torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_batch,
        num_workers=0,
    )


def evaluate(
    model: NMRShiftGNN,
    loader: DataLoader,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> float:
    model.eval()
    errs = []
    ns = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            target_atom = batch["target_atom"].to(device)
            target_shift = batch["target_shift"].to(device)
            target_mask = batch["target_mask"].to(device)
            pred_norm = model(x, adj, atom_mask)
            pred = pred_norm * std + mean
            safe_idx = target_atom.clamp(min=0)
            gathered = pred.gather(1, safe_idx)
            err = (gathered - target_shift).abs() * target_mask.float()
            errs.append(err.sum().item())
            ns.append(target_mask.sum().item())
    model.train()
    return sum(errs) / max(sum(ns), 1)


def train_one_variant(
    cfg: TrainConfig,
    train_dataset: NMRDataset,
    val_dataset: NMRDataset,
    test_dataset: NMRDataset,
    labeled_indices: list[int],
    unlabeled_indices: list[int],
    in_dim: int,
    log_path: Path,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    mean = torch.tensor(cfg.target_mean, device=device)
    std = torch.tensor(cfg.target_std, device=device).clamp_min(1e-3)

    model = NMRShiftGNN(
        in_dim=in_dim,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Build labeled-only loader for the supervised variant; the combined loader
    # used for semi-supervised variants uses all train indices and differentiates
    # labeled vs unlabeled at loss time.
    if cfg.variant == "supervised":
        train_loader = make_loader(
            train_dataset, labeled_indices, batch_size=cfg.batch_size, shuffle=True
        )
    else:
        all_indices = labeled_indices + unlabeled_indices
        train_loader = make_loader(
            train_dataset, all_indices, batch_size=cfg.batch_size, shuffle=True
        )

    val_loader = make_loader(val_dataset, None, batch_size=cfg.batch_size, shuffle=False)
    test_loader = make_loader(test_dataset, None, batch_size=cfg.batch_size, shuffle=False)

    labeled_set = set(labeled_indices)

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_state: dict[str, Any] | None = None
    start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        epoch_metrics: dict[str, list[float]] = {}
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            target_atom = batch["target_atom"].to(device)
            target_shift = batch["target_shift"].to(device)
            target_mask = batch["target_mask"].to(device)

            pred_norm = model(x, adj, atom_mask)
            pred = pred_norm * std + mean

            # Reconstruct per-sample "labeled" flags by looking at molecule_ids.
            # We shipped indices in a Subset, but DataLoader strips that. Rely on
            # labeled/unlabeled being disjoint sets of dataset indices and use
            # molecule_ids (unique) to tag each batch element.
            is_labeled = torch.tensor(
                [mid in _LABELED_MID_CACHE for mid in batch["molecule_ids"]],
                device=device,
            )

            loss, metrics = variant_loss(
                cfg.variant,
                pred,
                target_atom,
                target_shift,
                target_mask,
                is_labeled,
                cfg.ssl_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

        val_mae = evaluate(model, val_loader, device, mean, std)
        avg = {k: float(np.mean(v)) for k, v in epoch_metrics.items()}
        history.append({"epoch": epoch, "val_mae": val_mae, **avg})
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"[{cfg.variant}] epoch {epoch:3d}  val MAE = {val_mae:6.3f}  "
            + "  ".join(f"{k}={v:.3f}" for k, v in avg.items())
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader, device, mean, std)
    elapsed = time.time() - start

    result = {
        "config": asdict(cfg),
        "best_val_mae": best_val,
        "test_mae": test_mae,
        "history": history,
        "elapsed_sec": elapsed,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        json.dump(result, f, indent=2)
    return result


# Module-level cache used by the inner training loop to decide which samples
# in a batch count as "labeled". Populated by run_experiments() before
# train_one_variant() is called.
_LABELED_MID_CACHE: set[str] = set()


def set_labeled_cache(ids: Iterable[str]) -> None:
    global _LABELED_MID_CACHE
    _LABELED_MID_CACHE = set(ids)
