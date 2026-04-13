"""Sliced sort-match loss for 2D NMR HSQC peak lists.

Implements Theorem 2 (docs/2d/theorem_2d.md): the 2D bipartite matching cost
between two point sets is estimated as the average of 1-D sort-match losses
along K random directions. Each direction reuses the masked 1-D sort-match
from ``src/losses.py``.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from src.losses import masked_sort_match_loss


def _random_directions(K: int, d: int, device, dtype, generator=None) -> Tensor:
    """K unit vectors in R^d, drawn iid from the unit sphere."""
    v = torch.randn(K, d, generator=generator, device=device, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v


def sliced_sort_match_loss_2d(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    K: int = 8,
    kind: str = "mse",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sliced sort-match loss between two 2-D point sets.

    Parameters
    ----------
    pred : (B, N, 2)
        Predicted 2-D point set for each batch element.
    target : (B, N, 2)
        Observed 2-D point set (unassigned) for each batch element.
    mask : (B, N) bool
        Validity mask; padding positions contribute zero.
    K : int
        Number of random 1-D projection directions.
    kind : "mae" | "mse" | "huber"
        Per-pair convex cost passed through to the masked 1-D sort-match.
    generator : torch.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    Tensor
        Scalar loss averaged across directions and batch.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} target={target.shape}")
    if pred.dim() != 3 or pred.size(-1) != 2:
        raise ValueError(f"expected (B, N, 2) tensor, got {pred.shape}")
    B, N, D = pred.shape
    if mask.shape != (B, N):
        raise ValueError(f"mask shape must be (B, N), got {mask.shape}")

    directions = _random_directions(K, D, pred.device, pred.dtype, generator)

    # (B, N, 2) @ (K, 2).T -> (B, N, K)
    pred_proj = pred @ directions.t()  # (B, N, K)
    target_proj = target @ directions.t()  # (B, N, K)

    # Reshape to run K sort-match losses in parallel.
    # Move K to batch: (B, N, K) -> (B*K, N) with mask repeated.
    pred_flat = pred_proj.permute(0, 2, 1).reshape(B * K, N)
    target_flat = target_proj.permute(0, 2, 1).reshape(B * K, N)
    mask_flat = mask.unsqueeze(1).expand(B, K, N).reshape(B * K, N)

    loss = masked_sort_match_loss(pred_flat, target_flat, mask_flat, kind=kind)
    return loss


def axis_aligned_sort_match_loss_2d(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    kind: str = "mse",
) -> Tensor:
    """Cheapest possible 2-D sort-match loss: K=2 axis-aligned projections.

    This is biased (it can be zero when the true W_2^2 is positive) but is
    useful as a warm-start regularizer and for ablation against the random-
    direction sliced loss.
    """
    if pred.shape[-1] != 2:
        raise ValueError(f"expected (B, N, 2), got {pred.shape}")
    loss_x = masked_sort_match_loss(pred[..., 0], target[..., 0], mask, kind=kind)
    loss_y = masked_sort_match_loss(pred[..., 1], target[..., 1], mask, kind=kind)
    return loss_x + loss_y


def hungarian_reference_2d(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    kind: str = "mse",
) -> Tensor:
    """Exact 2-D optimal matching via scipy Hungarian, used only for tests.

    Computes a per-example scalar loss by building the full N x N cost matrix
    and running linear_sum_assignment. CPU-only, slow, no gradient flow.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if pred.shape[-1] != 2:
        raise ValueError(f"expected (B, N, 2), got {pred.shape}")

    B = pred.shape[0]
    losses = []
    for b in range(B):
        valid = mask[b].bool().cpu().numpy()
        if not valid.any():
            losses.append(0.0)
            continue
        p = pred[b][valid].detach().cpu().numpy()
        t = target[b][valid].detach().cpu().numpy()
        n = p.shape[0]
        cost = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                dx = p[i, 0] - t[j, 0]
                dy = p[i, 1] - t[j, 1]
                if kind == "mse":
                    cost[i, j] = dx * dx + dy * dy
                elif kind == "mae":
                    cost[i, j] = abs(dx) + abs(dy)
                else:
                    raise ValueError(kind)
        row, col = linear_sum_assignment(cost)
        losses.append(float(cost[row, col].mean()))
    return torch.tensor(losses, dtype=pred.dtype).mean()
