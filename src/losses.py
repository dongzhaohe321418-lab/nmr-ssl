"""Sort-match loss for permutation-invariant set supervision of NMR shifts.

See ``docs/theorem.md`` for the derivation. The short version:

    Given predicted shifts y_hat in R^n and an unassigned target set y_star in R^n,
    the optimal bipartite matching under any convex per-pair cost phi(a - b)
    equals the sum of phi over the sorted alignment.

    Hungarian cost: O(n^3), non-batched, non-differentiable.
    Sort cost:      O(n log n), batched, differentiable (via torch.sort).

This file exposes ``sort_match_loss`` for MAE / MSE / Huber and a
``hungarian_reference`` function that we use only in tests.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_sort_match_loss(
    y_hat: Tensor,
    y_star: Tensor,
    mask: Tensor,
    *,
    kind: str = "mse",
    huber_delta: float = 1.0,
    large_value: float = 1e6,
) -> Tensor:
    """Sort-match loss with a per-element validity mask.

    All three tensors have shape (B, K). ``mask[i, j] == True`` means
    position j of row i is a real peak; padding positions are masked out
    and contribute exactly zero to the final loss.

    Trick: we replace padding positions with a very large constant in *both*
    predicted and target rows. After sorting, padding positions end up at the
    tail of each row and align with each other (LARGE vs LARGE → diff = 0).
    The valid positions sort among themselves within their own row.

    The row-level reduction divides by the per-row count of real peaks so that
    rows of different valid length are weighted per atom, not per padded slot.
    """
    if y_hat.shape != y_star.shape or y_hat.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: y_hat={y_hat.shape} y_star={y_star.shape} mask={mask.shape}"
        )

    fill = torch.full_like(y_hat, large_value)
    y_hat_p = torch.where(mask, y_hat, fill)
    y_star_p = torch.where(mask, y_star, fill)

    y_hat_sorted, _ = torch.sort(y_hat_p, dim=-1)
    y_star_sorted, _ = torch.sort(y_star_p, dim=-1)
    diff = y_hat_sorted - y_star_sorted

    if kind == "mae":
        per_pair = diff.abs()
    elif kind == "mse":
        per_pair = diff.pow(2)
    elif kind == "huber":
        absd = diff.abs()
        quadratic = 0.5 * diff.pow(2)
        linear = huber_delta * (absd - 0.5 * huber_delta)
        per_pair = torch.where(absd <= huber_delta, quadratic, linear)
    else:
        raise ValueError(f"unknown kind: {kind}")

    # The pad-vs-pad pairs contribute exactly zero (LARGE - LARGE = 0), so we
    # can safely sum over the whole row and then divide by the real count.
    row_sum = per_pair.sum(dim=-1)
    row_count = mask.sum(dim=-1).clamp(min=1).float()
    per_example = row_sum / row_count
    return per_example.mean()


def sort_match_loss(
    y_hat: Tensor,
    y_star: Tensor,
    *,
    reduction: str = "mean",
    kind: str = "mae",
    huber_delta: float = 1.0,
) -> Tensor:
    """Permutation-invariant sort-match loss.

    Parameters
    ----------
    y_hat : (B, n) or (n,)
        Predicted chemical shifts.
    y_star : (B, n) or (n,)
        Unassigned ground-truth peaks. Must have the same shape as ``y_hat``.
    reduction : "mean" | "sum" | "none"
        "none" returns per-example loss of shape (B,).
    kind : "mae" | "mse" | "huber"
        Convex per-pair loss.
    huber_delta : float
        Delta parameter for Huber loss; ignored otherwise.

    Returns
    -------
    Tensor
        Scalar if reduction is "mean" or "sum", else (B,).
    """
    if y_hat.shape != y_star.shape:
        raise ValueError(f"shape mismatch: {y_hat.shape} vs {y_star.shape}")

    squeeze_back = False
    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
        y_star = y_star.unsqueeze(0)
        squeeze_back = True

    y_hat_sorted, _ = torch.sort(y_hat, dim=-1)
    y_star_sorted, _ = torch.sort(y_star, dim=-1)
    diff = y_hat_sorted - y_star_sorted

    if kind == "mae":
        per_atom = diff.abs()
    elif kind == "mse":
        per_atom = diff.pow(2)
    elif kind == "huber":
        absd = diff.abs()
        quadratic = 0.5 * diff.pow(2)
        linear = huber_delta * (absd - 0.5 * huber_delta)
        per_atom = torch.where(absd <= huber_delta, quadratic, linear)
    else:
        raise ValueError(f"unknown kind: {kind}")

    per_example = per_atom.mean(dim=-1)

    if squeeze_back:
        per_example = per_example.squeeze(0)

    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    if reduction == "none":
        return per_example
    raise ValueError(f"unknown reduction: {reduction}")


def hungarian_reference(
    y_hat: Tensor,
    y_star: Tensor,
    *,
    kind: str = "mae",
    huber_delta: float = 1.0,
) -> Tensor:
    """Reference implementation using scipy's Hungarian algorithm.

    Used only in tests to verify Theorem 1 numerically. Not for training.
    Runs on CPU, one example at a time — deliberately slow and simple.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if y_hat.dim() == 1:
        y_hat = y_hat.unsqueeze(0)
        y_star = y_star.unsqueeze(0)

    results = []
    for i in range(y_hat.shape[0]):
        a = y_hat[i].detach().cpu().numpy()
        b = y_star[i].detach().cpu().numpy()
        n = a.shape[0]
        cost = np.empty((n, n), dtype=np.float64)
        for p in range(n):
            for q in range(n):
                d = a[p] - b[q]
                if kind == "mae":
                    cost[p, q] = abs(d)
                elif kind == "mse":
                    cost[p, q] = d * d
                elif kind == "huber":
                    ad = abs(d)
                    cost[p, q] = 0.5 * d * d if ad <= huber_delta else huber_delta * (ad - 0.5 * huber_delta)
                else:
                    raise ValueError(kind)
        row, col = linear_sum_assignment(cost)
        per_example = cost[row, col].mean()
        results.append(per_example)

    return torch.tensor(results, dtype=y_hat.dtype)
