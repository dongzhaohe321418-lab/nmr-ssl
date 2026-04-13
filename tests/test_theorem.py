"""Numerical verification of Theorem 1 (Sort-Match Optimality).

If any of these tests fail, the theorem in docs/theorem.md is wrong and
the entire paper premise collapses. This is the hard stopping criterion
for the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import hungarian_reference, sort_match_loss  # noqa: E402


def _run(kind: str, n_trials: int = 200, n_max: int = 24, seed: int = 0) -> None:
    torch.manual_seed(seed)
    max_rel_diff = 0.0
    for t in range(n_trials):
        n = int(torch.randint(2, n_max + 1, (1,)).item())
        y_hat = (torch.randn(n) * 50).double()
        y_star = (torch.randn(n) * 50).double()
        sort_val = sort_match_loss(y_hat, y_star, kind=kind, reduction="none").item()
        hung_val = hungarian_reference(y_hat.unsqueeze(0), y_star.unsqueeze(0), kind=kind).item()
        denom = max(abs(sort_val), abs(hung_val), 1e-12)
        rel = abs(sort_val - hung_val) / denom
        if rel > max_rel_diff:
            max_rel_diff = rel
        assert rel < 1e-10, (
            f"kind={kind} trial={t} n={n}: sort={sort_val:.10f} hungarian={hung_val:.10f} rel_err={rel:.2e}"
        )
    print(f"  [{kind}] {n_trials} trials passed, max rel err = {max_rel_diff:.2e}")


def test_mae():
    _run("mae")


def test_mse():
    _run("mse")


def test_huber():
    _run("huber")


def test_batched_agreement():
    torch.manual_seed(1)
    B, n = 16, 10
    y_hat = (torch.randn(B, n) * 50).double()
    y_star = (torch.randn(B, n) * 50).double()
    batched = sort_match_loss(y_hat, y_star, kind="mse", reduction="none")
    hungarian = hungarian_reference(y_hat, y_star, kind="mse")
    rel = ((batched - hungarian).abs() / batched.abs().clamp_min(1e-12)).max().item()
    assert rel < 1e-10, f"batched sort_match disagrees with hungarian: rel_err={rel:.2e}"
    print(f"  [batched] B={B} n={n} max rel err = {rel:.2e}")


def test_gradient_flow():
    torch.manual_seed(2)
    n = 8
    y_hat = (torch.randn(n) * 50).requires_grad_(True)
    y_star = torch.randn(n) * 50
    loss = sort_match_loss(y_hat, y_star, kind="mse")
    loss.backward()
    assert y_hat.grad is not None
    assert torch.isfinite(y_hat.grad).all()
    print(f"  [grad] grad norm = {y_hat.grad.norm().item():.4f}")


def test_known_easy_case():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert sort_match_loss(y, y, kind="mae").item() == 0.0
    assert sort_match_loss(y, y, kind="mse").item() == 0.0

    y_rev = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert sort_match_loss(y, y_rev, kind="mae").item() == 0.0

    y_shift = y + 2.5
    assert abs(sort_match_loss(y, y_shift, kind="mae").item() - 2.5) < 1e-6
    print("  [known] trivial cases pass")


def test_minimality_over_random_permutations():
    torch.manual_seed(3)
    for _ in range(50):
        n = 12
        y_hat = torch.randn(n) * 50
        y_star = torch.randn(n) * 50
        sort_val = sort_match_loss(y_hat, y_star, kind="mse", reduction="none").item()
        for _ in range(5):
            perm = torch.randperm(n)
            arbitrary = ((y_hat - y_star[perm]) ** 2).mean().item()
            assert arbitrary >= sort_val - 1e-8, (
                f"arbitrary permutation gave {arbitrary:.6f} < sort-match {sort_val:.6f}"
            )
    print("  [minimality] sort-match is a lower bound over random permutations")


if __name__ == "__main__":
    print("Numerical verification of Theorem 1 (Sort-Match Optimality)")
    print("=" * 60)
    test_mae()
    test_mse()
    test_huber()
    test_batched_agreement()
    test_gradient_flow()
    test_known_easy_case()
    test_minimality_over_random_permutations()
    print("=" * 60)
    print("ALL TESTS PASSED. Theorem 1 verified numerically.")
