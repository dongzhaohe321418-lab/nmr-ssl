"""Dual-head GNN predicting both 1-H and 13-C chemical shifts.

Shared GIN encoder (reused from src/model.py style), followed by two linear
readout heads — one for 13-C shift (per atom) and one for the mean 1-H shift
at each heavy atom. Only C atoms contribute to the 13-C supervision and only
heavy atoms that actually carry H contribute to the 1-H mean supervision.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model import GINLayer


class NMRDualHeadGNN(nn.Module):
    """Predicts per-atom 13-C shift and per-heavy-atom mean 1-H shift."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 192,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.layers = nn.ModuleList([GINLayer(hidden, dropout) for _ in range(n_layers)])
        self.readout_c = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.readout_h = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: Tensor,  # (B, N, F)
        adj: Tensor,  # (B, N, N)
        atom_mask: Tensor,  # (B, N)
    ) -> tuple[Tensor, Tensor]:
        """Return (c_pred, h_pred), each of shape (B, N).

        c_pred is per-atom predicted 13-C shift (only meaningful at C atoms)
        h_pred is per-atom predicted mean 1-H shift (only meaningful at heavy
                atoms that carry at least one H)
        """
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, adj, atom_mask)
        c_pred = self.readout_c(h).squeeze(-1)
        h_pred = self.readout_h(h).squeeze(-1)
        return c_pred, h_pred
