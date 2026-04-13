"""Simple GIN-style graph neural network for NMR chemical shift prediction.

Deliberately minimal: dense adjacency, fixed-dim node features, no
torch_geometric. Good enough for an MVP on NMRShiftDB2 and runs on MPS.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GINLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, adj: Tensor, mask: Tensor) -> Tensor:
        """h: (B, N, D), adj: (B, N, N), mask: (B, N) bool.

        The adjacency already contains self-loops (added in data.py). For the
        standard GIN update h'_v = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)
        we compute sum over neighbours *excluding* self (so we separate eps).
        """
        self_loop = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj_no_self = (adj - self_loop).clamp(min=0.0)
        agg = torch.bmm(adj_no_self, h)  # (B, N, D)
        updated = self.mlp((1 + self.eps) * h + agg)
        h = self.norm(h + self.dropout(updated))
        return h * mask.unsqueeze(-1).float()


class NMRShiftGNN(nn.Module):
    """Node-level regression: per-atom chemical shift prediction.

    Args
    ----
    in_dim   : atom feature dim
    hidden   : hidden dim (shared across all layers)
    n_layers : number of GIN layers
    n_solvents : optional number of solvent classes (0 to disable)
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        n_layers: int = 4,
        n_solvents: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_solvents = n_solvents
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.layers = nn.ModuleList([GINLayer(hidden, dropout) for _ in range(n_layers)])
        if n_solvents > 0:
            self.solvent_embed = nn.Embedding(n_solvents, hidden)
        readout_in = hidden
        self.readout = nn.Sequential(
            nn.Linear(readout_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        atom_mask: Tensor,
        solvent_idx: Tensor | None = None,
    ) -> Tensor:
        """Return predicted per-atom chemical shifts (B, N)."""
        h = self.embed(x)
        if self.n_solvents > 0 and solvent_idx is not None:
            solvent_vec = self.solvent_embed(solvent_idx)  # (B, D)
            h = h + solvent_vec.unsqueeze(1)
        for layer in self.layers:
            h = layer(h, adj, atom_mask)
        shifts = self.readout(h).squeeze(-1)  # (B, N)
        return shifts
