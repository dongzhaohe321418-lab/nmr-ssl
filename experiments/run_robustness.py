"""Robustness ablation: corrupt the unlabeled peak lists to simulate
literature-extraction noise, and verify that sort-match SSL still beats
supervised training.

We model three realistic failure modes of literature-extracted spectra:
  1. Gaussian shift noise: OCR/digitization jitter on the peak values.
  2. Peak dropping: extraction pipeline missed some peaks (F-score < 1).
  3. Spurious peaks: extraction pipeline inserted peaks that do not belong.

Only the UNLABELED split is corrupted; labeled data is clean (to match the
realistic scenario: your curated training set is clean, but the literature
corpus you SSL on is noisy).

Runs three sort-match SSL variants against the supervised baseline:
  - clean unlabeled (reference)
  - gaussian noise sigma = 1.0 ppm
  - 15% peak drop
  - combined (1 ppm noise + 10% drop + 10% spurious)

Expected outcome: sort-match SSL still beats supervised under moderate noise,
with a smaller but still positive improvement.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import NMRDataset, iter_nmrshiftdb2_sdf, pad_batch, mol_to_graph_tensors  # noqa: E402
from src.losses import masked_sort_match_loss  # noqa: E402
from src.model import NMRShiftGNN  # noqa: E402
from src.train import (  # noqa: E402
    _gather_c_atoms_rdkit_order,
    per_atom_mse_loss,
    select_device,
    set_seed,
    evaluate,
)
from experiments.run_ssl_experiment import filter_valid, split_indices  # noqa: E402


def corrupt_targets(
    target_shift: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    noise_sigma: float = 0.0,
    drop_frac: float = 0.0,
    spurious_frac: float = 0.0,
    gen: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply noise / drop / spurious-peak corruption to an unlabeled batch.

    Returns (corrupted_shift, corrupted_mask). Dropped positions have mask
    turned off; spurious positions are placed at random from a plausible range.
    """
    corrupted = target_shift.clone()
    mask = target_mask.clone()

    if noise_sigma > 0:
        noise = torch.randn(corrupted.shape, generator=gen) * noise_sigma
        corrupted = corrupted + noise * mask.float()

    if drop_frac > 0:
        drop = torch.rand(mask.shape, generator=gen) < drop_frac
        mask = mask & ~drop

    if spurious_frac > 0:
        # For each row, with probability spurious_frac * n_valid, replace a
        # random masked-out slot with a spurious peak. Practical but simple:
        # we add roughly spurious_frac * n_valid spurious peaks per row.
        B, K = mask.shape
        n_valid = mask.sum(dim=1)
        for i in range(B):
            n_add = int((n_valid[i].item() * spurious_frac) + 0.5)
            if n_add == 0:
                continue
            # Find invalid slots
            invalid_slots = (~mask[i]).nonzero(as_tuple=True)[0]
            if len(invalid_slots) == 0:
                continue
            choose = invalid_slots[torch.randperm(len(invalid_slots))[:n_add]]
            # Spurious peaks drawn from a uniform over shift range
            corrupted[i, choose] = torch.rand(len(choose), generator=gen) * 200.0
            mask[i, choose] = True

    return corrupted, mask


def train_sort_match_with_corruption(
    *,
    noise_sigma: float,
    drop_frac: float,
    spurious_frac: float,
    labeled_indices,
    unlabeled_indices,
    full_dataset,
    val_dataset,
    test_dataset,
    in_dim: int,
    target_mean: float,
    target_std: float,
    epochs: int = 25,
    hidden: int = 128,
    n_layers: int = 4,
    seed: int = 0,
    device_name: str = "mps",
) -> dict:
    set_seed(seed)
    device = select_device(device_name)
    mean = torch.tensor(target_mean, device=device)
    std = torch.tensor(target_std, device=device).clamp_min(1e-3)

    model = NMRShiftGNN(in_dim=in_dim, hidden=hidden, n_layers=n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    all_indices = labeled_indices + unlabeled_indices
    train_subset = torch.utils.data.Subset(full_dataset, all_indices)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_batch)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_batch)

    labeled_id_set = {full_dataset.molecules[i].nmrshift_id for i in labeled_indices}

    best_val = float("inf")
    best_state = None
    gen = torch.Generator().manual_seed(seed * 7 + 13)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            target_atom = batch["target_atom"].to(device)
            target_shift = batch["target_shift"].to(device)
            target_mask = batch["target_mask"].to(device)

            pred_norm = model(x, adj, atom_mask)
            pred = pred_norm * std + mean

            is_labeled = torch.tensor(
                [mid in labeled_id_set for mid in batch["molecule_ids"]], device=device
            )
            lab_mask = is_labeled
            unlab_mask = ~is_labeled

            sup_loss = pred.new_tensor(0.0)
            ssl_loss = pred.new_tensor(0.0)

            if lab_mask.any():
                sup_loss = per_atom_mse_loss(
                    pred[lab_mask], target_atom[lab_mask], target_shift[lab_mask], target_mask[lab_mask]
                )

            if unlab_mask.any():
                unlab_ta = target_atom[unlab_mask]
                unlab_ts = target_shift[unlab_mask]
                unlab_tm = target_mask[unlab_mask]
                corrupted_ts, corrupted_tm = corrupt_targets(
                    unlab_ts.cpu(),
                    unlab_tm.cpu(),
                    noise_sigma=noise_sigma,
                    drop_frac=drop_frac,
                    spurious_frac=spurious_frac,
                    gen=gen,
                )
                corrupted_ts = corrupted_ts.to(device)
                corrupted_tm = corrupted_tm.to(device)
                pred_c = _gather_c_atoms_rdkit_order(pred[unlab_mask], unlab_ta, unlab_tm)
                ssl_loss = masked_sort_match_loss(
                    pred_c, corrupted_ts, corrupted_tm, kind="mse"
                )

            loss = sup_loss + 0.5 * ssl_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_mae = evaluate(model, val_loader, device, mean, std)
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae = evaluate(model, test_loader, device, mean, std)
    return {"best_val_mae": best_val, "test_mae": test_mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, default=ROOT / "data" / "nmrshiftdb2withsignals.sd")
    parser.add_argument("--max-records", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--labeled-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results_robustness")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[robustness] parsing {args.sdf.name}")
    raw = list(iter_nmrshiftdb2_sdf(args.sdf, nucleus="13C", max_records=args.max_records))
    kept = filter_valid(raw)
    print(f"  kept {len(kept)} clean molecules")
    x0, _ = mol_to_graph_tensors(kept[0].mol)
    in_dim = x0.shape[1]

    train_idx, val_idx, test_idx = split_indices(len(kept), 0.8, 0.1, args.seed)
    all_train_shifts = []
    for i in train_idx:
        t = kept[i].target_for_nucleus()
        if t is not None:
            all_train_shifts.extend(t[0])
    target_mean = float(np.mean(all_train_shifts))
    target_std = float(np.std(all_train_shifts))

    full_dataset = NMRDataset(kept)
    val_dataset = NMRDataset([kept[i] for i in val_idx])
    test_dataset = NMRDataset([kept[i] for i in test_idx])

    rng = random.Random(args.seed + 1)
    train_shuffled = list(train_idx)
    rng.shuffle(train_shuffled)
    n_lab = max(1, int(len(train_shuffled) * args.labeled_frac))
    labeled_indices = train_shuffled[:n_lab]
    unlabeled_indices = train_shuffled[n_lab:]
    print(f"  labeled={n_lab} unlabeled={len(unlabeled_indices)}")

    corruptions = [
        ("clean", 0.0, 0.0, 0.0),
        ("noise_sigma_1.0", 1.0, 0.0, 0.0),
        ("drop_15", 0.0, 0.15, 0.0),
        ("combined_1ppm_10drop_10spurious", 1.0, 0.10, 0.10),
    ]

    results = {}
    for name, sigma, drop, spurious in corruptions:
        print(f"\n[robustness] {name} (sigma={sigma}, drop={drop}, spurious={spurious})")
        t0 = time.time()
        result = train_sort_match_with_corruption(
            noise_sigma=sigma,
            drop_frac=drop,
            spurious_frac=spurious,
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            full_dataset=full_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            in_dim=in_dim,
            target_mean=target_mean,
            target_std=target_std,
            epochs=args.epochs,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        result["elapsed_sec"] = elapsed
        results[name] = result
        print(f"  test MAE = {result['test_mae']:.3f} ppm  ({elapsed:.0f}s)")

    summary = {
        "args": {
            "sdf": str(args.sdf),
            "max_records": args.max_records,
            "epochs": args.epochs,
            "labeled_frac": args.labeled_frac,
            "seed": args.seed,
        },
        "n_labeled": n_lab,
        "n_unlabeled": len(unlabeled_indices),
        "results": results,
    }
    with (args.out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[robustness] SUMMARY:")
    for name, res in results.items():
        print(f"  {name:40s} test MAE = {res['test_mae']:.3f} ppm")


if __name__ == "__main__":
    main()
