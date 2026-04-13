# Sort-match SSL for NMR chemical shift prediction

Training NMR chemical shift predictors from **unassigned** peak lists — no atom-level labels required.

> **Key claim**: Optimal bipartite matching between a predicted set of shifts and an observed set of peaks reduces exactly (not approximately) to a simple `torch.sort` alignment under any convex per-atom cost. This makes set-based training at literature scale practical.

## One-line version of what this does

```python
from src.losses import sort_match_loss

# predicted shifts for n atoms, observed unassigned peaks
loss = sort_match_loss(predicted, observed_peaks, kind="mse")
loss.backward()  # works, batched, GPU-friendly, and provably optimal
```

## v2 extension — unassigned 2-D HSQC supervision

A follow-up project (`src/nmr2d/`, `experiments/run_option_b_master.py`, `docs/2d/`) extends the 1-D sort-match reduction to 2-D HSQC peak sets via a sliced-Wasserstein loss. The v4 preprint and full revision pipeline including causal H-zero audit, combined-supervision experiment (C 3.23 / H 0.30 ppm), Bonferroni molecule-level conformal, scaffold-OOD split, and a full peer-review-driven iteration log live in [`docs/2d/`](docs/2d/).

```bash
# Re-run the entire Option-B revision pipeline (~26 min on M4 Pro MPS)
python3 experiments/run_option_b_master.py --seeds 0 1 2

# Causal audit that falsifies the "encoder leakage" counter-hypothesis (~3 min)
python3 experiments/run_h_zero_ablation.py --seeds 0 1 2

# Bonferroni-corrected molecule-level conformal calibration (~1 min)
python3 experiments/compute_bonferroni_conformal.py

# Compile the v4 preprint to PDF (requires pandoc + xelatex)
python3 experiments/compile_preprint_2d.py
```

The v4 preprint is at [`docs/2d/preprint_2d.pdf`](docs/2d/preprint_2d.pdf).

## What's here

```
docs/
  lit-review.md         — verified-citation literature review
  theorem.md            — formal statement + proof of the sort-match reduction
  preprint_draft_v0.md  — proof-of-concept preprint (v0)
src/
  losses.py             — sort-match loss + masked variant + Hungarian reference
  data.py               — NMRShiftDB2 SDF parser, atom features, scaffold split
  model.py              — minimal GIN-style GNN (no torch_geometric dependency)
  train.py              — training loop with three variants
tests/
  test_theorem.py       — numerical verification at float64 machine precision
experiments/
  run_ssl_experiment.py — main single-seed experiment
  run_ablation.py       — labeled-fraction ablation
  run_full_suite.py     — multi-seed main + scaffold + 1H
  make_figures.py       — single-column figures
  make_nature_figures.py — Nature-style multi-panel figures
```

## Reproducing the main result

```bash
# 1) download NMRShiftDB2 (~150 MB)
mkdir -p data
curl -sL -o data/nmrshiftdb2withsignals.sd \
  https://sourceforge.net/projects/nmrshiftdb2/files/data/nmrshiftdb2withsignals.sd/download

# 2) verify the theorem (runs in under a second)
python3 tests/test_theorem.py

# 3) run the main experiment (10 min on M4 Pro with MPS)
python3 experiments/run_ssl_experiment.py \
  --sdf data/nmrshiftdb2withsignals.sd \
  --max-records 10000 \
  --epochs 30

# 4) generate figures
python3 experiments/make_figures.py
```

## Expected output (main experiment, single seed, labeled_frac=0.1)

```
  === supervised ===     test MAE = 4.088 ppm  (26 s)
  === naive_ssl  ===     test MAE = 10.700 ppm (249 s)
  === sort_match_ssl === test MAE = 3.574 ppm  (267 s)
```

Sort-match SSL beats the supervised baseline by 12.6% relative (0.51 ppm absolute) using the **same** labeled data, **same** architecture, **same** optimizer. The only difference is two `torch.sort` calls on the unlabeled batch.

## What this is not

- Not a new state of the art on NMRShiftDB2 — our 4-layer GIN is intentionally tiny.
- Not a Nature CS submission in its current form — see `docs/preprint_draft_v0.md` for honest scope.
- Not a replacement for NMRNet (which it complements: combine sort-match with their SE(3) Transformer).

## Dependencies

- Python ≥ 3.9
- PyTorch ≥ 2.0 (MPS or CUDA optional)
- RDKit ≥ 2024.3
- NumPy, SciPy
- Matplotlib (for figures)

```bash
python3 -m pip install --user rdkit matplotlib
# PyTorch install instructions: https://pytorch.org/
```

## Citation

If this work is useful, cite the preprint once released. This repo is released under CC-BY 4.0.

## Acknowledgments

Built iteratively with AI assistance (Claude Opus 4.6) under human oversight. All empirical results are from real experiments on publicly available NMRShiftDB2 data. Every citation in the lit review was fetched and verified via WebFetch; no vibe-citing. Theorem proved independently and verified numerically at float64 machine precision against `scipy.optimize.linear_sum_assignment`.
