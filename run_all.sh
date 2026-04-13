#!/usr/bin/env bash
# Reproduce every result in the paper end-to-end from scratch.
#
# Expected runtime on an Apple M4 Pro MacBook with MPS: ~90 minutes.
# Expected peak memory: < 3 GB.
#
# Stages:
#   1. Download NMRShiftDB2 (~150 MB)
#   2. Verify Theorem 1 numerically (< 1 s)
#   3. Run the main single-seed experiment (~10 min)
#   4. Run the labeled-fraction ablation (~25 min)
#   5. Run the 3-seed main + scaffold-split suite (~45 min)
#   6. Generate all figures

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "=== [1/6] Downloading NMRShiftDB2 ==="
mkdir -p data
if [[ ! -f data/nmrshiftdb2withsignals.sd ]]; then
  curl -sL -o data/nmrshiftdb2withsignals.sd \
    "https://sourceforge.net/projects/nmrshiftdb2/files/data/nmrshiftdb2withsignals.sd/download"
  echo "  downloaded $(du -h data/nmrshiftdb2withsignals.sd | cut -f1)"
else
  echo "  already present: $(du -h data/nmrshiftdb2withsignals.sd | cut -f1)"
fi

echo ""
echo "=== [2/6] Verifying Theorem 1 ==="
python3 tests/test_theorem.py

echo ""
echo "=== [3/6] Main single-seed experiment ==="
python3 experiments/run_ssl_experiment.py \
  --sdf data/nmrshiftdb2withsignals.sd \
  --max-records 10000 \
  --epochs 30 \
  --hidden 128 \
  --n-layers 4 \
  --labeled-frac 0.1 \
  --out experiments/results_main

echo ""
echo "=== [4/6] Labeled-fraction ablation ==="
python3 experiments/run_ablation.py \
  --sdf data/nmrshiftdb2withsignals.sd \
  --max-records 6000 \
  --epochs 20 \
  --fractions 0.02 0.05 0.1 0.2 0.5 \
  --out experiments/results_ablation

echo ""
echo "=== [5/6] 3-seed suite (main + scaffold) ==="
python3 experiments/run_full_suite.py \
  --sdf data/nmrshiftdb2withsignals.sd \
  --max-records 10000 \
  --epochs 25 \
  --seeds 0 1 2 \
  --out experiments/results_suite \
  --skip h1

echo ""
echo "=== [6/6] Generating figures ==="
python3 experiments/make_nature_figures.py \
  --main experiments/results_main \
  --ablation experiments/results_ablation \
  --suite experiments/results_suite \
  --out figures

echo ""
echo "=== DONE ==="
echo "  figures in:    figures/"
echo "  raw results:   experiments/results_*/"
echo "  preprint:      docs/preprint_v1.md"
