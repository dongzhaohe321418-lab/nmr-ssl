"""Final assembly: regenerate figures, fill preprint placeholders, print
the final summary table.

Run after experiments/run_full_suite.py completes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDOUT:")
        print(proc.stdout)
        print("STDERR:")
        print(proc.stderr)
        sys.exit(proc.returncode)
    print(proc.stdout[-2000:])


def main():
    print("========================================")
    print("FINALIZE: figures + preprint + summary")
    print("========================================")

    run([sys.executable, "experiments/make_nature_figures.py"])
    run([sys.executable, "experiments/fill_preprint.py"])

    summary_path = ROOT / "experiments" / "results_suite" / "suite_summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            suite = json.load(f)
        print("\n========================================")
        print("FINAL SUITE RESULTS")
        print("========================================")
        for name in ("A_main", "B_scaffold"):
            if name not in suite:
                continue
            s = suite[name]
            print(f"\n{name}  ({s['split_mode']} split, n={len(s['seeds'])} seeds):")
            print(f"  {'variant':25s}  {'mean ± std':>16s}")
            print(f"  {'-' * 25}  {'-' * 16}")
            for variant, agg in s["aggregate"].items():
                print(
                    f"  {variant:25s}  {agg['mean']:6.3f} ± {agg['std']:.3f}"
                )

    print("\n========================================")
    print("ARTIFACTS")
    print("========================================")
    print("  figures/fig1_theorem.pdf         — theorem + verification + complexity")
    print("  figures/fig2_main.pdf            — main empirical result w/ error bars")
    print("  figures/fig3_generalization.pdf  — ablation + scaffold + nucleus panels")
    print("  figures/fig_ed1_sinkhorn.pdf     — Sinkhorn OT comparison (Extended Data)")
    print("  docs/preprint_v1_filled.md       — preprint with real numbers filled in")
    print("  experiments/results_*/           — raw JSON logs")
    print()


if __name__ == "__main__":
    main()
