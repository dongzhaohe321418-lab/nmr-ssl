"""Analyze the composition of the retained dataset to address the domain
reviewer's concern that the non-degenerate filter biases the dataset away
from drug chemistry.

Reports:
  - Total parsed vs kept
  - Scaffold frequency distribution
  - Lipinski compliance of retained molecules
  - Element composition
  - Molecular size distribution
  - Top 20 scaffolds
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import iter_nmrshiftdb2_sdf  # noqa: E402
from experiments.run_ssl_experiment import filter_valid  # noqa: E402


def lipinski_compliant(mol: Chem.Mol) -> bool:
    """Ro5: MW <= 500, logP <= 5, HBD <= 5, HBA <= 10."""
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
    except Exception:
        return False


def scaffold_key(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""


def main():
    print("[analyze] parsing NMRShiftDB2 ...")
    raw = list(iter_nmrshiftdb2_sdf(
        ROOT / "data" / "nmrshiftdb2withsignals.sd",
        nucleus="13C",
        max_records=20000,
    ))
    print(f"  parsed {len(raw)} 13C records")
    kept = filter_valid(raw)
    print(f"  kept {len(kept)} after non-degenerate filter (retention rate {len(kept)/max(len(raw),1)*100:.1f}%)")

    # Scaffold analysis
    scaffolds = Counter()
    mw_list = []
    n_atoms_list = []
    n_heavy_list = []
    lipinski_count = 0
    element_counter = Counter()
    for m in kept:
        scaffolds[scaffold_key(m.mol)] += 1
        mw_list.append(Descriptors.MolWt(m.mol))
        n_atoms_list.append(m.mol.GetNumAtoms())
        n_heavy_list.append(m.mol.GetNumHeavyAtoms())
        if lipinski_compliant(m.mol):
            lipinski_count += 1
        for atom in m.mol.GetAtoms():
            element_counter[atom.GetSymbol()] += 1

    print("\n## Molecular size distribution")
    print(f"  MW (g/mol) : mean={np.mean(mw_list):.1f}, median={np.median(mw_list):.1f}, "
          f"p25={np.percentile(mw_list, 25):.1f}, p75={np.percentile(mw_list, 75):.1f}, max={max(mw_list):.1f}")
    print(f"  Heavy atoms: mean={np.mean(n_heavy_list):.1f}, median={np.median(n_heavy_list):.1f}, "
          f"max={max(n_heavy_list)}")
    print(f"  Total atoms: mean={np.mean(n_atoms_list):.1f}, median={np.median(n_atoms_list):.1f}, "
          f"max={max(n_atoms_list)}")

    print("\n## Drug-likeness (Lipinski Rule of 5)")
    print(f"  Lipinski-compliant : {lipinski_count} / {len(kept)} ({lipinski_count/len(kept)*100:.1f}%)")
    print(f"  Non-compliant     : {len(kept) - lipinski_count}")

    print("\n## Element composition (total atoms across all molecules)")
    total = sum(element_counter.values())
    for element, count in element_counter.most_common():
        print(f"  {element:4s} {count:8d}  ({count/total*100:.2f}%)")

    print("\n## Scaffold diversity")
    n_unique_scaffolds = len(scaffolds)
    print(f"  unique scaffolds : {n_unique_scaffolds}")
    print(f"  molecules / scaffold : mean={len(kept)/n_unique_scaffolds:.1f}, "
          f"median={np.median(list(scaffolds.values())):.0f}, "
          f"max={max(scaffolds.values())}")

    print("\n## Top 20 scaffolds by frequency")
    for scaffold, count in scaffolds.most_common(20):
        if scaffold == "":
            name = "(acyclic)"
        else:
            name = scaffold[:60]
        print(f"  {count:5d}  {name}")

    # Save to JSON
    out = {
        "n_parsed": len(raw),
        "n_kept": len(kept),
        "retention_rate": len(kept) / max(len(raw), 1),
        "mw_stats": {
            "mean": float(np.mean(mw_list)),
            "median": float(np.median(mw_list)),
            "p25": float(np.percentile(mw_list, 25)),
            "p75": float(np.percentile(mw_list, 75)),
            "max": float(max(mw_list)),
        },
        "heavy_atoms_stats": {
            "mean": float(np.mean(n_heavy_list)),
            "median": float(np.median(n_heavy_list)),
            "max": int(max(n_heavy_list)),
        },
        "lipinski_compliant_fraction": lipinski_count / len(kept),
        "element_composition": dict(element_counter.most_common()),
        "n_unique_scaffolds": n_unique_scaffolds,
        "scaffold_concentration_top20": sum(count for _, count in scaffolds.most_common(20)) / len(kept),
        "top_20_scaffolds": [
            {"scaffold_smiles": s if s else "(acyclic)", "count": c}
            for s, c in scaffolds.most_common(20)
        ],
    }
    out_path = ROOT / "experiments" / "results_overnight" / "dataset_composition.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[analyze] wrote {out_path}")


if __name__ == "__main__":
    main()
