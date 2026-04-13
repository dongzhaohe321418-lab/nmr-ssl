"""Build synthetic 2-D HSQC cross-peak sets from NMRShiftDB2.

An HSQC cross-peak is the correlation between a $^1$H shift and the $^{13}$C
shift of its directly-bonded carbon. Given NMRShiftDB2's 1-D $^1$H spectrum
(indexed by heavy atom) and 1-D $^{13}$C spectrum (indexed by atom), we can
compute the HSQC cross-peak set deterministically:

    HSQC(mol) = { (mean(delta_H_i : H on C_k), delta_C_k) : C_k has >=1 H }

This is "synthetic" in the sense that we derive it from the 1-D labels, not
from a genuine 2-D experiment. But the set is equivalent to what a 2-D HSQC
would report, and it is the same multiset a literature-mining pipeline would
extract from a 2-D figure.

We use these sets to simulate the "unassigned 2-D SSL" setting: for each
molecule we know the HSQC peak multiset (no atom assignment), and we train
a joint 1-H/13-C predictor to match the set via the 2-D sort-match loss.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import Molecule, iter_nmrshiftdb2_sdf, parse_spectrum_field


@dataclass
class HSQCMolecule:
    """A molecule with both 13C and 1H spectra plus a derived HSQC peak set."""

    mol: Chem.Mol
    smiles: str
    nmr_id: str
    # 13C data indexed by RDKit atom index of the carbon
    c_shift_by_atom: dict[int, float]
    # 1H data: for each heavy atom, mean of its attached-H shifts
    h_mean_by_heavy_atom: dict[int, float]
    # HSQC cross-peaks: list of (H_shift, C_shift) tuples, one per C with H
    hsqc_peaks: list[tuple[float, float]]
    # Carbon atom indices that contribute a HSQC peak (for atom-assignment tests)
    hsqc_c_atoms: list[int]
    # Solvent, if any
    solvent: str | None = None

    @property
    def n_hsqc_peaks(self) -> int:
        return len(self.hsqc_peaks)

    @property
    def n_atoms(self) -> int:
        return self.mol.GetNumAtoms()


def _collect_molecule_spectra(sdf_path: str | Path, max_records: int = 20000):
    """Return two dicts keyed by molecule id (sans spectrum key) with parsed 13C and 1H data."""
    c_records: dict[str, Molecule] = {}
    h_records: dict[str, Molecule] = {}

    for rec in iter_nmrshiftdb2_sdf(sdf_path, nucleus="13C", max_records=max_records):
        base = rec.nmrshift_id.split("::")[0]
        if base not in c_records:
            c_records[base] = rec

    for rec in iter_nmrshiftdb2_sdf(sdf_path, nucleus="1H", max_records=max_records):
        base = rec.nmrshift_id.split("::")[0]
        if base not in h_records:
            h_records[base] = rec

    return c_records, h_records


def _mean_h_shift_by_heavy_atom(h_rec: Molecule) -> dict[int, float]:
    """Group 1-H peaks by their heavy-atom index and return the mean shift."""
    groups: dict[int, list[float]] = {}
    for shift, atom_idx in h_rec.peaks:
        groups.setdefault(atom_idx, []).append(shift)
    return {idx: sum(vs) / len(vs) for idx, vs in groups.items()}


def build_hsqc_molecules(
    sdf_path: str | Path,
    *,
    max_records: int = 20000,
    max_atoms: int = 60,
    min_hsqc_peaks: int = 3,
) -> list[HSQCMolecule]:
    """Return the list of clean, filtered HSQC molecules."""
    c_records, h_records = _collect_molecule_spectra(sdf_path, max_records=max_records)
    common_ids = set(c_records.keys()) & set(h_records.keys())

    kept: list[HSQCMolecule] = []
    for base_id in common_ids:
        c_rec = c_records[base_id]
        h_rec = h_records[base_id]
        mol = c_rec.mol  # both records reference the same molecule

        if mol.GetNumAtoms() > max_atoms:
            continue

        # Build 13C dict: atom idx -> shift. Must be non-degenerate (each C with
        # exactly one peak).
        c_shift_by_atom: dict[int, float] = {}
        valid = True
        for shift, idx in c_rec.peaks:
            if idx < 0 or idx >= mol.GetNumAtoms():
                valid = False
                break
            if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
                valid = False
                break
            if idx in c_shift_by_atom:
                valid = False
                break
            c_shift_by_atom[idx] = shift
        if not valid:
            continue

        # Check every C atom is in the 13C dict (non-degenerate)
        c_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
        if set(c_shift_by_atom.keys()) != set(c_indices):
            continue

        # Build 1-H mean shift dict grouped by heavy atom (NMRShiftDB2 convention)
        h_by_heavy = _mean_h_shift_by_heavy_atom(h_rec)
        if not h_by_heavy:
            continue

        # HSQC: for each C atom with an entry in h_by_heavy, create a cross-peak
        hsqc_peaks = []
        hsqc_c_atoms = []
        for c_idx in c_indices:
            if c_idx in h_by_heavy:
                hsqc_peaks.append((h_by_heavy[c_idx], c_shift_by_atom[c_idx]))
                hsqc_c_atoms.append(c_idx)

        if len(hsqc_peaks) < min_hsqc_peaks:
            continue

        smiles = Chem.MolToSmiles(mol)
        kept.append(
            HSQCMolecule(
                mol=mol,
                smiles=smiles,
                nmr_id=base_id,
                c_shift_by_atom=c_shift_by_atom,
                h_mean_by_heavy_atom=h_by_heavy,
                hsqc_peaks=hsqc_peaks,
                hsqc_c_atoms=hsqc_c_atoms,
                solvent=c_rec.solvent or h_rec.solvent,
            )
        )

    return kept


def quick_stats(molecules: list[HSQCMolecule]) -> None:
    import numpy as np

    print(f"kept {len(molecules)} molecules")
    if not molecules:
        return
    n_atoms = [m.n_atoms for m in molecules]
    n_c = [len(m.c_shift_by_atom) for m in molecules]
    n_hsqc = [m.n_hsqc_peaks for m in molecules]
    all_h = [p[0] for m in molecules for p in m.hsqc_peaks]
    all_c = [p[1] for m in molecules for p in m.hsqc_peaks]
    print(f"  atoms     : mean={np.mean(n_atoms):.1f}  median={np.median(n_atoms):.0f}  max={max(n_atoms)}")
    print(f"  C atoms   : mean={np.mean(n_c):.1f}  median={np.median(n_c):.0f}")
    print(f"  HSQC peaks: mean={np.mean(n_hsqc):.1f}  median={np.median(n_hsqc):.0f}  max={max(n_hsqc)}")
    print(f"  H shifts  : range [{min(all_h):.2f}, {max(all_h):.2f}]  mean={np.mean(all_h):.2f}")
    print(f"  C shifts  : range [{min(all_c):.2f}, {max(all_c):.2f}]  mean={np.mean(all_c):.2f}")


if __name__ == "__main__":
    from pathlib import Path

    sdf = Path(__file__).resolve().parents[2] / "data" / "nmrshiftdb2withsignals.sd"
    molecules = build_hsqc_molecules(sdf, max_records=20000)
    quick_stats(molecules)
    if molecules:
        print("\nExample molecule:")
        m = molecules[0]
        print(f"  SMILES : {m.smiles}")
        print(f"  n_atoms: {m.n_atoms}")
        print(f"  HSQC peaks (H, C):")
        for h, c in m.hsqc_peaks:
            print(f"    ({h:6.2f}, {c:7.2f})")
