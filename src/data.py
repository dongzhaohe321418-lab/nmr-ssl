"""NMRShiftDB2 SDF parser and PyTorch dataset.

NMRShiftDB2 SDF files store each molecule as a standard MOL block followed by
property fields. The ones we care about are named like "Spectrum 13C 0",
"Spectrum 1H 0", and contain peak lists in a semicolon-delimited form:

    "107.5;0.0;15|128.3;0.0;14|..."
            ^      ^
         shift   atom index (0-based in the molecule's atom list)

The middle field is the peak multiplicity / intensity placeholder; we ignore it
for chemical-shift prediction.

A single molecule can have multiple spectra per nucleus (for different solvents,
different conformers, different references). We keep each as a separate example
and record the spectrum key as metadata.

This module deliberately has zero dependency on torch_geometric — we pad graphs
to a fixed max size and use dense adjacency tensors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_ATOM_TYPES = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
_ATOM_TYPE_TO_IDX = {sym: i for i, sym in enumerate(_ATOM_TYPES)}
_NUM_ATOM_FEATS = len(_ATOM_TYPES) + 7  # one-hot type + degree + charge + aromatic + hyb + nH + inring

_HYB_MAP = {
    Chem.HybridizationType.SP: 0,
    Chem.HybridizationType.SP2: 1,
    Chem.HybridizationType.SP3: 2,
    Chem.HybridizationType.SP3D: 3,
    Chem.HybridizationType.SP3D2: 3,
}


def atom_features(atom: Chem.Atom) -> list[float]:
    feats = [0.0] * _NUM_ATOM_FEATS
    sym = atom.GetSymbol()
    if sym in _ATOM_TYPE_TO_IDX:
        feats[_ATOM_TYPE_TO_IDX[sym]] = 1.0
    base = len(_ATOM_TYPES)
    feats[base + 0] = atom.GetDegree() / 6.0
    feats[base + 1] = atom.GetFormalCharge() / 2.0
    feats[base + 2] = 1.0 if atom.GetIsAromatic() else 0.0
    feats[base + 3] = _HYB_MAP.get(atom.GetHybridization(), -1) / 3.0
    feats[base + 4] = atom.GetTotalNumHs() / 4.0
    feats[base + 5] = 1.0 if atom.IsInRing() else 0.0
    feats[base + 6] = atom.GetMass() / 35.0
    return feats


_PEAK_PATTERN = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*;\s*[^;|]*;\s*(\d+)")


def parse_spectrum_field(field: str) -> list[tuple[float, int]]:
    """Parse a Spectrum 13C / Spectrum 1H field into a list of (shift, atom_idx).

    The NMRShiftDB2 format is::

        "107.5;0.0;15|128.3;0.0;14|..."

    Some records have trailing whitespace, missing middle fields, or different
    separators. We extract all (float; anything; int) triples we can find.
    """
    peaks = []
    for m in _PEAK_PATTERN.finditer(field):
        shift = float(m.group(1))
        atom_idx = int(m.group(2))
        peaks.append((shift, atom_idx))
    return peaks


def extract_solvent(prop_dict: dict[str, str], spec_key: str) -> str | None:
    """Try to extract solvent information for a given spectrum.

    NMRShiftDB2 "Solvent" field is multi-line, one line per spectrum, with the
    form ``"<spec_idx>:<solvent_name>"``. E.g. ``"0:Chloroform-D1 (CDCl3)"``.
    We parse the spectrum index from ``spec_key`` (e.g. ``"Spectrum 13C 2"``)
    and return the matching line's solvent name.
    """
    raw = prop_dict.get("Solvent")
    if raw is None:
        return None
    # spec_key looks like "Spectrum 13C 2" → we want the trailing int
    parts = spec_key.strip().split()
    if not parts:
        return None
    try:
        spec_idx = int(parts[-1])
    except ValueError:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        prefix, _, name = line.partition(":")
        try:
            idx = int(prefix.strip())
        except ValueError:
            continue
        if idx == spec_idx:
            return _normalize_solvent_name(name.strip())
    return None


_SOLVENT_CANONICAL: list[tuple[str, str]] = [
    # (substring match on lowercased name → canonical short name)
    ("chloroform", "CDCl3"),
    ("cdcl3", "CDCl3"),
    ("dimethylsulphoxide", "DMSO"),
    ("dimethyl sulphoxide", "DMSO"),
    ("dimethyl sulfoxide", "DMSO"),
    ("dmso", "DMSO"),
    ("methanol", "MeOD"),
    ("cd3od", "MeOD"),
    ("deuteriumoxide", "D2O"),
    ("deuterium oxide", "D2O"),
    ("d2o", "D2O"),
    ("benzene", "C6D6"),
    ("c6d6", "C6D6"),
    ("acetone", "Acetone"),
    ("pyridin", "Pyridine"),
    ("water", "H2O"),
    ("acetonitrile", "CD3CN"),
    ("thf", "THF"),
    ("tetrahydrofuran", "THF"),
    ("toluene", "Toluene"),
    ("dichloromethane", "DCM"),
    ("cd2cl2", "DCM"),
]


def _normalize_solvent_name(name: str) -> str:
    key = name.lower().strip()
    if not key or "unreported" in key:
        return "Unreported"
    for needle, canonical in _SOLVENT_CANONICAL:
        if needle in key:
            return canonical
    return name.strip()


@dataclass
class Molecule:
    """A parsed NMRShiftDB2 entry with a single nucleus spectrum attached."""

    mol: Chem.Mol
    smiles: str
    nucleus: str  # "13C" or "1H"
    peaks: list[tuple[float, int]]  # (shift, atom_idx) in original SDF indexing
    solvent: str | None
    nmrshift_id: str

    @property
    def n_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    def target_for_nucleus(self) -> tuple[list[float], list[int]] | None:
        """Return (shifts, atom_indices) where each atom_idx points to an atom
        of the correct nucleus element in the molecule. Returns None if the
        spectrum is inconsistent with the molecule (e.g., atom index out of
        range, duplicate assignments).
        """
        element = "C" if self.nucleus == "13C" else "H"
        shifts: list[float] = []
        indices: list[int] = []
        seen_indices: set[int] = set()
        for shift, idx in self.peaks:
            if idx < 0 or idx >= self.mol.GetNumAtoms():
                return None
            atom = self.mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() != element:
                return None
            if idx in seen_indices:
                return None
            seen_indices.add(idx)
            shifts.append(shift)
            indices.append(idx)
        return shifts, indices


def iter_nmrshiftdb2_sdf(
    sdf_path: str | Path,
    *,
    nucleus: str = "13C",
    max_records: int | None = None,
) -> Iterator[Molecule]:
    """Yield Molecule records from an NMRShiftDB2 SDF file, one per spectrum.

    If a molecule has multiple spectra for the target nucleus (e.g., recorded
    in different solvents), each is yielded as a separate Molecule.
    """
    path = Path(sdf_path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)

    yielded = 0
    for mol_idx, mol in enumerate(supplier):
        if mol is None:
            continue
        prop_dict = {key: mol.GetProp(key) for key in mol.GetPropNames()}
        smiles = Chem.MolToSmiles(mol)

        for key, value in prop_dict.items():
            if not key.startswith(f"Spectrum {nucleus}"):
                continue
            peaks = parse_spectrum_field(value)
            if not peaks:
                continue
            solvent = extract_solvent(prop_dict, key)
            nmr_id = prop_dict.get("nmrshiftdb2 ID", f"mol{mol_idx}")
            record = Molecule(
                mol=mol,
                smiles=smiles,
                nucleus=nucleus,
                peaks=peaks,
                solvent=solvent,
                nmrshift_id=f"{nmr_id}::{key}",
            )
            yield record
            yielded += 1
            if max_records is not None and yielded >= max_records:
                return


def mol_to_graph_tensors(mol: Chem.Mol) -> tuple[Tensor, Tensor]:
    """Return (atom_features (N, F), adjacency (N, N)) as torch tensors."""
    n = mol.GetNumAtoms()
    feats = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32
    )
    adj = torch.zeros((n, n), dtype=torch.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    # Add self-loops (typical for GNNs)
    adj += torch.eye(n)
    return feats, adj


class NMRDataset(Dataset):
    """Atom-assigned NMR dataset for supervised training.

    Each item is a dict with keys:
        x           : (N, F) atom features
        adj         : (N, N) adjacency
        target_atom : (K,) indices of atoms with observed shifts
        target_shift: (K,) observed shifts for those atoms
        solvent     : Optional[str]
        molecule_id : str
    """

    def __init__(self, molecules: list[Molecule]):
        self.molecules = molecules

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.molecules[idx]
        target = rec.target_for_nucleus()
        assert target is not None, f"inconsistent record {rec.nmrshift_id}"
        shifts, indices = target
        x, adj = mol_to_graph_tensors(rec.mol)
        return {
            "x": x,
            "adj": adj,
            "target_atom": torch.tensor(indices, dtype=torch.long),
            "target_shift": torch.tensor(shifts, dtype=torch.float32),
            "solvent": rec.solvent,
            "molecule_id": rec.nmrshift_id,
        }


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    """Return the canonical SMILES of the Bemis-Murcko scaffold of ``mol``.

    Degenerate scaffold cases (acyclic / tiny molecules) return an empty string,
    which we treat as a unique "no-scaffold" bucket for split purposes.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return ""


def scaffold_split(
    molecules: list["Molecule"],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Bemis-Murcko scaffold split of molecules into train / val / test.

    Molecules sharing a scaffold are assigned to the same split, so the test
    set contains chemical scaffolds never seen in training.

    Implementation: we partition scaffolds into three pools by a seeded
    random assignment, placing each scaffold into train/val/test with the
    target proportions, and re-anchoring the largest scaffold into train to
    avoid the pathological case where a dominant scaffold (e.g. benzene)
    overflows val/test. This is more robust under seed rotation than the
    old "largest-first rotate by seed" algorithm, which could leave val
    empty for some seed values.

    Returns lists of dataset indices for train, val, and test.
    """
    import random as _random

    scaffold_to_idx: dict[str, list[int]] = {}
    for i, m in enumerate(molecules):
        sm = murcko_scaffold_smiles(m.mol)
        scaffold_to_idx.setdefault(sm, []).append(i)

    groups = sorted(scaffold_to_idx.values(), key=lambda g: -len(g))
    total = len(molecules)
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)

    # Force the largest group into train if it's larger than the val or test
    # target — otherwise the greedy fill algorithm puts it wherever it fits
    # first and can overflow val/test. After this, shuffle the remaining
    # groups with a seeded RNG so different seeds see different assignments.
    if groups and len(groups[0]) > max(n_val, total - n_train - n_val):
        forced_train = groups[0]
        remaining = groups[1:]
    else:
        forced_train = []
        remaining = groups

    rng = _random.Random(seed)
    rng.shuffle(remaining)

    train: list[int] = list(forced_train)
    val: list[int] = []
    test: list[int] = []
    for g in remaining:
        # Fill val and test first in proportion, then dump the rest into train.
        if len(val) + len(g) <= n_val:
            val.extend(g)
        elif len(test) + len(g) <= total - n_train - n_val:
            test.extend(g)
        else:
            train.extend(g)
    return train, val, test


def pad_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad atom features, adjacency, and target arrays to a common size.

    Returns a dict of batched tensors plus a boolean ``atom_mask`` of shape
    (B, N_max) that is True for real atoms.
    """
    B = len(batch)
    n_max = max(item["x"].shape[0] for item in batch)
    k_max = max(item["target_shift"].shape[0] for item in batch)
    feat_dim = batch[0]["x"].shape[1]

    x = torch.zeros(B, n_max, feat_dim)
    adj = torch.zeros(B, n_max, n_max)
    atom_mask = torch.zeros(B, n_max, dtype=torch.bool)
    target_atom = torch.full((B, k_max), -1, dtype=torch.long)
    target_shift = torch.zeros(B, k_max)
    target_mask = torch.zeros(B, k_max, dtype=torch.bool)
    solvents = []
    ids = []

    for i, item in enumerate(batch):
        n = item["x"].shape[0]
        k = item["target_shift"].shape[0]
        x[i, :n] = item["x"]
        adj[i, :n, :n] = item["adj"]
        atom_mask[i, :n] = True
        target_atom[i, :k] = item["target_atom"]
        target_shift[i, :k] = item["target_shift"]
        target_mask[i, :k] = True
        solvents.append(item["solvent"])
        ids.append(item["molecule_id"])

    return {
        "x": x,
        "adj": adj,
        "atom_mask": atom_mask,
        "target_atom": target_atom,
        "target_shift": target_shift,
        "target_mask": target_mask,
        "solvents": solvents,
        "molecule_ids": ids,
    }
