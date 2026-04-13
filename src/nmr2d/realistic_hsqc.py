"""Realistic HSQC degradation pipeline.

Takes the synthetic HSQC peak multiset (H_mean_at_C, delta_C) derived from
NMRShiftDB2's assigned 1-H / 13-C spectra, and subjects it to the degradation
modes a literature-mined HSQC peak list is likely to show. This is the closest
we can get to "real HSQC data" without an external experimental corpus,
because NMRShiftDB2's native HSQC SDF records are empty stubs (the 2-D
spectra are referenced but the peak lists are not provided in the SDF dump).

Degradation modes — each can be toggled independently:

1. Per-peak Gaussian noise: sigma_H, sigma_C ppm on each (H, C) coordinate.
2. Missing-peak dropout: each peak is independently dropped with probability p_drop
   (simulating peaks below SNR threshold, obscured by diagonal, or cropped
   from a published figure).
3. Peak merging: adjacent peaks within (delta_H, delta_C) tolerance are
   merged to their mean (simulating "we could not resolve these as separate
   cross-peaks in the published spectrum").
4. Per-molecule systematic solvent offset: one constant offset drawn from
   (N(0, 0.05), N(0, 1.0)) ppm added to every peak in a molecule (simulating
   solvent / field / temperature reference-shift differences between papers).

Usage
-----
    degrader = RealisticHSQCDegrader(
        sigma_h=0.02, sigma_c=0.5, p_drop=0.1,
        merge_h=0.05, merge_c=1.0, offset_std_h=0.05, offset_std_c=1.0,
    )
    degraded_peaks = degrader(original_peaks, rng=Generator)

The returned peak list preserves the `(h, c)` tuple convention and carries
no atom-identity information (consistent with the unassigned-HSQC setting).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RealisticHSQCDegrader:
    sigma_h: float = 0.02       # ppm Gaussian noise on 1H axis
    sigma_c: float = 0.5        # ppm Gaussian noise on 13C axis
    p_drop: float = 0.10        # probability of dropping each peak
    merge_h: float = 0.05       # ppm 1H tolerance for merging
    merge_c: float = 1.0        # ppm 13C tolerance for merging
    offset_std_h: float = 0.05  # per-molecule global H offset (ppm)
    offset_std_c: float = 1.0   # per-molecule global C offset (ppm)

    def __call__(
        self,
        peaks: list[tuple[float, float]],
        rng: random.Random,
    ) -> list[tuple[float, float]]:
        if not peaks:
            return []

        # 1. Apply per-peak Gaussian noise
        noisy = [
            (h + rng.gauss(0, self.sigma_h), c + rng.gauss(0, self.sigma_c))
            for (h, c) in peaks
        ]

        # 2. Per-molecule systematic offset (solvent / field / reference)
        off_h = rng.gauss(0, self.offset_std_h)
        off_c = rng.gauss(0, self.offset_std_c)
        noisy = [(h + off_h, c + off_c) for (h, c) in noisy]

        # 3. Random peak dropout (simulate below-SNR or missing-from-figure)
        kept = [p for p in noisy if rng.random() > self.p_drop]
        if not kept:
            kept = [noisy[rng.randrange(len(noisy))]]

        # 4. Peak merging: greedy single-linkage within the tolerance box
        merged = _greedy_merge(kept, self.merge_h, self.merge_c)
        return merged


def _greedy_merge(
    peaks: list[tuple[float, float]],
    tol_h: float,
    tol_c: float,
) -> list[tuple[float, float]]:
    """Greedy single-linkage merging of peaks within an (tol_h, tol_c) box."""
    if len(peaks) <= 1:
        return list(peaks)
    groups: list[list[tuple[float, float]]] = []
    for p in peaks:
        placed = False
        for g in groups:
            for q in g:
                if abs(p[0] - q[0]) <= tol_h and abs(p[1] - q[1]) <= tol_c:
                    g.append(p)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([p])
    out = []
    for g in groups:
        h_mean = sum(q[0] for q in g) / len(g)
        c_mean = sum(q[1] for q in g) / len(g)
        out.append((h_mean, c_mean))
    return out
