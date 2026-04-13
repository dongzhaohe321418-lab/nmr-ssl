"""Conformal calibration for NMR shift predictors.

Given a trained model and a held-out calibration set, compute a per-atom
residual quantile so that future predictions can be wrapped in a symmetric
interval with rigorous marginal coverage.

Method: *Split conformal regression* (Vovk, Gammerman, Shafer; Lei et al.
2018). For each atom in the calibration set compute the absolute residual
|y_true - y_pred|, take the empirical (1 - alpha) quantile with a finite-sample
correction, and use that quantile as the half-width of the prediction
interval. Coverage guarantee:

    P(y_test in [y_pred - q, y_pred + q]) >= 1 - alpha

holds marginally over the i.i.d. joint distribution of calibration and test
atoms. The guarantee is distribution-free — no assumption on the predictor
or on the true residual distribution.

Usage
-----
    conformal = ConformalCalibrator(alpha=0.05)
    conformal.fit(residuals)            # 1-D array of |y_true - y_pred| on cal set
    q = conformal.quantile()            # scalar half-width for 95% coverage
    lo, hi = conformal.intervals(y_pred)  # (y_pred - q, y_pred + q)
    covered = conformal.coverage(y_true, y_pred)  # empirical coverage on test set
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalCalibrator:
    alpha: float = 0.05
    _quantile: float | None = None
    _n_calibration: int = 0

    def fit(self, residuals: np.ndarray) -> float:
        """Compute and store the conformal quantile from a pool of absolute residuals."""
        residuals = np.asarray(residuals, dtype=np.float64)
        residuals = residuals[np.isfinite(residuals)]
        n = residuals.size
        if n == 0:
            raise ValueError("need at least one calibration residual")
        # Finite-sample corrected quantile level
        k = math.ceil((n + 1) * (1 - self.alpha))
        k = min(k, n)
        sorted_res = np.sort(residuals)
        self._quantile = float(sorted_res[k - 1])
        self._n_calibration = n
        return self._quantile

    def quantile(self) -> float:
        if self._quantile is None:
            raise RuntimeError("call fit() first")
        return self._quantile

    def intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.quantile()
        y_pred = np.asarray(y_pred, dtype=np.float64)
        return y_pred - q, y_pred + q

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        q = self.quantile()
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        covered = np.abs(y_true - y_pred) <= q
        return {
            "n": int(covered.size),
            "covered_frac": float(np.mean(covered)),
            "target_coverage": 1 - self.alpha,
            "half_width_ppm": q,
        }

    def structure_verification_score(
        self,
        observed_shifts: np.ndarray,
        predicted_shifts: np.ndarray,
    ) -> dict[str, float]:
        """Decide whether a proposed structure is consistent with observed shifts.

        We assume the observed and predicted shifts correspond to the same set
        of atoms in the same order. The structure is declared consistent if
        all residuals are within the conformal half-width. We also report a
        continuous consistency score = the fraction of atoms within the
        interval and the worst residual.

        Returns a dict with: n_atoms, n_within_interval, fraction_within,
        worst_residual_ppm, consistent_at_alpha, alpha, half_width_ppm.
        """
        q = self.quantile()
        observed = np.asarray(observed_shifts, dtype=np.float64)
        predicted = np.asarray(predicted_shifts, dtype=np.float64)
        if observed.shape != predicted.shape:
            raise ValueError(
                f"shape mismatch observed={observed.shape} predicted={predicted.shape}"
            )
        residuals = np.abs(observed - predicted)
        within = residuals <= q
        return {
            "n_atoms": int(observed.size),
            "n_within_interval": int(np.sum(within)),
            "fraction_within": float(np.mean(within)),
            "worst_residual_ppm": float(np.max(residuals)),
            "consistent_at_alpha": bool(np.all(within)),
            "alpha": self.alpha,
            "half_width_ppm": q,
        }
