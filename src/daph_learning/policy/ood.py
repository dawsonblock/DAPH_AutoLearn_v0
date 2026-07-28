"""v0.3.10 — OOD detection via Mahalanobis distance (Section 12).

The router must not blindly extrapolate to unseen activation distributions.
This module implements a simple first OOD detector using Mahalanobis
distance with regularized covariance.

For very high-dimensional activations, reduce dimension first (see
:mod:`daph_learning.policy.features`).
"""

from __future__ import annotations

import numpy as np


class MahalanobisOOD:
    """Mahalanobis-distance OOD detector (Section 12).

    Estimates the training activation mean ``μ`` and regularized
    covariance ``Σ``, then scores new activations by::

        d_M(h) = sqrt((h - μ)^T Σ^{-1} (h - μ))

    If ``d_M(h) > τ_OOD``, the task is out-of-distribution and should
    route to ABSTAIN or a conservative fallback.
    """

    def __init__(self, ridge: float = 1e-4) -> None:
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        self.ridge = ridge
        self.mean: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> None:
        """Fit the detector on training activations.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[N, D]``. Must be fitted on TRAIN only (Section 34).
        """
        if x.ndim != 2:
            raise ValueError("x must be [N, D]")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN/Inf")
        self.mean = x.mean(axis=0)
        if x.shape[0] < 2:
            cov = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
        else:
            cov = np.cov(x, rowvar=False)
        # Ensure 2-D covariance (np.cov returns a scalar for D=1).
        cov = np.atleast_2d(cov)
        cov = cov + self.ridge * np.eye(cov.shape[0])
        self.inv_cov = np.linalg.pinv(cov)

    def score(self, h: np.ndarray) -> float:
        """Mahalanobis distance of ``h`` from the training distribution.

        Parameters
        ----------
        h : np.ndarray
            Shape ``[D]`` or ``[N, D]``. For ``[N, D]`` returns a scalar
            (mean distance); use :meth:`score_batch` for per-sample scores.
        """
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("OOD detector not fitted")
        h = np.asarray(h, dtype=np.float64)
        if h.ndim == 2:
            return float(np.mean([self._score_single(row) for row in h]))
        return self._score_single(h)

    def _score_single(self, h: np.ndarray) -> float:
        delta = h - self.mean  # type: ignore[operator]
        value = float(delta @ self.inv_cov @ delta)  # type: ignore[operator]
        return float(np.sqrt(max(value, 0.0)))

    def score_batch(self, h: np.ndarray) -> np.ndarray:
        """Per-sample Mahalanobis distances for ``[N, D]`` input."""
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("OOD detector not fitted")
        h = np.asarray(h, dtype=np.float64)
        if h.ndim != 2:
            raise ValueError("h must be [N, D]")
        delta = h - self.mean
        # Vectorized: d_i = sqrt(sum(delta_i @ inv_cov * delta_i))
        mahal = np.einsum("ij,jk,ik->i", delta, self.inv_cov, delta)
        return np.sqrt(np.maximum(mahal, 0.0))

    def is_ood(self, h: np.ndarray, threshold: float) -> bool:
        """Return True if ``d_M(h) > threshold``."""
        return self.score(h) > threshold


__all__ = ["MahalanobisOOD"]
