"""v0.3.10 — feature reduction (Section 13).

Do not train a dense logistic router over a huge hidden dimension without
checking conditioning. This module supports:

- raw activations (no reduction)
- PCA projection (fitted on TRAIN only — Section 34)
- existing TopKFeatureSelector (via the existing routing module)

PCA must be fitted on TRAIN only. No dev/calibration/final data may
influence the PCA basis.
"""

from __future__ import annotations

import numpy as np


class PCAFeatureReducer:
    """PCA feature reduction fitted on TRAIN only (Section 13).

    Parameters
    ----------
    n_components : int
        Target dimension (e.g. 16, 32, 64, 128). Tune on development
        regret (Section 34).
    """

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be > 0")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "PCAFeatureReducer":
        """Fit PCA on training activations.

        Parameters
        ----------
        x : np.ndarray
            Shape ``[N, D]``. Must be TRAIN only.
        """
        if x.ndim != 2:
            raise ValueError("x must be [N, D]")
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN/Inf")
        n, d = x.shape
        k = min(self.n_components, d, n)
        if k < 1:
            raise ValueError("not enough samples/features for PCA")
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        # Use SVD for numerical stability.
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = Vt[:k].astype(np.float32)
        total_var = float((S ** 2).sum())
        if total_var > 0:
            self.explained_variance_ratio_ = (S[:k] ** 2 / total_var).astype(np.float32)
        else:
            self.explained_variance_ratio_ = np.zeros(k, dtype=np.float32)
        # Update n_components to the actual number used.
        self.n_components = k
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project activations into the reduced space."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA not fitted")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x must be [N, D]")
        return ((x - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


__all__ = ["PCAFeatureReducer"]
