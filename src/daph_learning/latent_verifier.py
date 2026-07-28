"""v0.3.10 — latent verifier experiment (Section 30).

An optional internal correctness feature. For each completed reasoning
trace capture ``h_start`` and ``h_end``. Define ``Δh = h_end - h_start``.
Maintain labeled centroids ``μ_correct`` and ``μ_incorrect``. Score::

    d_correct   = ||Δh - μ_correct||
    d_incorrect = ||Δh - μ_incorrect||
    s = d_incorrect - d_correct

Positive ``s`` means closer to the correct trajectory class. This can
later become a router feature. Do NOT use it as ground truth — it is
only an auxiliary learned signal.
"""

from __future__ import annotations

import numpy as np


class LatentVerifier:
    """Latent correctness verifier using trajectory centroids (Section 30)."""

    def __init__(self) -> None:
        self.mu_correct: np.ndarray | None = None
        self.mu_incorrect: np.ndarray | None = None

    def fit(
        self,
        correct_deltas: np.ndarray,
        incorrect_deltas: np.ndarray,
    ) -> None:
        """Fit centroids from labeled trajectory deltas.

        Parameters
        ----------
        correct_deltas : np.ndarray
            Shape ``[N_c, D]``. ``Δh = h_end - h_start`` for correct traces.
        incorrect_deltas : np.ndarray
            Shape ``[N_i, D]``. ``Δh`` for incorrect traces.
        """
        if correct_deltas.ndim != 2 or incorrect_deltas.ndim != 2:
            raise ValueError("deltas must be [N, D]")
        if correct_deltas.shape[1] != incorrect_deltas.shape[1]:
            raise ValueError("correct/incorrect delta dims must match")
        self.mu_correct = correct_deltas.mean(axis=0).astype(np.float32)
        self.mu_incorrect = incorrect_deltas.mean(axis=0).astype(np.float32)

    def score(self, delta_h: np.ndarray) -> float:
        """Latent correctness score: ``d_incorrect - d_correct``.

        Positive => closer to the correct trajectory class.
        """
        if self.mu_correct is None or self.mu_incorrect is None:
            raise RuntimeError("LatentVerifier not fitted")
        delta_h = np.asarray(delta_h, dtype=np.float32)
        d_correct = float(np.linalg.norm(delta_h - self.mu_correct))
        d_incorrect = float(np.linalg.norm(delta_h - self.mu_incorrect))
        return d_incorrect - d_correct

    def score_batch(self, deltas: np.ndarray) -> np.ndarray:
        if self.mu_correct is None or self.mu_incorrect is None:
            raise RuntimeError("LatentVerifier not fitted")
        deltas = np.asarray(deltas, dtype=np.float32)
        if deltas.ndim != 2:
            raise ValueError("deltas must be [N, D]")
        d_correct = np.linalg.norm(deltas - self.mu_correct, axis=1)
        d_incorrect = np.linalg.norm(deltas - self.mu_incorrect, axis=1)
        return d_incorrect - d_correct


__all__ = ["LatentVerifier"]
