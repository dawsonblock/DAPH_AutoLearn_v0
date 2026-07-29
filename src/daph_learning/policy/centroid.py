"""v0.3.10 — utility-weighted centroid baseline (Section 3, BASELINE A/B).

The weighted centroid is a **transparent baseline**, not the final
AutoLearn algorithm. It improves on the unweighted contrastive mean by
weighting each example by ``w_i = |ΔU_i| × c_i`` (utility importance ×
confidence), so that high-gap, high-confidence examples dominate the
direction while near-ties contribute nothing.

Important theoretical correction (Section 4): the utility gap measures the
*importance* of choosing the correct action. It does NOT identify the
causal activation direction that changes the policy. Therefore this
baseline must be compared against an actual learned decision boundary
(:mod:`daph_learning.policy.logistic`).

Weighted class means::

    μ_S = Σ_{i∈S} w_i h_i / Σ_{i∈S} w_i
    μ_L = Σ_{i∈L} w_i h_i / Σ_{i∈L} w_i
    v_centroid = μ_S - μ_L
"""

from __future__ import annotations

import numpy as np


def weighted_mean(
    activations: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Weighted mean of activation rows.

    Parameters
    ----------
    activations : np.ndarray
        Shape ``[N, D]``.
    weights : np.ndarray
        Shape ``[N]``. Non-negative.
    eps : float
        Effective sample weight below which we raise (zero effective weight).

    Returns
    -------
    np.ndarray
        Shape ``[D]`` weighted mean.

    Raises
    ------
    ValueError
        On shape mismatch, NaN/Inf in activations or weights, negative
        weights (Section 7), all-zero effective weights, or
        non-finite inputs.
    """
    if activations.ndim != 2:
        raise ValueError("activations must have shape [N, D]")
    if weights.ndim != 1:
        raise ValueError("weights must have shape [N]")
    if len(activations) != len(weights):
        raise ValueError("activation/weight length mismatch")
    if not np.all(np.isfinite(activations)):
        raise ValueError("activations contain NaN/Inf")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain NaN/Inf")
    # v0.3.10.1 — Section 7: reject negative weights.
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= eps:
        raise ValueError("effective sample weight is zero")
    return (activations * weights[:, None]).sum(axis=0) / total


def weighted_contrastive_mean(
    positive_activations: np.ndarray,
    positive_weights: np.ndarray,
    negative_activations: np.ndarray,
    negative_weights: np.ndarray,
    *,
    normalize: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """Utility-weighted contrastive mean direction: ``μ_pos - μ_neg``.

    This is BASELINE A (unweighted) or BASELINE B/D (utility-weighted)
    depending on the weights supplied. The caller is responsible for
    computing the weights via :func:`daph_learning.policy.weighting.utility_weight`.

    Parameters
    ----------
    positive_activations, positive_weights : np.ndarray
        Activations and weights for the "symbolic-preferred" class.
    negative_activations, negative_weights : np.ndarray
        Activations and weights for the "llm-preferred" class.
    normalize : bool
        If True, L2-normalize the resulting direction.
    eps : float
        Effective sample weight floor.

    Returns
    -------
    np.ndarray
        Shape ``[D]`` steering direction.
    """
    mu_pos = weighted_mean(positive_activations, positive_weights, eps=eps)
    mu_neg = weighted_mean(negative_activations, negative_weights, eps=eps)
    vector = mu_pos - mu_neg
    if not np.all(np.isfinite(vector)):
        raise ValueError("invalid steering vector")
    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("contrastive direction has zero norm")
        vector = vector / norm
    return vector.astype(np.float32)


def unweighted_contrastive_mean(
    positive_activations: np.ndarray,
    negative_activations: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Unweighted contrastive mean (BASELINE A): ``mean(pos) - mean(neg)``.

    Provided for the baseline matrix comparison (Section 25, baseline C).
    """
    if positive_activations.ndim != 2 or negative_activations.ndim != 2:
        raise ValueError("activations must be [N, D]")
    if positive_activations.shape[1] != negative_activations.shape[1]:
        raise ValueError("positive/negative hidden size mismatch")
    vector = positive_activations.mean(axis=0) - negative_activations.mean(axis=0)
    if normalize:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("contrastive direction has zero norm")
        vector = vector / norm
    return vector.astype(np.float32)


__all__ = [
    "unweighted_contrastive_mean",
    "weighted_contrastive_mean",
    "weighted_mean",
]
