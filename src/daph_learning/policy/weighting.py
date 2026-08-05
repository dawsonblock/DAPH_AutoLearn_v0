"""v0.3.10.1-alpha — utility-weighted sample weighting (Sections 2, 7, 9).

This module implements the weighting functions that connect the
counterfactual utility gap ``|ΔU|``, outcome confidence, and (for SNR
mode) the standard deviation of the utility estimate to per-example
training weights.

v0.3.10.1 — breaking change: the old ``WeightMode.GAP`` /
``WeightMode.SNR`` two-mode enum was a *correctness bug*. The
configuration exposed ``weight_mode: "gap" | "snr"`` but the
experience construction code always called :func:`utility_weight`
(the gap function), silently ignoring ``snr``. This release replaces
the two-mode enum with four explicit, fully-implemented modes:

============  ==================================================
UNIFORM       ``w_i = 1`` (regardless of gap or confidence)
ABSOLUTE_GAP  ``w_i = |ΔU_i| * c_i``
CLIPPED_GAP   ``w_i = clip(|ΔU_i| * c_i, 0, w_max)``
SNR           ``w_i = |ΔU_i| / (σ_ΔU_i + ε)``  (optionally × c_i),
              clipped to ``w_max``
============  ==================================================

Near-ties (``|ΔU| <= gap_threshold``) are *truncated to zero weight*
in all non-uniform modes (Section 7 Mode A). A positive floor would
accidentally make near-ties matter again, defeating the purpose of
utility-weighting. UNIFORM mode intentionally does NOT truncate ties
— the caller is responsible for combining UNIFORM with a target mode
that masks ties (e.g. ``TargetMode.HARD``) when tie-truncation is
desired.

A numerical stability floor ``min_weight`` is applied **only after**
gap filtering in non-uniform modes, and only to avoid division-by-zero
in downstream normalization — it is NOT a statistical minimum
importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class WeightMode(str, Enum):
    """Sample weighting mode (v0.3.10.1 — clean break).

    UNIFORM       : ``w_i = 1`` for all examples.
    ABSOLUTE_GAP  : ``w_i = |ΔU_i| * c_i`` (no clipping).
    CLIPPED_GAP   : ``w_i = clip(|ΔU_i| * c_i, 0, w_max)``.
    SNR           : ``w_i = clip(|ΔU_i| / (σ_ΔU_i + ε) * c_i, 0, w_max)``.

    The legacy ``GAP`` and ``SNR`` string aliases from v0.3.10 are NOT
    accepted — clean break per v0.3.10.1-alpha. Callers must migrate
    ``gap`` -> ``absolute_gap`` (or ``clipped_gap`` to keep the
    historical max_weight=1.0 clipping) and ``snr`` -> ``snr``.
    """

    UNIFORM = "uniform"
    ABSOLUTE_GAP = "absolute_gap"
    CLIPPED_GAP = "clipped_gap"
    SNR = "snr"

    @classmethod
    def from_str(cls, value: str) -> "WeightMode":
        """Parse a weight-mode string, raising ``ValueError`` on unknown values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown weight_mode {value!r}; expected one of "
            f"{[m.value for m in cls]}"
        )


class ZeroWeightPolicy(str, Enum):
    """v0.3.10.3.1 — behavior when a class has zero effective training
    weight (Section 2).

    Silently substituting the unweighted estimator for the weighted one
    is scientifically dangerous: it changes the estimator without
    recording it in provenance. This enum makes the behavior explicit.

    ERROR                : raise ``InsufficientEffectiveWeight`` (default).
    UNWEIGHTED_FALLBACK  : fall back to unweighted mean, record it in
                           provenance, and report the estimator as
                           ``weighted_centroid_with_unweighted_fallback``
                           rather than ``weighted_centroid``.
    """

    ERROR = "error"
    UNWEIGHTED_FALLBACK = "unweighted_fallback"

    @classmethod
    def from_str(cls, value: str) -> "ZeroWeightPolicy":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown zero_weight_policy {value!r}; expected one of "
            f"{[m.value for m in cls]}")


class InsufficientEffectiveWeight(ValueError):
    """Raised when a class has zero (or near-zero) effective training
    weight and ``ZeroWeightPolicy.ERROR`` is in effect (Section 2).

    A zero effective weight means the weighted estimator is undefined;
    silently falling back to the unweighted estimator would change the
    experiment without recording it.
    """


def require_effective_class_weight(
    weights: np.ndarray,
    class_name: str,
    eps: float = 1e-12,
) -> None:
    """Raise ``InsufficientEffectiveWeight`` if ``weights.sum() <= eps``.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative class weights.
    class_name : str
        Human-readable class label for the error message
        (e.g. ``"symbolic"``, ``"llm"``).
    eps : float
        Threshold below which the effective weight is considered zero.
    """
    total = float(np.asarray(weights, dtype=np.float64).sum())
    if total <= eps:
        raise InsufficientEffectiveWeight(
            f"{class_name} has zero effective training weight "
            f"(sum={total!r} <= eps={eps!r}); the weighted estimator is "
            f"undefined. Enable ZeroWeightPolicy.UNWEIGHTED_FALLBACK to "
            f"fall back to the unweighted estimator with recorded provenance."
        )


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for utility-weighted sample weighting.

    Attributes
    ----------
    min_weight : float
        Numerical stability floor applied AFTER gap filtering in
        non-uniform modes. This is NOT a statistical minimum importance
        — it only prevents division-by-zero in downstream normalization.
        Set to 0.0 by default (Section 7).
    max_weight : float
        Maximum weight (clipping) for CLIPPED_GAP and SNR modes.
    eps : float
        Numerical epsilon for division safety in SNR mode.
    gap_threshold : float
        Tasks with ``|ΔU| <= gap_threshold`` get zero weight in
        non-uniform modes (tie truncation, Section 7 Mode A).
        Must be >= 0. Ignored for UNIFORM mode.
    """

    min_weight: float = 0.0
    max_weight: float = 1.0
    eps: float = 1e-8
    gap_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.min_weight < 0:
            raise ValueError("min_weight must be >= 0")
        if self.max_weight < self.min_weight:
            raise ValueError("max_weight must be >= min_weight")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")
        if self.gap_threshold < 0:
            raise ValueError("gap_threshold must be >= 0")


def compute_weight(
    delta_u: float,
    confidence: float,
    mode: "WeightMode | str",
    gap_threshold: float,
    max_weight: float,
    *,
    sigma_delta_u: float | None = None,
    min_weight: float = 0.0,
    eps: float = 1e-8,
) -> float:
    """Compute a single per-example training weight.

    Parameters
    ----------
    delta_u : float
        Utility difference. Must be finite.
    confidence : float
        Combined outcome confidence in ``[0, 1]``.
    mode : WeightMode | str
        One of ``UNIFORM``, ``ABSOLUTE_GAP``, ``CLIPPED_GAP``, ``SNR``.
    gap_threshold : float
        Tie-band. Tasks with ``|ΔU| <= gap_threshold`` get zero weight
        in non-uniform modes. Must be ``>= 0``.
    max_weight : float
        Clipping ceiling for CLIPPED_GAP and SNR modes. Must be ``>= 0``.
    sigma_delta_u : float | None
        Required for SNR mode. Standard deviation of the utility
        estimate. Must be ``>= 0``.
    min_weight : float
        Numerical stability floor (applied after gap filtering in
        non-uniform modes). Must be ``>= 0``.
    eps : float
        Numerical epsilon for SNR division. Must be ``> 0``.

    Returns
    -------
    float
        The per-example weight. Always non-negative.

    Raises
    ------
    ValueError
        On non-finite ``delta_u``, out-of-range ``confidence``, unknown
        mode, or SNR mode called without ``sigma_delta_u``.
    """
    if not np.isfinite(delta_u):
        raise ValueError("delta_u must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if gap_threshold < 0:
        raise ValueError("gap_threshold must be >= 0")
    if max_weight < 0:
        raise ValueError("max_weight must be >= 0")
    if min_weight < 0:
        raise ValueError("min_weight must be >= 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if isinstance(mode, str):
        mode = WeightMode.from_str(mode)
    if not isinstance(mode, WeightMode):
        raise ValueError(f"mode must be a WeightMode or str, got {type(mode)}")

    if mode == WeightMode.UNIFORM:
        # UNIFORM intentionally does not truncate ties — the caller
        # combines it with TargetMode.HARD if tie-truncation is desired.
        return 1.0

    # Non-uniform modes: truncate ties to zero (Section 7 Mode A).
    if abs(delta_u) <= gap_threshold:
        return 0.0

    if mode == WeightMode.ABSOLUTE_GAP:
        raw = abs(delta_u) * confidence
        return float(max(raw, min_weight))

    if mode == WeightMode.CLIPPED_GAP:
        raw = abs(delta_u) * confidence
        weighted = max(raw, min_weight)
        return float(min(weighted, max_weight))

    if mode == WeightMode.SNR:
        if sigma_delta_u is None:
            raise ValueError("SNR mode requires sigma_delta_u")
        if sigma_delta_u < 0:
            raise ValueError("sigma_delta_u must be >= 0")
        raw = abs(delta_u) / (sigma_delta_u + eps)
        weighted = max(raw * confidence, min_weight)
        return float(min(weighted, max_weight))

    raise ValueError(f"Unknown weight mode: {mode}")


def compute_weights_batch(
    delta_u: np.ndarray,
    confidence: np.ndarray,
    mode: "WeightMode | str",
    gap_threshold: float,
    max_weight: float,
    *,
    sigma_delta_u: np.ndarray | None = None,
    min_weight: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Vectorized per-example weight computation.

    All arguments are broadcastable. Returns a float64 array of the
    broadcasted shape, with non-negative entries.

    Raises
    ------
    ValueError
        On shape mismatch between required arrays, non-finite inputs,
        or SNR mode called without ``sigma_delta_u``.
    """
    if isinstance(mode, str):
        mode = WeightMode.from_str(mode)
    du = np.asarray(delta_u, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if du.shape != conf.shape:
        raise ValueError(
            f"delta_u shape {du.shape} != confidence shape {conf.shape}")
    if not np.all(np.isfinite(du)):
        raise ValueError("delta_u contains NaN/Inf")
    if not np.all(np.isfinite(conf)):
        raise ValueError("confidence contains NaN/Inf")
    if np.any(conf < 0.0) or np.any(conf > 1.0):
        raise ValueError("confidence must be in [0, 1]")
    if gap_threshold < 0:
        raise ValueError("gap_threshold must be >= 0")
    if max_weight < 0:
        raise ValueError("max_weight must be >= 0")

    if mode == WeightMode.UNIFORM:
        return np.ones(du.shape, dtype=np.float64)

    tie_mask = np.abs(du) <= gap_threshold
    if mode == WeightMode.ABSOLUTE_GAP:
        raw = np.abs(du) * conf
        out = np.maximum(raw, min_weight)
    elif mode == WeightMode.CLIPPED_GAP:
        raw = np.abs(du) * conf
        out = np.clip(raw, min_weight, max_weight)
    elif mode == WeightMode.SNR:
        if sigma_delta_u is None:
            raise ValueError("SNR mode requires sigma_delta_u")
        sig = np.asarray(sigma_delta_u, dtype=np.float64)
        if sig.shape != du.shape:
            raise ValueError(
                f"sigma_delta_u shape {sig.shape} != delta_u shape {du.shape}")
        if np.any(sig < 0):
            raise ValueError("sigma_delta_u must be >= 0")
        snr = np.abs(du) / (sig + eps)
        out = np.clip(snr * conf, min_weight, max_weight)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    out = np.where(tie_mask, 0.0, out)
    return out.astype(np.float64)


# ------------------------------------------------------------------
# Backward-compatible thin wrappers (kept for callers that already
# import these names; they all route through :func:`compute_weight`
# with the equivalent v0.3.10.1 mode).
# ------------------------------------------------------------------

def utility_weight(
    reward_gap: float,
    confidence: float,
    cfg: WeightConfig,
) -> float:
    """v0.3.10.1 — thin wrapper around :func:`compute_weight` using
    ``CLIPPED_GAP`` semantics (the historical default behaviour of this
    function: ``clip(|ΔU| * c, w_min, w_max)`` with tie truncation).

    Kept so existing callers continue to work after the clean break
    on the ``WeightMode`` enum. New callers should use
    :func:`compute_weight` directly with an explicit ``WeightMode``.
    """
    return compute_weight(
        reward_gap, confidence, WeightMode.CLIPPED_GAP,
        gap_threshold=cfg.gap_threshold,
        max_weight=cfg.max_weight,
        min_weight=cfg.min_weight,
        eps=cfg.eps,
    )


def compute_sample_weight(
    delta_u: float,
    confidence: float,
    gap_threshold: float,
    max_weight: float,
    min_weight: float = 0.0,
) -> float:
    """v0.3.10.1 — thin wrapper around :func:`compute_weight` with
    ``CLIPPED_GAP`` semantics (the historical default behaviour of
    this function).
    """
    return compute_weight(
        delta_u, confidence, WeightMode.CLIPPED_GAP,
        gap_threshold=gap_threshold,
        max_weight=max_weight,
        min_weight=min_weight,
    )


def snr_weight(
    delta_u: float,
    sigma_delta_u: float,
    max_weight: float,
    eps: float = 1e-8,
) -> float:
    """v0.3.10.1 — thin wrapper around :func:`compute_weight` with
    ``SNR`` semantics and ``confidence = 1.0``. Does NOT apply
    gap-threshold truncation (``gap_threshold = 0``); callers that need
    tie-truncation should use :func:`compute_weight` directly.
    """
    return compute_weight(
        delta_u, 1.0, WeightMode.SNR,
        gap_threshold=0.0,
        max_weight=max_weight,
        sigma_delta_u=sigma_delta_u,
        eps=eps,
    )


__all__ = [
    "InsufficientEffectiveWeight",
    "WeightConfig",
    "WeightMode",
    "ZeroWeightPolicy",
    "compute_sample_weight",
    "compute_weight",
    "compute_weights_batch",
    "require_effective_class_weight",
    "snr_weight",
    "utility_weight",
]
