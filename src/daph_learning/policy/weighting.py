"""v0.3.10 — utility-weighted sample weighting (Sections 3, 7, 9).

This module implements the weighting functions that connect the
counterfactual utility gap ``|ΔU|`` and outcome confidence to per-example
training weights.

Key design decisions (Section 7):
- Near-ties (``|ΔU| <= gap_threshold``) are **truncated to zero weight**,
  not floored. A positive floor would accidentally make near-ties matter
  again, defeating the purpose of utility-weighting.
- A numerical stability floor is applied **only after** gap filtering, and
  only to avoid division-by-zero in downstream normalization — it is NOT
  a statistical minimum importance.
- Two explicit modes: ``gap`` (utility-gap × confidence) and ``snr``
  (signal-to-noise, ``|ΔU| / (σ_ΔU + ε)``), per Section 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class WeightMode(str, Enum):
    """Sample weighting mode."""

    GAP = "gap"   # |ΔU| × confidence (default, Section 3)
    SNR = "snr"   # |ΔU| / (σ_ΔU + ε) (Section 9, uncertainty-aware)


@dataclass(frozen=True)
class WeightConfig:
    """Configuration for utility-weighted sample weighting.

    Attributes
    ----------
    min_weight : float
        Numerical stability floor applied AFTER gap filtering. This is NOT
        a statistical minimum importance — it only prevents division-by-zero
        in downstream normalization. Set to 0.0 by default (Section 7).
    max_weight : float
        Maximum weight (clipping).
    eps : float
        Numerical epsilon for division safety.
    gap_threshold : float
        Tasks with ``|ΔU| <= gap_threshold`` get zero weight (tie truncation,
        Section 7 Mode A). Must be >= 0.
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


def utility_weight(
    reward_gap: float,
    confidence: float,
    cfg: WeightConfig,
) -> float:
    """Utility-weighted sample weight:
    ``clip(|ΔU| × confidence, w_min, w_max)``.

    Near-ties (``|ΔU| <= gap_threshold``) are truncated to zero BEFORE
    clipping (Section 7 Mode A). The ``min_weight`` floor is a numerical
    stability floor applied after gap filtering, not a statistical minimum.

    Parameters
    ----------
    reward_gap : float
        ``ΔU`` (utility difference). Must be finite.
    confidence : float
        Combined outcome confidence in ``[0, 1]``.
    cfg : WeightConfig
        Weighting configuration.

    Returns
    -------
    float
        The clipped sample weight.
    """
    if not np.isfinite(reward_gap):
        raise ValueError("reward_gap must be finite")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    # Section 7 Mode A: truncate ties to zero weight.
    if abs(reward_gap) <= cfg.gap_threshold:
        return 0.0
    raw = abs(reward_gap) * confidence
    # Section 7 Mode B: numerical stability floor only after filtering.
    weighted = max(raw, cfg.min_weight)
    return float(min(weighted, cfg.max_weight))


def compute_sample_weight(
    delta_u: float,
    confidence: float,
    gap_threshold: float,
    max_weight: float,
    min_weight: float = 0.0,
) -> float:
    """Standalone sample weight with gap-threshold tie truncation (Section 7).

    Returns 0.0 if ``|ΔU| <= gap_threshold``, else
    ``max(min_weight, min(|ΔU| × confidence, max_weight))``.

    The ``min_weight`` floor is a numerical stability floor applied AFTER
    gap filtering, consistent with :func:`utility_weight`.
    """
    if abs(delta_u) <= gap_threshold:
        return 0.0
    weighted = max(abs(delta_u) * confidence, min_weight)
    return min(weighted, max_weight)


def snr_weight(
    delta_u: float,
    sigma_delta_u: float,
    max_weight: float,
    eps: float = 1e-8,
) -> float:
    """Signal-to-noise weighting: ``min(|ΔU| / (σ_ΔU + ε), max_weight)``
    (Section 9).

    Use this when utility estimates have meaningful variance. Do NOT enable
    by default unless the uncertainty estimate is meaningful.

    Parameters
    ----------
    delta_u : float
        Utility difference.
    sigma_delta_u : float
        Standard deviation of the utility estimate. Must be >= 0.
    max_weight : float
        Maximum weight (clipping).
    eps : float
        Numerical epsilon.
    """
    if sigma_delta_u < 0:
        raise ValueError("sigma_delta_u must be >= 0")
    raw = abs(delta_u) / (sigma_delta_u + eps)
    return min(raw, max_weight)


__all__ = [
    "WeightConfig",
    "WeightMode",
    "compute_sample_weight",
    "snr_weight",
    "utility_weight",
]
