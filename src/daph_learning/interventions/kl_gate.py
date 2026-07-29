"""v0.3.10 — KL / capability-preservation promotion gates (Section 17).

A steering intervention may improve routing while damaging unrelated
behavior. Candidate promotion must require::

    utility gain >= minimum_gain
    AND  neutral_KL <= KL_budget
    AND  protected capability regression <= allowed_drop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PromotionConstraints:
    """Promotion safety constraints (Section 17).

    A candidate is promoted only when ALL of these hold:
    - ``mean_utility_gain >= min_mean_utility_gain``
    - ``mean_neutral_kl <= max_neutral_kl``
    - ``capability_drop <= max_capability_drop`` for every protected capability
    - ``n_samples >= min_samples``
    """

    min_mean_utility_gain: float = 0.01
    max_neutral_kl: float = 0.03
    max_capability_drop: float = 0.02
    min_samples: int = 100


@dataclass(frozen=True)
class PromotionGateResult:
    """Result of the KL/capability promotion gate."""

    promoted: bool
    reason: str
    mean_utility_gain: float
    mean_neutral_kl: float
    capability_drops: dict[str, float]
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "mean_utility_gain": self.mean_utility_gain,
            "mean_neutral_kl": self.mean_neutral_kl,
            "capability_drops": dict(self.capability_drops),
            "n_samples": self.n_samples,
        }


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """``KL(p || q)`` for discrete distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))))


def mean_kl_neutral(
    base_distributions: Sequence[np.ndarray],
    steered_distributions: Sequence[np.ndarray],
) -> float:
    """Mean KL divergence on neutral prompts (Section 17).

    Parameters
    ----------
    base_distributions : sequence of array-like
        Token distributions from the base (unsteered) model on neutral
        prompts.
    steered_distributions : sequence of array-like
        Token distributions from the steered model on the same neutral
        prompts.
    """
    if len(base_distributions) != len(steered_distributions):
        raise ValueError("base and steered distribution lists must match")
    if not base_distributions:
        raise ValueError("empty distribution lists")
    kls = [
        kl_divergence(np.asarray(b, dtype=np.float64), np.asarray(s, dtype=np.float64))
        for b, s in zip(base_distributions, steered_distributions)
    ]
    return float(np.mean(kls))


def evaluate_kl_capability_gate(
    *,
    mean_utility_gain: float,
    neutral_kl: float,
    capability_drops: dict[str, float] | None = None,
    n_samples: int,
    constraints: PromotionConstraints,
) -> PromotionGateResult:
    """Evaluate the KL/capability promotion gate (Section 17).

    Returns a :class:`PromotionGateResult`. Promotion requires all
    constraints to pass.
    """
    cap_drops = dict(capability_drops or {})
    if n_samples < constraints.min_samples:
        return PromotionGateResult(
            promoted=False,
            reason=f"n_samples {n_samples} < min_samples {constraints.min_samples}",
            mean_utility_gain=mean_utility_gain,
            mean_neutral_kl=neutral_kl,
            capability_drops=cap_drops,
            n_samples=n_samples,
        )
    if mean_utility_gain < constraints.min_mean_utility_gain:
        return PromotionGateResult(
            promoted=False,
            reason=f"utility gain {mean_utility_gain:.4f} < min {constraints.min_mean_utility_gain:.4f}",
            mean_utility_gain=mean_utility_gain,
            mean_neutral_kl=neutral_kl,
            capability_drops=cap_drops,
            n_samples=n_samples,
        )
    if neutral_kl is None or neutral_kl > constraints.max_neutral_kl:
        kl_str = "N/A" if neutral_kl is None else f"{neutral_kl:.4f}"
        return PromotionGateResult(
            promoted=False,
            reason=f"neutral KL {kl_str} > max {constraints.max_neutral_kl:.4f}",
            mean_utility_gain=mean_utility_gain,
            mean_neutral_kl=neutral_kl,
            capability_drops=cap_drops,
            n_samples=n_samples,
        )
    for cap, drop in cap_drops.items():
        if drop > constraints.max_capability_drop:
            return PromotionGateResult(
                promoted=False,
                reason=f"capability {cap!r} drop {drop:.4f} > max {constraints.max_capability_drop:.4f}",
                mean_utility_gain=mean_utility_gain,
                mean_neutral_kl=neutral_kl,
                capability_drops=cap_drops,
                n_samples=n_samples,
            )
    return PromotionGateResult(
        promoted=True,
        reason="all KL/capability constraints satisfied",
        mean_utility_gain=mean_utility_gain,
        mean_neutral_kl=neutral_kl,
        capability_drops=cap_drops,
        n_samples=n_samples,
    )


__all__ = [
    "PromotionConstraints",
    "PromotionGateResult",
    "evaluate_kl_capability_gate",
    "kl_divergence",
    "mean_kl_neutral",
]
