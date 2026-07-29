"""v0.3.10.1-alpha — capability regression gates (Section 24) and
comparative gates (Section 11 / G14).

Aggregate metrics can hide local failure. This module provides:

* :class:`CapabilityGateConfig` — per-family utility/accuracy drop
  thresholds. A candidate is rejected if any protected family regresses
  beyond the threshold (Section 24).
* :func:`capability_gate` — evaluate a candidate vs incumbent on
  per-family slices and reject regressions.
* :func:`comparative_gate` — literally compare two methods by computing
  BOTH sides and asserting the expected ordering (Section 11 / G14).
  Never mark a comparative gate PASS without computing both sides.

Both gates return a structured :class:`GateResult` that records both
measurements so the release artifact is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CapabilityGateConfig:
    """Configuration for capability regression gates (Section 24).

    Attributes
    ----------
    max_family_utility_drop : float
        Maximum allowed drop in mean per-family utility
        (``U_incumbent - U_candidate``). Default 0.02.
    max_family_accuracy_drop : float
        Maximum allowed drop in per-family routing accuracy. Default 0.03.
    min_family_samples : int
        Minimum samples in a family for the gate to apply. Families
        with fewer samples are skipped (not enough signal). Default 20.
    """

    max_family_utility_drop: float = 0.02
    max_family_accuracy_drop: float = 0.03
    min_family_samples: int = 20


@dataclass
class FamilyResult:
    """Per-family comparison result."""

    family: str
    n_samples: int
    incumbent_mean_utility: float
    candidate_mean_utility: float
    utility_delta: float  # candidate - incumbent
    incumbent_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    utility_drop_exceeds: bool
    accuracy_drop_exceeds: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_samples": self.n_samples,
            "incumbent_mean_utility": self.incumbent_mean_utility,
            "candidate_mean_utility": self.candidate_mean_utility,
            "utility_delta": self.utility_delta,
            "incumbent_accuracy": self.incumbent_accuracy,
            "candidate_accuracy": self.candidate_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "utility_drop_exceeds": self.utility_drop_exceeds,
            "accuracy_drop_exceeds": self.accuracy_drop_exceeds,
        }


@dataclass
class GateResult:
    """Result of a capability or comparative gate.

    Attributes
    ----------
    passed : bool
    reason : str
        Human-readable pass/fail reason.
    families : list[FamilyResult]
        Per-family breakdown (capability gate).
    measurements : dict
        Both-side measurements for auditability (comparative gate).
    """

    passed: bool
    reason: str
    families: list[FamilyResult] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "families": [f.to_dict() for f in self.families],
            "measurements": self.measurements,
        }


def capability_gate(
    candidate_utilities: np.ndarray,
    incumbent_utilities: np.ndarray,
    candidate_routes: Sequence[str],
    incumbent_routes: Sequence[str],
    correct_routes: Sequence[str],
    families: Sequence[str],
    *,
    config: CapabilityGateConfig | None = None,
) -> GateResult:
    """Capability regression gate (Section 24).

    For each family with >= ``min_family_samples`` samples, compare
    candidate vs incumbent on mean utility and routing accuracy. Reject
    the candidate if any protected family regresses beyond the
    threshold.

    Parameters
    ----------
    candidate_utilities, incumbent_utilities : np.ndarray  shape ``[N]``
    candidate_routes, incumbent_routes : sequence of str  shape ``[N]``
    correct_routes : sequence of str  shape ``[N]``
        The correct route per task (for accuracy).
    families : sequence of str  shape ``[N]``
        The family/capability_id per task.
    config : CapabilityGateConfig | None
    """
    cfg = config or CapabilityGateConfig()
    cu = np.asarray(candidate_utilities, dtype=np.float64)
    iu = np.asarray(incumbent_utilities, dtype=np.float64)
    n = len(cu)
    if len(iu) != n or len(candidate_routes) != n or len(incumbent_routes) != n:
        raise ValueError("all inputs must have the same length")
    if len(correct_routes) != n or len(families) != n:
        raise ValueError("correct_routes and families must have length N")
    fams = np.asarray(families)
    unique_fams = sorted(set(fams))
    family_results: list[FamilyResult] = []
    failures: list[str] = []
    for fam in unique_fams:
        mask = fams == fam
        n_fam = int(mask.sum())
        if n_fam < cfg.min_family_samples:
            continue
        c_u = float(cu[mask].mean())
        i_u = float(iu[mask].mean())
        u_delta = c_u - i_u  # positive => candidate better
        # Accuracy: fraction where route == correct_route.
        c_acc = float(np.mean(np.array(candidate_routes)[mask] == np.array(correct_routes)[mask]))
        i_acc = float(np.mean(np.array(incumbent_routes)[mask] == np.array(correct_routes)[mask]))
        a_delta = c_acc - i_acc
        u_drop = -u_delta > cfg.max_family_utility_drop  # incumbent - candidate > threshold
        a_drop = -a_delta > cfg.max_family_accuracy_drop
        family_results.append(FamilyResult(
            family=fam, n_samples=n_fam,
            incumbent_mean_utility=i_u, candidate_mean_utility=c_u,
            utility_delta=u_delta,
            incumbent_accuracy=i_acc, candidate_accuracy=c_acc,
            accuracy_delta=a_delta,
            utility_drop_exceeds=u_drop,
            accuracy_drop_exceeds=a_drop,
        ))
        if u_drop:
            failures.append(
                f"family {fam!r}: utility drop {i_u - c_u:.4f} > "
                f"threshold {cfg.max_family_utility_drop}")
        if a_drop:
            failures.append(
                f"family {fam!r}: accuracy drop {i_acc - c_acc:.4f} > "
                f"threshold {cfg.max_family_accuracy_drop}")
    passed = not failures
    reason = "all families within thresholds" if passed else "; ".join(failures)
    return GateResult(
        passed=passed, reason=reason,
        families=family_results,
        measurements={"n_families_evaluated": len(family_results)},
    )


def comparative_gate(
    candidate_result: Mapping[str, Any],
    incumbent_result: Mapping[str, Any],
    *,
    metric: str = "mean_regret",
    expected_ordering: str = "candidate_lower",
    tolerance: float = 0.0,
) -> GateResult:
    """Comparative gate (Section 11 / G14).

    Literally compare two methods by computing BOTH sides and asserting
    the expected ordering. Never mark a comparative gate PASS without
    computing both sides.

    Parameters
    ----------
    candidate_result, incumbent_result : mapping
        Result dicts that must contain ``metric`` as a key (e.g.
        ``{"mean_regret": 0.04, ...}``).
    metric : str
        The metric to compare (e.g. ``"mean_regret"``,
        ``"mean_utility"``).
    expected_ordering : str
        ``"candidate_lower"`` (candidate <= incumbent + tolerance,
        e.g. for regret) or ``"candidate_higher"`` (candidate >=
        incumbent - tolerance, e.g. for utility).
    tolerance : float
        Allowed tolerance. For ``candidate_lower``: pass if
        ``candidate <= incumbent + tolerance``.

    Returns
    -------
    GateResult
        With ``measurements`` recording both sides for auditability.
    """
    if metric not in candidate_result:
        raise ValueError(
            f"candidate_result is missing metric {metric!r}; cannot "
            f"compute comparative gate without both sides")
    if metric not in incumbent_result:
        raise ValueError(
            f"incumbent_result is missing metric {metric!r}; cannot "
            f"compute comparative gate without both sides")
    c_val = float(candidate_result[metric])
    i_val = float(incumbent_result[metric])
    measurements = {
        "metric": metric,
        "candidate_value": c_val,
        "incumbent_value": i_val,
        "tolerance": tolerance,
        "expected_ordering": expected_ordering,
    }
    if expected_ordering == "candidate_lower":
        passed = c_val <= i_val + tolerance
        reason = (
            f"candidate {metric}={c_val:.6f} <= incumbent {i_val:.6f} "
            f"+ tol {tolerance:.6f}" if passed
            else f"candidate {metric}={c_val:.6f} > incumbent "
                 f"{i_val:.6f} + tol {tolerance:.6f}")
    elif expected_ordering == "candidate_higher":
        passed = c_val >= i_val - tolerance
        reason = (
            f"candidate {metric}={c_val:.6f} >= incumbent {i_val:.6f} "
            f"- tol {tolerance:.6f}" if passed
            else f"candidate {metric}={c_val:.6f} < incumbent "
                 f"{i_val:.6f} - tol {tolerance:.6f}")
    else:
        raise ValueError(
            f"unknown expected_ordering {expected_ordering!r}; expected "
            f"'candidate_lower' or 'candidate_higher'")
    return GateResult(passed=passed, reason=reason, measurements=measurements)


__all__ = [
    "CapabilityGateConfig",
    "FamilyResult",
    "GateResult",
    "capability_gate",
    "comparative_gate",
]
