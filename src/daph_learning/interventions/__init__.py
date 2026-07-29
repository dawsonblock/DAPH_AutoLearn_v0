"""v0.3.10 — causal intervention experiments (Sections 14-16).

This is the most important scientific addition after real policy
evaluation. Observation (correlation between ``h`` and ``ΔU``) does not
prove that modifying ``h`` in a direction changes policy utility.

Intervention experiment: given a candidate direction ``v``::

    h'(δ) = h + δ v

Test several ``δ`` (e.g. ``{-α, -α/2, 0, α/2, α}``) and measure route
probability, selected action, and downstream utility.

The experiment must distinguish:
    "direction correlates with preferred action"
from:
    "intervening in direction causes useful policy change"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..policy.abstention import choose_route
from ..policy.types import Route


@dataclass(frozen=True)
class InterventionResult:
    """Result of a single intervention at one dose (Section 14).

    v0.3.10.1 — adds ``evidence_level`` (Section 12) to distinguish
    mechanical sanity tests from synthetic/real causal evidence.
    """

    task_id: str
    direction_id: str
    strength: float  # δ
    baseline_route: str
    intervened_route: str
    baseline_utility: float
    intervened_utility: float
    baseline_prob_symbolic: float
    intervened_prob_symbolic: float
    # v0.3.10.1 — Section 12: evidence level tag.
    evidence_level: str = "unit_sanity"

    @property
    def utility_delta(self) -> float:
        return self.intervened_utility - self.baseline_utility

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "direction_id": self.direction_id,
            "strength": self.strength,
            "baseline_route": self.baseline_route,
            "intervened_route": self.intervened_route,
            "baseline_utility": self.baseline_utility,
            "intervened_utility": self.intervened_utility,
            "utility_delta": self.utility_delta,
            "baseline_prob_symbolic": self.baseline_prob_symbolic,
            "intervened_prob_symbolic": self.intervened_prob_symbolic,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class RealInterventionResult:
    """Result of a real-model residual-stream intervention (Section 13).

    For a loaded transformer model with a residual-stream hook installed
    at ``layer``, the baseline hidden state ``h`` is captured, then the
    model is run with ``h' = h + alpha * v`` for each ``alpha`` in the
    dose grid. This record captures the route, utility, KL, and clamp
    telemetry for one (task, vector, alpha) tuple.

    Attributes
    ----------
    task_id : str
    vector_id : str
    layer : int
    alpha : float
        Dose strength.
    baseline_route : str
    intervened_route : str
    baseline_utility : float
    intervened_utility : float
    utility_delta : float
        ``intervened_utility - baseline_utility``.
    route_score_before : float | None
        ``P(S | h)`` before intervention (None if not captured).
    route_score_after : float | None
        ``P(S | h + alpha * v)`` after intervention.
    neutral_kl : float | None
        KL divergence on a neutral prompt suite at this alpha
        (None if not measured).
    clamp_triggered : bool
        Whether a norm/cosine safety clamp fired during the intervention.
    evidence_level : str
        ``"REAL_MODEL_LATENT_INTERVENTION"`` for real-model interventions
        where the hidden state changed and downstream policy score changed.
        Use ``"REAL_MODEL_BEHAVIORAL_INTERVENTION"`` when the actual routed
        behavior changed, and ``"REAL_MODEL_UTILITY_INTERVENTION"`` when
        verified downstream utility changed (Section 32).
    """

    task_id: str
    vector_id: str
    layer: int
    alpha: float
    baseline_route: str
    intervened_route: str
    baseline_utility: float
    intervened_utility: float
    utility_delta: float
    route_score_before: float | None = None
    route_score_after: float | None = None
    neutral_kl: float | None = None
    clamp_triggered: bool = False
    evidence_level: str = "REAL_MODEL_LATENT_INTERVENTION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "vector_id": self.vector_id,
            "layer": self.layer,
            "alpha": self.alpha,
            "baseline_route": self.baseline_route,
            "intervened_route": self.intervened_route,
            "baseline_utility": self.baseline_utility,
            "intervened_utility": self.intervened_utility,
            "utility_delta": self.utility_delta,
            "route_score_before": self.route_score_before,
            "route_score_after": self.route_score_after,
            "neutral_kl": self.neutral_kl,
            "clamp_triggered": self.clamp_triggered,
            "evidence_level": self.evidence_level,
        }


# Type for a function that maps a (possibly intervened) activation to a
# P(S | h) probability and a utility.
PolicyProbFn = Callable[[np.ndarray], float]
UtilityFn = Callable[[np.ndarray, Route], float]


def run_intervention_experiment(
    task_id: str,
    direction_id: str,
    h: np.ndarray,
    v: np.ndarray,
    *,
    policy_prob_fn: PolicyProbFn,
    utility_fn: UtilityFn,
    alphas: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    confidence_threshold: float = 0.5,
) -> list[InterventionResult]:
    """Run a dose-response intervention experiment (Sections 14, 16).

    For each ``δ`` in ``alphas``::

        h'(δ) = h + δ v
        p'(δ) = policy_prob_fn(h'(δ))
        route'(δ) = choose_route(p'(δ), confidence_threshold)
        U'(δ) = utility_fn(h'(δ), route'(δ))

    The baseline (``δ=0``) is included. Results are returned for all
    doses.

    Parameters
    ----------
    policy_prob_fn : callable
        ``h -> P(S | h)`` (a float in [0, 1]).
    utility_fn : callable
        ``(h, route) -> utility`` (a float). The utility of executing the
        selected route under activation ``h``.
    alphas : sequence of float
        Dose strengths. Should include 0.0 for the baseline.
    confidence_threshold : float
        Abstention threshold for :func:`choose_route`.
    """
    h = np.asarray(h, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if h.shape != v.shape:
        raise ValueError("h and v must have the same shape")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
        raise ValueError("h and v must be finite")
    results: list[InterventionResult] = []
    # Baseline (δ=0).
    p0 = float(policy_prob_fn(h))
    route0 = choose_route(p0, confidence_threshold)
    u0 = float(utility_fn(h, route0))
    for delta in alphas:
        h_int = h + float(delta) * v
        p_int = float(policy_prob_fn(h_int))
        route_int = choose_route(p_int, confidence_threshold)
        u_int = float(utility_fn(h_int, route_int))
        results.append(InterventionResult(
            task_id=task_id,
            direction_id=direction_id,
            strength=float(delta),
            baseline_route=route0.value,
            intervened_route=route_int.value,
            baseline_utility=u0,
            intervened_utility=u_int,
            baseline_prob_symbolic=p0,
            intervened_prob_symbolic=p_int,
        ))
    return results


@dataclass(frozen=True)
class DirectionReversalResult:
    """Aggregate result of a direction reversal test (Section 15)."""

    direction_id: str
    n_tasks: int
    mean_prob_plus: float
    mean_prob_zero: float
    mean_prob_minus: float
    plus_greater_than_zero: int  # count where P(S|h+αv) > P(S|h)
    minus_less_than_zero: int    # count where P(S|h-αv) < P(S|h)
    reversal_consistent: bool    # both counts > n/2 (statistically measurable)
    p_value_sign_test: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_id": self.direction_id,
            "n_tasks": self.n_tasks,
            "mean_prob_plus": self.mean_prob_plus,
            "mean_prob_zero": self.mean_prob_zero,
            "mean_prob_minus": self.mean_prob_minus,
            "plus_greater_than_zero": self.plus_greater_than_zero,
            "minus_less_than_zero": self.minus_less_than_zero,
            "reversal_consistent": self.reversal_consistent,
            "p_value_sign_test": self.p_value_sign_test,
        }


def direction_reversal_test(
    direction_id: str,
    activations: np.ndarray,
    v: np.ndarray,
    *,
    policy_prob_fn: PolicyProbFn,
    alpha: float = 1.0,
) -> DirectionReversalResult:
    """Mandatory causal sanity check (Section 15).

    Evaluates ``+v``, ``0``, ``-v`` and checks that the policy score
    responds consistently::

        P(S | h + αv) > P(S | h)   (for a symbolic-preference vector)
        P(S | h - αv) < P(S | h)

    This need not hold for every task, but there must be statistically
    measurable aggregate sensitivity. If reversing ``v`` has no
    measurable effect, the direction is not a credible control vector.
    """
    activations = np.asarray(activations, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    n = len(activations)
    if n == 0:
        return DirectionReversalResult(
            direction_id=direction_id,
            n_tasks=0,
            mean_prob_plus=float("nan"),
            mean_prob_zero=float("nan"),
            mean_prob_minus=float("nan"),
            plus_greater_than_zero=0,
            minus_less_than_zero=0,
            reversal_consistent=False,
            p_value_sign_test=1.0,
        )
    probs_plus = np.array([
        policy_prob_fn(h + alpha * v) for h in activations
    ])
    probs_zero = np.array([policy_prob_fn(h) for h in activations])
    probs_minus = np.array([
        policy_prob_fn(h - alpha * v) for h in activations
    ])
    plus_gt = int(np.sum(probs_plus > probs_zero))
    minus_lt = int(np.sum(probs_minus < probs_zero))
    # Sign test: under the null (no effect), plus_gt ~ Binomial(n, 0.5).
    # Use a simple exact binomial p-value without scipy.
    p_value = _binomial_two_sided(plus_gt, n, 0.5)
    reversal_consistent = (
        plus_gt > n / 2
        and minus_lt > n / 2
        and p_value < 0.05
    )
    return DirectionReversalResult(
        direction_id=direction_id,
        n_tasks=n,
        mean_prob_plus=float(probs_plus.mean()),
        mean_prob_zero=float(probs_zero.mean()),
        mean_prob_minus=float(probs_minus.mean()),
        plus_greater_than_zero=plus_gt,
        minus_less_than_zero=minus_lt,
        reversal_consistent=reversal_consistent,
        p_value_sign_test=p_value,
    )


def _binomial_two_sided(k: int, n: int, p: float) -> float:
    """Exact two-sided binomial test p-value (no scipy dependency)."""
    if n <= 0:
        return 1.0
    from math import comb
    # Two-sided: sum of probabilities <= P(X=k).
    pk = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if pi <= pk + 1e-15:
            total += pi
    return min(total, 1.0)


def dose_response_summary(
    results: Sequence[InterventionResult],
) -> dict[str, Any]:
    """Summarize a dose-response experiment (Section 16).

    Looks for monotonic useful range, saturation, reversal, or
    destructive high-alpha regime.
    """
    results = list(results)
    if not results:
        return {"n": 0}
    # Sort by strength.
    results.sort(key=lambda r: r.strength)
    strengths = [r.strength for r in results]
    utilities = [r.intervened_utility for r in results]
    probs = [r.intervened_prob_symbolic for r in results]
    baseline_u = next(
        (r.baseline_utility for r in results
         if r.strength == 0.0),
        utilities[0],
    )
    # Monotonicity check on utility.
    monotonic = (
        all(utilities[i] <= utilities[i + 1] + 1e-9
            for i in range(len(utilities) - 1))
        or all(utilities[i] >= utilities[i + 1] - 1e-9
               for i in range(len(utilities) - 1))
    )
    # Reversal: utility goes up then down (or vice versa).
    max_idx = int(np.argmax(utilities))
    min_idx = int(np.argmin(utilities))
    reversal = (
        max_idx not in (0, len(utilities) - 1)
        or min_idx not in (0, len(utilities) - 1)
    )
    # Destructive high-alpha: utility at max |alpha| is below baseline.
    max_alpha_idx = int(np.argmax(np.abs(strengths)))
    destructive = utilities[max_alpha_idx] < baseline_u - 1e-9
    return {
        "n": len(results),
        "strengths": strengths,
        "utilities": utilities,
        "probabilities": probs,
        "baseline_utility": baseline_u,
        "monotonic": monotonic,
        "reversal": reversal,
        "destructive_high_alpha": destructive,
        "max_utility": float(max(utilities)),
        "min_utility": float(min(utilities)),
    }


__all__ = [
    "DirectionReversalResult",
    "InterventionResult",
    "PolicyProbFn",
    "RealInterventionResult",
    "UtilityFn",
    "direction_reversal_test",
    "dose_response_summary",
    "run_intervention_experiment",
]
