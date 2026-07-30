"""v0.3.10.3.2-alpha — canonical backend utility function (Section 3).

This is the ONE place where ``U_b`` is computed from a
:class:`~daph_learning.policy.types.BackendOutcome` and an
:class:`~daph_learning.policy.config.ExperimentConfig`.

All other code must call :func:`backend_utility` (or
:func:`utility_for_route`) instead of duplicating the formula. This
prevents drift between the experience-construction path and the
evaluation path. The same utility definition is used by training, dev
evaluation, calibration, final evaluation, promotion, steering
experiments, random controls, and hand-router baselines.
"""

from __future__ import annotations

from .config import ExperimentConfig
from .types import BackendOutcome, CounterfactualExperience, Route


def utility_formula(
    *,
    quality: float,
    latency_ms: float,
    compute_cost: float,
    risk: float,
    quality_weight: float,
    lambda_time: float,
    lambda_compute: float,
    lambda_risk: float,
    time_reference_ms: float,
    compute_reference: float,
) -> float:
    """The single implementation of the canonical utility ``U_b`` (Section 3).

    ``U_b = w_q * quality - λ_t*(latency_ms/t_ref) - λ_c*(cost/c_ref) - λ_r*risk``

    Both :func:`backend_utility` (the ``BackendOutcome`` path) and
    ``daph_learning.autolearn.counterfactual.compute_backend_utility``
    (the ``BackendExecutionRecord`` path) delegate to this function so
    the formula cannot drift between the experience-construction and
    evaluation paths.
    """
    time_term = lambda_time * (latency_ms / time_reference_ms)
    compute_term = lambda_compute * (compute_cost / compute_reference)
    risk_term = lambda_risk * risk
    return quality_weight * quality - time_term - compute_term - risk_term


def backend_utility(outcome: BackendOutcome, cfg: ExperimentConfig) -> float:
    """Compute ``U_b`` from a :class:`BackendOutcome` and config.

    This is the canonical ``BackendOutcome``-typed entry point. It
    delegates to :func:`utility_formula` — do not duplicate the formula
    elsewhere.
    """
    return utility_formula(
        quality=outcome.quality,
        latency_ms=outcome.latency_sec * 1000.0,
        compute_cost=outcome.normalized_cost,
        risk=outcome.risk,
        quality_weight=cfg.quality_weight,
        lambda_time=cfg.lambda_time,
        lambda_compute=cfg.lambda_compute,
        lambda_risk=cfg.lambda_risk,
        time_reference_ms=cfg.time_reference_ms,
        compute_reference=cfg.compute_reference,
    )


def utility_for_route(
    experience: CounterfactualExperience,
    route: Route | str,
    config: ExperimentConfig,
) -> float:
    """Look up the canonical utility for a route on a given experience
    (Section 3).

    Centralizes route-utility lookup so that training, dev evaluation,
    calibration, final evaluation, promotion, steering experiments,
    random controls, and hand-router baselines all use the same
    :func:`backend_utility` definition rather than ad-hoc
    ``cfg.quality_weight * quality`` shortcuts.

    Parameters
    ----------
    experience : CounterfactualExperience
        The pre-computed counterfactual experience (both backend
        outcomes already captured + verified).
    route : Route | str
        ``"symbolic"``, ``"llm"``, or ``"abstain"``.
    config : ExperimentConfig
        Canonical config carrying the utility weights.

    Returns
    -------
    float
        ``U_b`` for the selected backend, or ``0.0`` for abstain
        (abstention earns no quality reward by definition).
    """
    if isinstance(route, Route):
        route = route.value
    if route == "abstain":
        return 0.0
    if route == "symbolic":
        return backend_utility(experience.symbolic, config)
    if route == "llm":
        return backend_utility(experience.llm, config)
    raise ValueError(f"unknown route {route!r}; expected symbolic|llm|abstain")


__all__ = ["backend_utility", "utility_for_route", "utility_formula"]

