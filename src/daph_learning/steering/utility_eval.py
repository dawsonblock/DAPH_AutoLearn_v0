"""v0.3.10.3.2-alpha — steering utility evaluation (Sections 21-25).

THE SECOND MOST IMPORTANT CHANGE.

The previous build showed that pushing harder in the learned symbolic
direction can raise ``P(symbolic | +v)`` while LOWERING utility at the
strongest dose. That is useful evidence. This module preserves that
negative result and evaluates steering strictly through::

    ΔU(α) = U(π(h + α v)) - U(π(h))

NOT through ``max_alpha P(symbolic | h + α v)``.

For every intervention strength α, the required chain is:

    intervention -> route change -> backend execution ->
    verification -> utility/regret change

A steering direction is practically useful only when beneficial route
flips dominate harmful flips (Section 23).

This module is policy/backend-agnostic: it takes a policy-probability
function, backend executors, a verifier, and the canonical utility, so
it works with both the simulated path (unit tests) and the real Qwen
path (integration gates G29-G31).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.types import BackendOutcome
from daph_learning.policy.utility import backend_utility


@dataclass(frozen=True)
class SteeringTaskRecord:
    """One task's steering evaluation across the alpha grid."""
    task_id: str
    family_id: str
    # Per-alpha outcomes (keyed by alpha).
    routes: dict[float, str]
    utilities: dict[float, float]
    regrets: dict[float, float]
    verified_correct: dict[float, bool]


@dataclass
class AlphaGridResult:
    """Section 22 / 45: per-alpha aggregate metrics."""
    alpha: float
    mean_utility: float
    mean_regret: float
    mean_p_symbolic: float
    flip_rate: float
    beneficial_flip_fraction: float
    harmful_flip_fraction: float
    neutral_flip_fraction: float
    n_tasks: int
    n_flips: int
    n_beneficial: int
    n_harmful: int
    n_neutral: int


@dataclass
class SteeringUtilityReport:
    """Full steering utility evaluation (Sections 21-25, 45)."""
    alphas: list[float]
    per_alpha: list[AlphaGridResult]
    per_task: list[SteeringTaskRecord]
    oracle_alpha_utility: float  # per-task oracle (DEV only, Section 24)
    best_global_alpha: float
    best_global_alpha_utility: float
    baseline_utility: float  # alpha=0
    random_control_p_emp: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alphas": self.alphas,
            "per_alpha": [
                {"alpha": r.alpha, "mean_utility": r.mean_utility,
                 "mean_regret": r.mean_regret,
                 "mean_p_symbolic": r.mean_p_symbolic,
                 "flip_rate": r.flip_rate,
                 "beneficial_flip_fraction": r.beneficial_flip_fraction,
                 "harmful_flip_fraction": r.harmful_flip_fraction,
                 "neutral_flip_fraction": r.neutral_flip_fraction,
                 "n_tasks": r.n_tasks, "n_flips": r.n_flips,
                 "n_beneficial": r.n_beneficial,
                 "n_harmful": r.n_harmful,
                 "n_neutral": r.n_neutral}
                for r in self.per_alpha],
            "oracle_alpha_utility": self.oracle_alpha_utility,
            "best_global_alpha": self.best_global_alpha,
            "best_global_alpha_utility": self.best_global_alpha_utility,
            "baseline_utility": self.baseline_utility,
            "random_control_p_emp": self.random_control_p_emp,
            "notes": self.notes,
        }


def _route_from_prob(p_symbolic: float, confidence_threshold: float) -> str:
    if p_symbolic >= confidence_threshold:
        return "symbolic"
    if p_symbolic <= 1.0 - confidence_threshold:
        return "llm"
    return "abstain"


def evaluate_steering_utility(
    tasks: list[Mapping[str, Any]],
    *,
    policy_proba_fn: Callable[[Mapping[str, Any], float], float],
    execute_symbolic_fn: Callable[[Mapping[str, Any]], tuple[BackendOutcome, str | None]],
    execute_llm_fn: Callable[[Mapping[str, Any]], str | None],
    verify_fn: Callable[[Mapping[str, Any], str | None], Any],
    config: ExperimentConfig,
    alphas: Sequence[float] | None = None,
    oracle_utilities: dict[str, dict[float, float]] | None = None,
) -> SteeringUtilityReport:
    """Evaluate steering by verified downstream utility (Section 21-22).

    Parameters
    ----------
    tasks : list of task mappings
    policy_proba_fn : callable(task, alpha) -> P(symbolic | h + alpha v)
        The steered policy probability. alpha=0 is the baseline.
    execute_symbolic_fn : callable(task) -> (BackendOutcome, output_text)
    execute_llm_fn : callable(task) -> output_text
    verify_fn : callable(task, output_text) -> VerificationResult
    config : ExperimentConfig
    alphas : sequence of float
        Defaults to config.intervention_alpha_grid.
    oracle_utilities : optional precomputed per-task per-alpha utilities
        (for the oracle alpha probe, Section 24).
    """
    alpha_grid = list(alphas) if alphas is not None else list(config.intervention_alpha_grid)
    cfg = config
    per_task: list[SteeringTaskRecord] = []
    # Pre-execute both backends once (outcomes don't depend on alpha;
    # only the route choice does). This is the counterfactual setup.
    sym_outcomes: dict[str, tuple[BackendOutcome, str | None]] = {}
    llm_texts: dict[str, str | None] = {}
    sym_verified: dict[str, Any] = {}
    llm_verified: dict[str, Any] = {}
    for t in tasks:
        tid = str(t.get("task_id", ""))
        sym_outcomes[tid] = execute_symbolic_fn(t)
        llm_texts[tid] = execute_llm_fn(t)
        sym_verified[tid] = verify_fn(t, sym_outcomes[tid][1])
        llm_verified[tid] = verify_fn(t, llm_texts[tid])
    # Build verified BackendOutcomes.
    from daph_learning.execution.real_backends import verify_arithmetic as _default_verify
    # Use the helper from crossover_benchmark to apply verification.
    from daph_learning.data.crossover_benchmark import _apply_verification, _verifier_status

    def _verified_outcome(tid: str, backend: str) -> BackendOutcome:
        if backend == "symbolic":
            base, _ = sym_outcomes[tid]
            vr = sym_verified[tid]
            return _apply_verification(base, vr)
        else:
            text = llm_texts[tid]
            vr = llm_verified[tid]
            return BackendOutcome(
                task_id=tid, backend="llm", available=True, executed=True,
                execution_success=text is not None, output_text=text,
                output_hash=None, verifier_status=_verifier_status(vr),
                correct=bool(getattr(vr, "verified_correct", None) is True),
                quality=1.0 if getattr(vr, "verified_correct", None) is True else 0.0,
                latency_sec=0.02, normalized_cost=0.001, risk=0.0,
                verifier_confidence=float(getattr(vr, "confidence", 0.0)),
                failure_reason=getattr(vr, "failure_reason", None),
            )

    # For each task, compute route + utility + regret per alpha.
    # Regret = U(optimal route) - U(chosen route), where optimal is the
    # better of the two verified backends.
    for t in tasks:
        tid = str(t.get("task_id", ""))
        family = t.get("metadata", {}).get("family_id", "")
        sym_o = _verified_outcome(tid, "symbolic")
        llm_o = _verified_outcome(tid, "llm")
        u_sym = backend_utility(sym_o, cfg)
        u_llm = backend_utility(llm_o, cfg)
        u_optimal = max(u_sym, u_llm)
        routes: dict[float, str] = {}
        utilities: dict[float, float] = {}
        regrets: dict[float, float] = {}
        verified: dict[float, bool] = {}
        for alpha in alpha_grid:
            p = float(policy_proba_fn(t, alpha))
            route = _route_from_prob(p, cfg.confidence_threshold)
            if route == "abstain":
                u = 0.0
                vc = False
            elif route == "symbolic":
                u = u_sym
                vc = sym_o.verified_correct is True
            else:
                u = u_llm
                vc = llm_o.verified_correct is True
            routes[alpha] = route
            utilities[alpha] = u
            regrets[alpha] = u_optimal - u
            verified[alpha] = vc
        per_task.append(SteeringTaskRecord(
            task_id=tid, family_id=family, routes=routes,
            utilities=utilities, regrets=regrets, verified_correct=verified))

    # Aggregate per alpha (Section 22, 23, 45).
    per_alpha_results: list[AlphaGridResult] = []
    baseline_route_by_task = {r.task_id: r.routes[0.0] for r in per_task
                              if 0.0 in r.routes}
    for alpha in alpha_grid:
        us = [r.utilities[alpha] for r in per_task if alpha in r.utilities]
        rs = [r.regrets[alpha] for r in per_task if alpha in r.regrets]
        n = len(us)
        n_flips = n_ben = n_harm = n_neut = 0
        for r in per_task:
            if alpha not in r.routes or r.task_id not in baseline_route_by_task:
                continue
            base_route = baseline_route_by_task[r.task_id]
            alpha_route = r.routes[alpha]
            if alpha_route != base_route:
                n_flips += 1
                du = r.utilities[alpha] - r.utilities.get(0.0, 0.0)
                if du > 1e-9:
                    n_ben += 1
                elif du < -1e-9:
                    n_harm += 1
                else:
                    n_neut += 1
        flip_rate = n_flips / n if n else 0.0
        per_alpha_results.append(AlphaGridResult(
            alpha=alpha,
            mean_utility=float(np.mean(us)) if us else 0.0,
            mean_regret=float(np.mean(rs)) if rs else 0.0,
            mean_p_symbolic=float(np.mean([
                float(policy_proba_fn(t, alpha)) for t in tasks])),
            flip_rate=flip_rate,
            beneficial_flip_fraction=n_ben / n_flips if n_flips else 0.0,
            harmful_flip_fraction=n_harm / n_flips if n_flips else 0.0,
            neutral_flip_fraction=n_neut / n_flips if n_flips else 0.0,
            n_tasks=n, n_flips=n_flips, n_beneficial=n_ben,
            n_harmful=n_harm, n_neutral=n_neut))

    # Best global alpha (Section 24).
    best_alpha = 0.0
    best_u = 0.0
    for r in per_alpha_results:
        if r.mean_utility > best_u:
            best_u = r.mean_utility
            best_alpha = r.alpha
    baseline_u = next((r.mean_utility for r in per_alpha_results if r.alpha == 0.0), 0.0)
    # Oracle alpha (per-task best) — DEV only diagnostic (Section 24).
    oracle_u = 0.0
    for r in per_task:
        best_task_u = max(r.utilities.values()) if r.utilities else 0.0
        oracle_u += best_task_u
    oracle_u = oracle_u / len(per_task) if per_task else 0.0

    return SteeringUtilityReport(
        alphas=alpha_grid,
        per_alpha=per_alpha_results,
        per_task=per_task,
        oracle_alpha_utility=oracle_u,
        best_global_alpha=best_alpha,
        best_global_alpha_utility=best_u,
        baseline_utility=baseline_u,
    )


def random_control_p_emp(
    learned_gain: float,
    random_gains: Sequence[float],
) -> float:
    """Section 25: empirical p-value for steering utility gain against
    matched random directions.

    ``p_emp = (1 + count(random_gain >= learned_gain)) / (N + 1)``
    """
    n = len(random_gains)
    count = sum(1 for g in random_gains if g >= learned_gain)
    return (1 + count) / (n + 1)


__all__ = [
    "AlphaGridResult",
    "SteeringTaskRecord",
    "SteeringUtilityReport",
    "evaluate_steering_utility",
    "random_control_p_emp",
]
