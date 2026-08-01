"""v0.3.9 (V039-001/003/005/006) — counterfactual backend execution, frozen
utility, and the explicit ΔU learning target.

This module implements Item 2 of the recommended repair sequence:

* **Execute both backends** for every feasible task (counterfactual
  execution), independently verify each, and never credit an unexecuted
  backend.
* **Frozen utility** per backend::

      U_b = Q_b − λ_t·T_b − λ_c·C_b − λ_r·R_b

  where ``Q_b`` is the verified-quality reward, ``T_b`` / ``C_b`` are
  normalized latency / compute, and ``R_b`` is the execution-risk penalty.
  The utility configuration (weights + normalization references) is frozen
  and hashed so a recorded ``U_b`` is reproducible.
* **ΔU learning target**::

      ΔU = U_symbolic − U_LLM

  Train on ``ΔU`` with an **abstention band** around zero: tasks with
  ``|ΔU| ≤ band`` yield the ``ABSTAIN`` target (no fabricated preference
  when both backends tie or both fail). ``y*(x) = argmax_a U(a, x)`` outside
  the band.
* **ABSTAIN** as a first-class route action (V039-003): ``symbolic wrong ∧
  llm wrong`` must not produce a fabricated preference.

The module is dependency-free (stdlib + numpy + the outcome/verification
layers) so it imports in any environment that imports ``daph_learning``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from daph_learning.autolearn.outcome import (
    BackendExecutionRecord,
    OutcomeSemantics,
    build_outcome_semantics,
)
from daph_learning.evaluation.verification import (
    OutcomeVerifier,
    RouteSuitabilityVerifier,
    VerificationStatus,
)
from daph_learning.evaluation.verifiers import select_verifier_for_task

# Route actions including the v0.3.9 ABSTAIN action.
RouteTarget = str  # "symbolic" | "llm" | "abstain"
ABSTAIN = "abstain"


# --- Frozen utility configuration ---

@dataclass(frozen=True)
class UtilityConfig:
    """Frozen configuration for the per-backend utility ``U_b``.

    ``U_b = quality_weight·Q_b − λ_t·(T_b/time_ref) − λ_c·(C_b/compute_ref) − λ_r·R_b``

    All weights and normalization references are part of the frozen config:
    once a run records a ``U_b`` the config hash must not change, or the
    recorded utility is not reproducible. :meth:`freeze` returns a copy with
    :attr:`config_sha256` populated.
    """

    quality_weight: float = 1.0
    lambda_time: float = 0.0
    lambda_compute: float = 0.0
    lambda_risk: float = 1.0
    time_reference_ms: float = 1000.0
    compute_reference: float = 1.0
    abstention_band: float = 0.0
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.time_reference_ms <= 0:
            raise ValueError("time_reference_ms must be > 0")
        if self.compute_reference <= 0:
            raise ValueError("compute_reference must be > 0")
        if self.abstention_band < 0:
            raise ValueError("abstention_band must be >= 0")

    def _hash_payload(self) -> str:
        payload = {
            "quality_weight": self.quality_weight,
            "lambda_time": self.lambda_time,
            "lambda_compute": self.lambda_compute,
            "lambda_risk": self.lambda_risk,
            "time_reference_ms": self.time_reference_ms,
            "compute_reference": self.compute_reference,
            "abstention_band": self.abstention_band,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def freeze(self) -> "UtilityConfig":
        """Return a copy with ``config_sha256`` populated (idempotent)."""
        if self.config_sha256 is not None:
            return self
        return dataclass_replace(self, config_sha256=self._hash_payload())


def dataclass_replace(instance: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` that preserves frozen dataclasses.

    For :class:`UtilityConfig`, any field replacement invalidates the cached
    ``config_sha256`` (set to ``None``) so the returned instance must be
    re-frozen. This prevents a stale hash from surviving a change to a
    hashed field. Pass ``config_sha256=...`` explicitly to override.
    """
    from dataclasses import replace
    if isinstance(instance, UtilityConfig) and "config_sha256" not in changes:
        changes["config_sha256"] = None
    return replace(instance, **changes)


# --- Utility computation ---

@dataclass(frozen=True)
class UtilityBreakdown:
    """The frozen, reproducible components of a single ``U_b``."""

    backend: str
    quality: float        # Q_b
    time_term: float      # λ_t·(T_b/time_ref)
    compute_term: float   # λ_c·(C_b/compute_ref)
    risk_term: float      # λ_r·R_b
    utility: float        # U_b = quality_weight·Q_b − time_term − compute_term − risk_term
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quality(record: BackendExecutionRecord) -> float:
    """Q_b: verified-quality reward. 1.0 iff executed and verified correct."""
    return 1.0 if record.is_correct else 0.0


def _risk(record: BackendExecutionRecord) -> float:
    """R_b: execution-risk penalty indicator.

    1.0 if the backend executed but failed (failure_type set, or verifier
    reports EXECUTION_FAILED / INVALID_OUTPUT), else 0.0. An unexecuted
    backend contributes 0.0 risk — it was not run, so it did not fail; it
    simply earns no quality reward.
    """
    if not record.executed:
        return 0.0
    if record.failure_type is not None:
        return 1.0
    if record.verification.status in {
        VerificationStatus.EXECUTION_FAILED,
        VerificationStatus.INVALID_OUTPUT,
    }:
        return 1.0
    return 0.0


def compute_backend_utility(
    record: BackendExecutionRecord,
    config: UtilityConfig,
) -> UtilityBreakdown:
    """Compute the frozen utility ``U_b`` for one backend record.

    v0.3.10.3.1 — Section 3: delegates to the canonical
    :func:`daph_learning.policy.utility.utility_formula` so the
    experience-construction path and the evaluation path share one
    formula implementation and cannot drift.

    v0.4 — now delegates to the generic
    :class:`daph_learning.executive.UtilityModel` which is the single
    canonical implementation of ``U(state, action)``. The legacy
    ``UtilityBreakdown`` is returned for backward compatibility.
    """
    from daph_learning.executive.adapters import (
        utility_model_from_legacy_config,
        action_from_backend,
    )
    from daph_learning.executive.types import ActionExecution

    cfg = config.freeze()
    generic_model = utility_model_from_legacy_config(cfg)

    # Convert BackendExecutionRecord → ActionExecution
    verification = record.verification
    verified_correct = None
    if hasattr(verification, "status"):
        status = verification.status
        status_val = status.value if hasattr(status, "value") else str(status)
        if status_val == "verified_correct":
            verified_correct = True
        elif status_val == "verified_incorrect":
            verified_correct = False

    action_id = action_from_backend(record.backend)
    action_exec = ActionExecution(
        action_id=action_id,
        selected=record.selected,
        executed=record.executed,
        output=record.output,
        verified_correct=verified_correct,
        verifier_name=getattr(verification, "verifier", "unknown"),
        latency_ms=record.latency_ms,
        compute_cost=record.compute_cost,
        failure_type=record.failure_type,
    )

    generic_bd = generic_model.compute(action_exec)

    # Return legacy UtilityBreakdown for backward compatibility
    return UtilityBreakdown(
        backend=record.backend,
        quality=generic_bd.quality,
        time_term=generic_bd.time_term,
        compute_term=generic_bd.compute_term,
        risk_term=generic_bd.risk_term,
        utility=generic_bd.utility,
        config_sha256=generic_bd.config_sha256,
    )


# --- ΔU learning target ---

@dataclass(frozen=True)
class LearningTarget:
    """The explicit learning target derived from counterfactual ΔU.

    Attributes
    ----------
    task_id : str
    u_symbolic : UtilityBreakdown
    u_llm : UtilityBreakdown
    delta_u : float
        ``U_symbolic − U_LLM``.
    target : str
        ``"symbolic"`` if ``ΔU > band``, ``"llm"`` if ``ΔU < −band``,
        else ``"abstain"``.
    weight : float
        Utility-weighted example weight (V039-006). ``g(|ΔU|)`` outside the
        abstention band, ``0.0`` inside it.
    """

    task_id: str
    u_symbolic: UtilityBreakdown
    u_llm: UtilityBreakdown
    delta_u: float
    target: RouteTarget
    weight: float

    @property
    def is_abstain(self) -> bool:
        return self.target == ABSTAIN


def derive_learning_target(
    outcome: OutcomeSemantics,
    config: UtilityConfig,
    *,
    weight_fn: Callable[[float, float], float] | None = None,
    w_max: float = 1.0,
) -> LearningTarget:
    """Derive the ΔU learning target for one task.

    ``weight_fn(|ΔU|, w_max)`` defaults to ``min(w_max, |ΔU|)`` (V039-006).
    The weight is zero inside the abstention band: abstain/both-fail/both-tie
    cases contribute no training signal.
    """
    if weight_fn is None:
        weight_fn = _default_weight
    cfg = config.freeze()
    u_sym = compute_backend_utility(outcome.backends["symbolic"], cfg)
    u_llm = compute_backend_utility(outcome.backends["llm"], cfg)
    delta = u_sym.utility - u_llm.utility
    band = cfg.abstention_band
    if delta > band:
        target: RouteTarget = "symbolic"
    elif delta < -band:
        target = "llm"
    else:
        target = ABSTAIN
    weight = 0.0 if target == ABSTAIN else float(weight_fn(abs(delta), w_max))
    return LearningTarget(
        task_id=outcome.task_id,
        u_symbolic=u_sym,
        u_llm=u_llm,
        delta_u=delta,
        target=target,
        weight=weight,
    )


def _default_weight(abs_delta: float, w_max: float) -> float:
    return min(w_max, abs_delta)


# --- Counterfactual both-backend execution ---

BackendRunner = Callable[[Mapping[str, Any]], dict[str, Any]]
"""A backend runner executes one backend on one task and returns a dict with
keys ``output`` (Any|None), ``execution_succeeded`` (bool), ``latency_ms``
(float|None), ``compute_cost`` (float|None), ``failure_type`` (str|None)."""


def _symbolic_runner_default(task: Mapping[str, Any]) -> dict[str, Any]:
    """Default symbolic runner using the bounded executor."""
    from daph_learning.autolearn.loop import _execute_symbolic
    output, success = _execute_symbolic(dict(task))
    return {
        "output": output,
        "execution_succeeded": success,
        "latency_ms": None,
        "compute_cost": None,
        "failure_type": None if success else "execution_error",
    }


def execute_both_backends(
    task: Mapping[str, Any],
    *,
    backend_selected: str | None,
    symbolic_runner: BackendRunner | None = None,
    llm_runner: BackendRunner | None = None,
    verifier: OutcomeVerifier | None = None,
    route_suitability_verifier: RouteSuitabilityVerifier | None = None,
    expected: Any | None = None,
    oracle_field: str = "utility_oracle",
    skip_llm: bool = False,
) -> OutcomeSemantics:
    """Execute **both** backends on one task and assemble the outcome.

    This is the counterfactual execution primitive (V039-001). Both backends
    are run independently and verified independently; the router's
    ``backend_selected`` decision is recorded for route-suitability analysis
    but does not gate execution.

    Parameters
    ----------
    skip_llm : bool
        If True, the LLM backend is not executed (e.g. no model available).
        The LLM record will be ``NOT_EXECUTED`` and can never be correct —
        the protocol invariant is preserved. ΔU-derived targets from such
        an outcome are not eligible to train a preference.
    """
    if symbolic_runner is None:
        symbolic_runner = _symbolic_runner_default
    if llm_runner is None:
        # No default LLM runner is possible without a model. If the caller
        # did not supply one we treat LLM as not executed rather than
        # fabricating an output.
        skip_llm = True

    executed: dict[str, dict[str, Any]] = {}
    try:
        executed["symbolic"] = symbolic_runner(task)
    except Exception as exc:  # noqa: BLE001 - runner contract failure
        executed["symbolic"] = {
            "output": None,
            "execution_succeeded": False,
            "latency_ms": None,
            "compute_cost": None,
            "failure_type": f"runner_error:{type(exc).__name__}",
        }
    if not skip_llm:
        try:
            executed["llm"] = llm_runner(task)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - runner contract failure
            executed["llm"] = {
                "output": None,
                "execution_succeeded": False,
                "latency_ms": None,
                "compute_cost": None,
                "failure_type": f"runner_error:{type(exc).__name__}",
            }

    return build_outcome_semantics(
        task,
        backend_selected=backend_selected,
        executed_backends=executed,
        verifier=verifier,
        route_suitability_verifier=route_suitability_verifier,
        expected=expected,
        oracle_field=oracle_field,
    )


def derive_targets_for_tasks(
    tasks: list[dict[str, Any]],
    *,
    backend_selected_for: Callable[[Mapping[str, Any]], str | None] | None = None,
    symbolic_runner: BackendRunner | None = None,
    llm_runner: BackendRunner | None = None,
    utility_config: UtilityConfig,
    verifier: OutcomeVerifier | None = None,
    route_suitability_verifier: RouteSuitabilityVerifier | None = None,
    expected_field: str = "expected",
    oracle_field: str = "utility_oracle",
    label_field: str | None = None,
    skip_llm: bool = False,
) -> list[LearningTarget]:
    """Run counterfactual execution over a task set and derive ΔU targets.

    ``label_field`` (V039-004) selects the policy-oracle field used for
    route-suitability. Changing it changes the route-suitability record and,
    where the utility target is tied (within the abstention band), the
    bootstrap policy label used to break ties for diagnostic class
    assignment — so changing ``label_field`` changes training, as the
    acceptance test requires.
    """
    if backend_selected_for is None:
        # Default: use the task's own oracle as the *selected* route, so the
        # route-suitability record is well-defined. The learning target
        # itself is ΔU-derived, not label-derived.
        def _default_selector(t: Mapping[str, Any]) -> str | None:
            return t.get(oracle_field)  # type: ignore[return-value]
        backend_selected_for = _default_selector

    effective_oracle = label_field or oracle_field
    if route_suitability_verifier is None:
        route_suitability_verifier = RouteSuitabilityVerifier(oracle_field=effective_oracle)

    targets: list[LearningTarget] = []
    for task in tasks:
        expected = task.get(expected_field)
        task_verifier = verifier if verifier is not None else select_verifier_for_task(task)
        outcome = execute_both_backends(
            task,
            backend_selected=backend_selected_for(task),
            symbolic_runner=symbolic_runner,
            llm_runner=llm_runner,
            verifier=task_verifier,
            route_suitability_verifier=route_suitability_verifier,
            expected=expected,
            oracle_field=effective_oracle,
            skip_llm=skip_llm,
        )
        targets.append(derive_learning_target(outcome, utility_config))
    return targets


__all__ = [
    "ABSTAIN",
    "BackendRunner",
    "LearningTarget",
    "RouteTarget",
    "UtilityBreakdown",
    "UtilityConfig",
    "compute_backend_utility",
    "dataclass_replace",
    "derive_learning_target",
    "derive_targets_for_tasks",
    "execute_both_backends",
]
