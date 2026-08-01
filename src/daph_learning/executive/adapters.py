"""DAPH v0.4 — Adapters between legacy binary types and generic executive types.

These adapters allow the existing v0.3.x binary routing experiment
(``symbolic`` vs ``llm``) to be represented through the new generic
executive interface. This is the key migration bridge:

    old binary engine
            │
            ▼
    adapter → new generic types
            │
            ▼
    same experimental semantics

Once all consumers use the generic types, the old hard-coded binary
path can be removed.
"""

from __future__ import annotations

from typing import Any, Mapping

from daph_learning.executive.types import (
    ActionDescriptor,
    ActionSpace,
    ExecutiveState,
    ActionExecution,
    CounterfactualSet,
    ActionDecision,
    UtilityModel,
)


# ──────────────────────────────────────────────────────────────────────
# Binary Action Space
# ──────────────────────────────────────────────────────────────────────

# The canonical binary action descriptors.
_SYMBOLIC = ActionDescriptor(
    action_id="action.symbolic_arithmetic",
    display_name="Symbolic Arithmetic",
    description="Deterministic symbolic computation backend (v0.3.x 'symbolic').",
    cost_estimate=0.05,
    tags=("symbolic", "arithmetic", "deterministic"),
)

_LLM = ActionDescriptor(
    action_id="action.llm_direct",
    display_name="LLM Direct",
    description="Direct LLM generation backend (v0.3.x 'llm').",
    cost_estimate=0.20,
    tags=("llm", "generation", "neural"),
)

_ABSTAIN = "action.abstain"


def binary_action_space() -> ActionSpace:
    """Return the ActionSpace for the legacy binary routing experiment.

    This represents the v0.3.x ``symbolic`` vs ``llm`` experiment
    through the generic interface. The abstain action is included
    as the ``abstain_id``.
    """
    return ActionSpace(
        actions=(_SYMBOLIC, _LLM),
        abstain_id=_ABSTAIN,
    )


# ──────────────────────────────────────────────────────────────────────
# Legacy → Generic Conversions
# ──────────────────────────────────────────────────────────────────────

_BACKEND_TO_ACTION: dict[str, str] = {
    "symbolic": _SYMBOLIC.action_id,
    "llm": _LLM.action_id,
    "abstain": _ABSTAIN,
}

_ACTION_TO_BACKEND: dict[str, str] = {
    v: k for k, v in _BACKEND_TO_ACTION.items()
}


def action_from_backend(backend: str) -> str:
    """Map a legacy backend name to a generic action_id."""
    if backend not in _BACKEND_TO_ACTION:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of "
            f"{list(_BACKEND_TO_ACTION.keys())}")
    return _BACKEND_TO_ACTION[backend]


def backend_from_action(action_id: str) -> str:
    """Map a generic action_id back to a legacy backend name."""
    if action_id not in _ACTION_TO_BACKEND:
        raise ValueError(
            f"unknown action_id {action_id!r}; expected one of "
            f"{list(_ACTION_TO_BACKEND.keys())}")
    return _ACTION_TO_BACKEND[action_id]


def executive_state_from_task(task: Mapping[str, Any]) -> ExecutiveState:
    """Build an ExecutiveState from a legacy task dict."""
    prompt = str(task.get("prompt", task.get("specification", "")))
    metadata = {
        k: v for k, v in task.items()
        if k not in ("prompt", "specification")
    }
    return ExecutiveState(
        task_id=str(task.get("task_id", "")),
        prompt=prompt,
        task_metadata=metadata,
    )


def action_execution_from_backend_record(
    backend: str,
    record: Any,
    selected: bool,
) -> ActionExecution:
    """Convert a legacy BackendExecutionRecord to ActionExecution.

    Works with both ``daph_learning.autolearn.outcome.BackendExecutionRecord``
    and ``daph_learning.policy.types.BackendOutcome``.
    """
    action_id = action_from_backend(backend)

    # Detect which legacy type we're converting from.
    if hasattr(record, "verification"):
        # BackendExecutionRecord (autolearn.outcome)
        verification = record.verification
        verified_correct = None
        if hasattr(verification, "status"):
            status = verification.status
            # Check enum value or string
            status_val = status.value if hasattr(status, "value") else str(status)
            if status_val == "verified_correct":
                verified_correct = True
            elif status_val == "verified_incorrect":
                verified_correct = False
        return ActionExecution(
            action_id=action_id,
            selected=selected,
            executed=record.executed,
            output=record.output,
            verified_correct=verified_correct,
            verifier_name=getattr(verification, "verifier", "unknown"),
            latency_ms=record.latency_ms,
            compute_cost=record.compute_cost,
            failure_type=record.failure_type,
        )
    else:
        # BackendOutcome (policy.types)
        verified_correct = record.verified_correct  # True/False/None
        return ActionExecution(
            action_id=action_id,
            selected=selected,
            executed=record.executed,
            output=record.output_text,
            verified_correct=verified_correct,
            verifier_name="legacy",
            latency_ms=record.latency_sec * 1000.0 if record.latency_sec else None,
            compute_cost=record.normalized_cost,
            failure_type=record.failure_reason,
        )


def counterfactual_set_from_outcome(
    outcome: Any,
    task: Mapping[str, Any] | None = None,
) -> CounterfactualSet:
    """Convert a legacy OutcomeSemantics to a CounterfactualSet.

    Works with ``daph_learning.autolearn.outcome.OutcomeSemantics``.
    """
    state = executive_state_from_task(task or {"task_id": outcome.task_id, "prompt": ""})

    executions: dict[str, ActionExecution] = {}
    for backend, record in outcome.backends.items():
        selected = (backend == outcome.backend_selected)
        executions[action_from_backend(backend)] = (
            action_execution_from_backend_record(backend, record, selected))

    # Map selected action
    selected_action = None
    if outcome.backend_selected and outcome.backend_selected in _BACKEND_TO_ACTION:
        selected_action = action_from_backend(outcome.backend_selected)

    return CounterfactualSet(
        state=state,
        executions=executions,
        selected_action=selected_action,
    )


# ──────────────────────────────────────────────────────────────────────
# Policy Decision Conversions
# ──────────────────────────────────────────────────────────────────────

def action_decision_from_symbolic_probability(
    task_id: str,
    symbolic_probability: float,
    *,
    calibrated: bool = False,
    action_space: ActionSpace | None = None,
) -> ActionDecision:
    """Convert a legacy ``symbolic_probability`` to an ActionDecision.

    The binary policy outputs ``P(symbolic | x)``. This maps to:
    - P(action.symbolic_arithmetic) = symbolic_probability
    - P(action.llm_direct) = 1 - symbolic_probability
    """
    if action_space is None:
        action_space = binary_action_space()

    sym_id = _SYMBOLIC.action_id
    llm_id = _LLM.action_id

    p = max(0.0, min(1.0, float(symbolic_probability)))
    probabilities = {sym_id: p, llm_id: 1.0 - p}
    scores = {sym_id: p, llm_id: 1.0 - p}  # raw scores = probabilities for binary

    selected = sym_id if p >= 0.5 else llm_id
    confidence = max(p, 1.0 - p)

    return ActionDecision(
        state_id=task_id,
        scores=scores,
        probabilities=probabilities,
        selected_action=selected,
        confidence=confidence,
        calibration_applied=calibrated,
    )


def symbolic_probability_from_action_decision(
    decision: ActionDecision,
) -> float:
    """Extract the legacy ``symbolic_probability`` from an ActionDecision.

    Returns the probability of ``action.symbolic_arithmetic``.
    """
    return decision.probability(_SYMBOLIC.action_id)


# ──────────────────────────────────────────────────────────────────────
# Utility Model Conversion
# ──────────────────────────────────────────────────────────────────────

def utility_model_from_legacy_config(
    legacy_config: Any,
) -> UtilityModel:
    """Convert a legacy ``UtilityConfig`` to a ``UtilityModel``."""
    return UtilityModel(
        quality_weight=legacy_config.quality_weight,
        lambda_time=legacy_config.lambda_time,
        lambda_compute=legacy_config.lambda_compute,
        lambda_risk=legacy_config.lambda_risk,
        time_reference_ms=legacy_config.time_reference_ms,
        compute_reference=legacy_config.compute_reference,
        abstention_band=legacy_config.abstention_band,
        config_sha256=legacy_config.config_sha256,
    )
