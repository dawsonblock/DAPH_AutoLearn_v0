"""DAPH v0.4 — Core generic executive types.

These types generalize the binary ``symbolic`` vs ``llm`` routing into
arbitrary N-action executive qualification. The key abstraction is:

    state + arbitrary actions + verified utilities

instead of:

    symbolic_probability = P(symbolic | x)
    ΔU = U_symbolic - U_LLM

All types are frozen dataclasses (immutable, hashable) for provenance
and reproducibility, matching the existing v0.3.x discipline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Action Space
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionDescriptor:
    """Describes an executable action in the action space.

    An action is anything that can be executed on a state to produce
    an output: ``reasoning.direct``, ``retrieval.vector``, ``symbolic_arithmetic``,
    ``llm_direct``, ``reasoning.decompose``, ``reasoning.coconut``, etc.

    Attributes
    ----------
    action_id : str
        Unique identifier (e.g. ``"reasoning.direct.v1"``).
    display_name : str
        Human-readable name for reports.
    description : str
        What this action does.
    cost_estimate : float
        Estimated relative cost (0.0 = free, 1.0 = expensive).
        Used for cost-sensitive utility, not for execution decisions.
    tags : tuple[str, ...]
        Categorical tags for grouping (e.g. ``("reasoning", "direct")``).
    """

    action_id: str
    display_name: str = ""
    description: str = ""
    cost_estimate: float = 0.0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not 0.0 <= self.cost_estimate <= 1.0:
            raise ValueError("cost_estimate must be in [0, 1]")
        if self.display_name == "":
            object.__setattr__(self, "display_name", self.action_id)

    @property
    def namespace(self) -> str:
        """The namespace prefix before the first dot (e.g. ``"reasoning"``)."""
        return self.action_id.split(".")[0] if "." in self.action_id else self.action_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionSpace:
    """The set of candidate actions for an experiment.

    This replaces the hard-coded ``BACKENDS = ("symbolic", "llm")`` tuple.
    An action space can have any number of actions (2 for binary routing,
    3+ for multi-action executive qualification).

    Attributes
    ----------
    actions : tuple[ActionDescriptor, ...]
        The candidate actions, in canonical order.
    abstain_id : str | None
        If non-None, the action_id that represents abstention
        (no action selected). This replaces the hard-coded ``"abstain"``.
    """

    actions: tuple[ActionDescriptor, ...]
    abstain_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.actions) < 2:
            raise ValueError("ActionSpace must have at least 2 actions")
        ids = [a.action_id for a in self.actions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate action_ids: {ids}")
        if self.abstain_id is not None and self.abstain_id in ids:
            raise ValueError(
                f"abstain_id {self.abstain_id!r} must not be in actions")

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(a.action_id for a in self.actions)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def executable_ids(self) -> tuple[str, ...]:
        """All action_ids including abstain (if present)."""
        if self.abstain_id is not None:
            return self.action_ids + (self.abstain_id,)
        return self.action_ids

    def get(self, action_id: str) -> ActionDescriptor | None:
        for a in self.actions:
            if a.action_id == action_id:
                return a
        return None

    def is_valid(self, action_id: str) -> bool:
        return action_id in self.executable_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "abstain_id": self.abstain_id,
        }


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Executive State
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutiveState:
    """The observable state at a decision point.

    This is the input to the policy: given this state, which action
    should be selected?

    Attributes
    ----------
    task_id : str
        Unique task identifier (joins with experiences and features).
    prompt : str
        The rendered prompt that would be sent to the action executor.
    task_metadata : Mapping[str, Any]
        Task-specific metadata (subtype, group_id, capability_ids, etc.).
    hidden_state_ref : str | None
        Optional reference to a captured hidden-state vector
        (e.g. ``"layer_24:last_token:model_v1"``). The actual vector
        is stored separately; this is a provenance reference.
    """

    task_id: str
    prompt: str
    task_metadata: Mapping[str, Any] = field(default_factory=dict)
    hidden_state_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")

    @property
    def group_id(self) -> str | None:
        return self.task_metadata.get("group_id")

    @property
    def subtype(self) -> str | None:
        return self.task_metadata.get("subtype")

    @property
    def split(self) -> str | None:
        return self.task_metadata.get("split")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "task_metadata": dict(self.task_metadata),
            "hidden_state_ref": self.hidden_state_ref,
        }


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Action Execution
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionExecution:
    """The result of executing one action on one state.

    This generalizes ``BackendExecutionRecord`` to any action.
    The protocol invariant is preserved: an unexecuted action can
    never be correct.

    Attributes
    ----------
    action_id : str
        Which action was executed.
    selected : bool
        Whether the policy selected this action (for route-suitability).
    executed : bool
        Whether this action was actually run.
    output : Any | None
        Raw output from the action.
    verified_correct : bool | None
        Independent verifier result: ``True``/``False``/``None`` (not verified).
    verifier_name : str
        Identity of the verifier used.
    latency_ms : float | None
        Wall-clock latency in milliseconds.
    compute_cost : float | None
        Normalized compute cost.
    failure_type : str | None
        ``None`` on success, else a short failure classifier.
    """

    action_id: str
    selected: bool
    executed: bool
    output: Any | None = None
    verified_correct: bool | None = None
    verifier_name: str = "none"
    latency_ms: float | None = None
    compute_cost: float | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        # Protocol invariant: unexecuted → not correct.
        if not self.executed and self.verified_correct is True:
            raise ValueError(
                f"protocol invariant violated: action {self.action_id!r} is "
                f"unexecuted but verified_correct is True")

    @property
    def is_correct(self) -> bool:
        """``True`` iff executed and independently verified correct."""
        return bool(self.executed and self.verified_correct is True)

    @property
    def quality(self) -> float:
        """Q_a: verified-quality reward. 1.0 iff correct, else 0.0."""
        return 1.0 if self.is_correct else 0.0

    @property
    def risk(self) -> float:
        """R_a: execution-risk penalty (1.0 if executed and failed).

        Risk is about execution failure (timeout, error, invalid output),
        NOT about incorrect answers. An incorrect answer is a quality
        issue (Q=0), not a risk issue. This matches the legacy
        ``_risk()`` semantics from v0.3.x.
        """
        if not self.executed:
            return 0.0
        if self.failure_type is not None:
            return 1.0
        # verified_correct=False alone is NOT a risk — it's a quality issue.
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "selected": self.selected,
            "executed": self.executed,
            "output": self.output,
            "verified_correct": self.verified_correct,
            "verifier_name": self.verifier_name,
            "latency_ms": self.latency_ms,
            "compute_cost": self.compute_cost,
            "failure_type": self.failure_type,
        }


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Counterfactual Set
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CounterfactualSet:
    """All action executions for a single state (counterfactual execution).

    This generalizes the binary ``OutcomeSemantics`` (which held
    ``backends["symbolic"]`` and ``backends["llm"]``) to N actions.

    Attributes
    ----------
    state : ExecutiveState
        The decision point.
    executions : Mapping[str, ActionExecution]
        Per-action execution records, keyed by action_id.
    selected_action : str | None
        The action selected by the policy (or ``None`` for abstention).
    """

    state: ExecutiveState
    executions: Mapping[str, ActionExecution]
    selected_action: str | None = None

    def __post_init__(self) -> None:
        for action_id, exec_record in self.executions.items():
            if not isinstance(exec_record, ActionExecution):
                raise TypeError(
                    f"executions[{action_id!r}] must be ActionExecution, "
                    f"got {type(exec_record)}")
            if exec_record.action_id != action_id:
                raise ValueError(
                    f"executions[{action_id!r}].action_id is "
                    f"{exec_record.action_id!r}, mismatch")

    @property
    def task_id(self) -> str:
        return self.state.task_id

    @property
    def all_executed(self) -> bool:
        """Whether all candidate actions were executed (counterfactual)."""
        return all(e.executed for e in self.executions.values())

    def execution(self, action_id: str) -> ActionExecution | None:
        return self.executions.get(action_id)

    def is_correct(self, action_id: str) -> bool:
        e = self.executions.get(action_id)
        return bool(e and e.is_correct)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "executions": {k: v.to_dict() for k, v in self.executions.items()},
            "selected_action": self.selected_action,
        }


# ──────────────────────────────────────────────────────────────────────
# Section 5 — Utility Model
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UtilityBreakdown:
    """The frozen, reproducible components of ``U(state, action)``.

    Attributes
    ----------
    action_id : str
    quality : float
        Q_a — verified-quality reward.
    time_term : float
        λ_t · (T_a / time_ref)
    compute_term : float
        λ_c · (C_a / compute_ref)
    risk_term : float
        λ_r · R_a
    utility : float
        U_a = quality_weight · Q_a − time_term − compute_term − risk_term
    config_sha256 : str
        Hash of the utility configuration (for reproducibility).
    """

    action_id: str
    quality: float
    time_term: float
    compute_term: float
    risk_term: float
    utility: float
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UtilityModel:
    """Frozen configuration for computing ``U(state, action)``.

    This generalizes ``UtilityConfig`` from v0.3.x. The formula is:

        U_a = quality_weight · Q_a − λ_t · (T_a / time_ref)
              − λ_c · (C_a / compute_ref) − λ_r · R_a

    All weights and references are frozen and hashed so a recorded
    utility is reproducible.

    Attributes
    ----------
    quality_weight : float
    lambda_time : float
    lambda_compute : float
    lambda_risk : float
    time_reference_ms : float
    compute_reference : float
    abstention_band : float
        Tasks where ``|U_a - U_b| ≤ band`` for all pairs yield abstention.
    config_sha256 : str | None
        Populated by ``freeze()``.
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

    def freeze(self) -> "UtilityModel":
        if self.config_sha256 is not None:
            return self
        from dataclasses import replace
        return replace(self, config_sha256=self._hash_payload())

    def compute(self, execution: ActionExecution) -> UtilityBreakdown:
        """Compute U(state, action) for one action execution."""
        cfg = self.freeze()
        q = execution.quality
        t = execution.latency_ms if execution.latency_ms is not None else 0.0
        c = execution.compute_cost if execution.compute_cost is not None else 0.0
        r = execution.risk
        time_term = self.lambda_time * (t / self.time_reference_ms)
        compute_term = self.lambda_compute * (c / self.compute_reference)
        risk_term = self.lambda_risk * r
        utility = (
            self.quality_weight * q
            - time_term
            - compute_term
            - risk_term
        )
        return UtilityBreakdown(
            action_id=execution.action_id,
            quality=q,
            time_term=time_term,
            compute_term=compute_term,
            risk_term=risk_term,
            utility=utility,
            config_sha256=cfg.config_sha256,
        )

    def compute_all(
        self, cf_set: CounterfactualSet
    ) -> dict[str, UtilityBreakdown]:
        """Compute U(state, action) for all actions in a counterfactual set."""
        return {
            action_id: self.compute(exec)
            for action_id, exec in cf_set.executions.items()
        }

    def best_action(
        self, cf_set: CounterfactualSet
    ) -> tuple[str, float]:
        """Return (best_action_id, best_utility) — the oracle action."""
        breakdowns = self.compute_all(cf_set)
        best_id = max(breakdowns, key=lambda a: breakdowns[a].utility)
        return best_id, breakdowns[best_id].utility

    def regret(
        self, cf_set: CounterfactualSet, chosen_action: str
    ) -> "Regret":
        """Compute regret of choosing ``chosen_action`` instead of the best."""
        breakdowns = self.compute_all(cf_set)
        best_id, best_u = self.best_action(cf_set)
        chosen_u = breakdowns[chosen_action].utility
        return Regret(
            state_id=cf_set.task_id,
            chosen_action=chosen_action,
            best_action=best_id,
            chosen_utility=chosen_u,
            best_utility=best_u,
            regret=best_u - chosen_u,
        )


# ──────────────────────────────────────────────────────────────────────
# Section 6 — Action Decision (Policy Output)
# ──────────────────────────────────────────────────────────────────────

# Type alias: action_id → probability
ActionProbability = dict[str, float]


@dataclass(frozen=True)
class ActionDecision:
    """The policy's output for one state.

    This replaces ``RoutingDecision`` (which had ``symbolic_probability``
    and hard-coded ``threshold_symbolic`` / ``threshold_llm``).

    Attributes
    ----------
    state_id : str
        Task identifier.
    scores : Mapping[str, float]
        Per-action raw scores (logits or similar).
    probabilities : Mapping[str, float]
        Per-action probabilities (softmax of scores, or calibrated).
    selected_action : str
        The action with the highest probability (or abstain_id).
    confidence : float
        Probability of the selected action.
    calibration_applied : bool
        Whether calibration was applied to the probabilities.
    """

    state_id: str
    scores: Mapping[str, float]
    probabilities: Mapping[str, float]
    selected_action: str
    confidence: float
    calibration_applied: bool = False

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must be non-empty")
        if not self.probabilities:
            raise ValueError("probabilities must be non-empty")
        total = sum(self.probabilities.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"probabilities must sum to ~1.0, got {total}")
        if self.selected_action not in self.probabilities:
            raise ValueError(
                f"selected_action {self.selected_action!r} not in "
                f"probabilities {list(self.probabilities.keys())}")

    def probability(self, action_id: str) -> float:
        return self.probabilities.get(action_id, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "scores": dict(self.scores),
            "probabilities": dict(self.probabilities),
            "selected_action": self.selected_action,
            "confidence": self.confidence,
            "calibration_applied": self.calibration_applied,
        }


# ──────────────────────────────────────────────────────────────────────
# Section 7 — Regret
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Regret:
    """Regret of choosing one action over the oracle best.

    R(s, a) = U(s, a*) - U(s, a)

    Attributes
    ----------
    state_id : str
    chosen_action : str
    best_action : str
    chosen_utility : float
    best_utility : float
    regret : float
        ``best_utility - chosen_utility`` (≥ 0).
    """

    state_id: str
    chosen_action: str
    best_action: str
    chosen_utility: float
    best_utility: float
    regret: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
