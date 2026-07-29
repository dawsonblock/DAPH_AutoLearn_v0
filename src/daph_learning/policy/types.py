"""v0.3.10 — required typed data structures for the counterfactual policy
learner (Section 32 of the spec).

These structures provide clean, frozen, hashable records for backend
outcomes, counterfactual experiences, captured activations, and
contextual-bandit policy decisions. They are designed to be joined
strictly by ``task_id`` (Section 31) and to carry full provenance for
experiment manifests (Section 33).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Route(str, Enum):
    """Primary action space for v0.3.10.

    The action space is designed to grow in future releases
    (HYBRID, VERIFY, RETRIEVE, THINK_MORE, TOOL, ...). For this release
    only SYMBOLIC, LLM, and ABSTAIN are produced by the learned policy.
    """

    SYMBOLIC = "symbolic"
    LLM = "llm"
    ABSTAIN = "abstain"

    @classmethod
    def from_str(cls, value: str) -> "Route":
        """Parse a route string, raising ``ValueError`` on unknown values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"unknown route {value!r}; expected one of {[r.value for r in cls]}")


@dataclass(frozen=True)
class BackendOutcome:
    """Verified outcome of executing one backend on one task (Section 32).

    v0.3.10.3 — expanded with execution semantics to distinguish:
    - backend unavailable (not installed / no model)
    - backend unsupported for this task type
    - not executed (skipped)
    - execution failed (error/timeout)
    - execution succeeded but answer wrong
    - execution succeeded and answer verified

    Attributes
    ----------
    task_id : str
    backend : str
        ``"symbolic"`` or ``"llm"``.
    available : bool
        Whether the backend was available for this task (not unsupported).
    executed : bool
        Whether the backend was actually executed (not skipped).
    execution_success : bool
        Whether execution completed without error.
    output_text : str | None
        The raw output produced by the backend (for verification).
    output_hash : str | None
        SHA-256 hash of the output text (for provenance).
    verifier_status : str
        One of: ``"verified_correct"``, ``"verified_incorrect"``,
        ``"verifier_unsupported"``, ``"not_verified"``.
    correct : bool
        Whether the verifier confirmed the output. False if not verified.
    quality : float
        Verified task quality / correctness score in ``[0, 1]``.
    latency_sec : float
        Wall-clock latency in seconds.
    normalized_cost : float
        Monetary or normalized compute cost.
    risk : float
        Risk / safety penalty in ``[0, 1]``.
    verifier_confidence : float
        Confidence of the verifier in ``[0, 1]``. 1.0 for deterministic
        verifiers. 0.0 if not verified.
    failure_reason : str | None
        If execution or verification failed, why.
    """

    task_id: str
    backend: str
    available: bool = True
    executed: bool = True
    execution_success: bool = True
    output_text: str | None = None
    output_hash: str | None = None
    verifier_status: str = "not_verified"
    correct: bool = False
    quality: float = 0.0
    latency_sec: float = 0.0
    normalized_cost: float = 0.0
    risk: float = 0.0
    verifier_confidence: float = 0.0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("symbolic", "llm"):
            raise ValueError(f"backend must be 'symbolic' or 'llm', got {self.backend!r}")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        if not 0.0 <= self.verifier_confidence <= 1.0:
            raise ValueError("verifier_confidence must be in [0, 1]")
        if self.latency_sec < 0:
            raise ValueError("latency_sec must be >= 0")
        if self.normalized_cost < 0:
            raise ValueError("normalized_cost must be >= 0")
        if self.verifier_status not in (
            "verified_correct", "verified_incorrect",
            "verifier_unsupported", "not_verified",
        ):
            raise ValueError(f"invalid verifier_status: {self.verifier_status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CounterfactualExperience:
    """A single counterfactual experience joining both backend outcomes
    with the utility difference and learning signal (Section 32).

    Attributes
    ----------
    task_id : str
    symbolic : BackendOutcome
    llm : BackendOutcome
    delta_utility : float
        ``U(symbolic) - U(llm)``.
    preferred_action : Route
        ``argmax_a U(a)`` outside the abstention band, else ``ABSTAIN``.
    sample_weight : float
        Utility-weighted example weight (zero inside the abstention band).
    """

    task_id: str
    symbolic: BackendOutcome
    llm: BackendOutcome
    delta_utility: float
    preferred_action: Route
    sample_weight: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preferred_action"] = self.preferred_action.value
        return d


@dataclass(frozen=True)
class CapturedActivation:
    """A captured activation vector for one task at one layer/location
    (Section 32).

    Attributes
    ----------
    task_id : str
    layer : int
    token_location : str
    vector : np.ndarray
        1-D activation vector.
    success : bool
        Whether the capture succeeded.
    failure_reason : str | None
        If ``success`` is False, why.
    """

    task_id: str
    layer: int
    token_location: str
    vector: np.ndarray
    success: bool
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.success and self.failure_reason is None:
            object.__setattr__(self, "failure_reason", "capture_failed")
        if self.success:
            arr = np.asarray(self.vector)
            if arr.ndim != 1:
                raise ValueError("vector must be 1-D on success")
            if not np.all(np.isfinite(arr)):
                raise ValueError("vector contains NaN/Inf on success")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["vector"] = np.asarray(self.vector).tolist() if self.success else None
        return d


@dataclass(frozen=True)
class FeatureRecord:
    """Task-bound feature vector (Section 5).

    Replaces naked ``np.ndarray`` rows at cross-module boundaries so
    that experiences and activations can be joined by ``task_id``
    instead of inferred from array order. Naked arrays at module
    boundaries are unsafe — a shuffled feature order would silently
    misalign features from experiences.

    Attributes
    ----------
    task_id : str
        Unique task identifier (must match a
        :class:`CounterfactualExperience` task_id).
    features : np.ndarray
        1-D feature vector (e.g. captured activation or PCA-projected
        representation). Must be finite.
    """

    task_id: str
    features: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.features)
        if arr.ndim != 1:
            raise ValueError("features must be 1-D")
        if not np.all(np.isfinite(arr)):
            raise ValueError("features contain NaN/Inf")
        object.__setattr__(self, "features", arr.astype(np.float32, copy=False))

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "features": np.asarray(self.features).tolist()}


@dataclass(frozen=True)
class PolicyDecision:
    """Contextual-bandit-compatible logging record for one deployment
    routing decision (Section 20).

    Every deployment routing decision must log the chosen action, the
    probability of the chosen action (for inverse propensity scoring),
    the full policy probability distribution, the policy version, and
    the set of available actions. This enables future off-policy
    learning (IPS, doubly-robust estimation).

    v0.3.10.1 — adds ``chosen_propensity`` as an explicit field (the
    historical ``propensity`` property is kept as a back-compat alias).
    """

    task_id: str
    action: Route
    probabilities: dict[str, float]
    policy_version: str
    chosen_propensity: float | None = None
    available_actions: tuple[str, ...] = ("symbolic", "llm", "abstain")

    def __post_init__(self) -> None:
        if self.action.value not in self.probabilities:
            raise ValueError(
                f"action {self.action.value!r} not in probabilities "
                f"{list(self.probabilities)}"
            )
        for name, prob in self.probabilities.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"probability for {name!r} must be in [0, 1], got {prob}")
        if self.action.value not in self.available_actions:
            raise ValueError(
                f"action {self.action.value!r} not in available_actions "
                f"{self.available_actions}"
            )

    @property
    def propensity(self) -> float:
        """Probability of the chosen action (for IPS weighting).

        v0.3.10.1 — back-compat alias for ``chosen_propensity``. If
        ``chosen_propensity`` was set explicitly at construction, that
        value is returned; otherwise it is looked up in
        ``probabilities``.
        """
        if self.chosen_propensity is not None:
            return float(self.chosen_propensity)
        return float(self.probabilities[self.action.value])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        d["available_actions"] = list(self.available_actions)
        d["chosen_propensity"] = self.propensity
        return d


__all__ = [
    "BackendOutcome",
    "CapturedActivation",
    "CounterfactualExperience",
    "FeatureRecord",
    "PolicyDecision",
    "Route",
]
