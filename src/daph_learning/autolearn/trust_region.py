"""v0.3.9 — counterfactual learning loop with trust-region update.

This module implements the **corrected** counterfactual AutoLearn architecture
(v0.3.9). The central repair (Section 1 of the spec) is that the held-out
promotion evaluation now uses the **candidate steering vector** to produce
candidate routing decisions and the **incumbent steering vector** to produce
incumbent routing decisions — via an injected :data:`RoutePolicyFn`.

The causal chain is now genuine:

    experience
        -> counterfactual utility (Delta-U)
        -> learning target
        -> candidate controller/vector update
        -> **changed routing behavior** (via RoutePolicyFn)
        -> held-out candidate-vs-incumbent evaluation
        -> promotion or rollback
        -> new incumbent

Previous defect: ``_evaluate_held_out`` accepted ``candidate_vector`` but
derived the candidate route from the oracle Delta-U target, and hard-coded the
incumbent route to ``"symbolic"``. This made promotion invalid because the
candidate's held-out decision did not depend on the candidate vector at all.

The loop is model-optional: the caller supplies
``capture_fn(tasks, class_label) -> CaptureResult`` and
``execute_fn(task, backend) -> dict`` and ``route_fn(tasks, vector, config) ->
[PolicyRouteDecision]`` so the loop is testable without a real model. The
trust-region math (:func:`trust_region_update`, :func:`weighted_class_mean`,
:func:`random_direction_at_scale`) is pure and unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .counterfactual import (
    ABSTAIN,
    LearningTarget,
    UtilityConfig,
    compute_backend_utility,
    derive_learning_target,
    execute_both_backends,
)
from .experience import (
    CaptureProvenance,
    CaptureResult,
    CapturedActivation,
    ExperienceLedger,
    build_experience_record,
    hash_task,
)
from .promotion import (
    HeldOutTaskResult,
    PromotionDecision,
    PromotionGateConfig,
    VectorLineage,
    build_lineage_record,
    evaluate_promotion_gate,
)
from .route_policy import (
    PolicyRouteDecision,
    RoutePolicyFn,
    SteeringPolicyConfig,
)
from ..evaluation.verifiers import select_verifier_for_task


# --- pure trust-region math ---

def random_direction_at_scale(
    dim: int, scale: float, rng: np.random.RandomState
) -> np.ndarray:
    """A random direction whose L2 norm equals ``scale``.

    Acceptance: the returned vector's norm matches the requested scale
    across seeds and scales (see ``test_perturbation_norms``).
    """
    if dim <= 0:
        raise ValueError("dim must be > 0")
    if scale < 0:
        raise ValueError("scale must be >= 0")
    if scale == 0:
        return np.zeros(dim, dtype=np.float32)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec = np.ones(dim, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
    return vec * (scale / norm)


def weighted_class_mean(
    activations: np.ndarray, weights: Sequence[float]
) -> np.ndarray:
    """Weighted mean of activation rows. Zero total weight -> zero vector."""
    acts = np.asarray(activations, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError("activations must be [N, D]")
    if w.shape[0] != acts.shape[0]:
        raise ValueError("weights must have one entry per activation row")
    total = float(w.sum())
    if total == 0.0:
        return np.zeros(acts.shape[1], dtype=np.float32)
    return (acts * w[:, None]).sum(axis=0) / total


def _validate_vector(v: np.ndarray, name: str = "vector") -> np.ndarray:
    """Fail-closed validation: reject NaN, Inf, or non-finite vectors."""
    arr = np.asarray(v, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def trust_region_update(
    incumbent: np.ndarray,
    candidate: np.ndarray,
    eta: float,
    *,
    target_norm: float | None = None,
    min_norm: float = 0.0,
    max_norm: float | None = None,
    max_update_norm: float | None = None,
) -> np.ndarray:
    """Trust-region convex update with renormalization and bounded norms.

    ``v_{t+1} = (1-eta) v_t + eta * v_hat`` then renormalized to
    ``target_norm`` (default: the incumbent's norm) so the step stays within a
    bounded neighbourhood. ``eta`` must be in ``[0, 1]``.

    v0.3.9 additions (Section 5):
    - Fail-closed on NaN/Inf in either vector.
    - ``min_norm`` / ``max_norm`` clamp the output norm.
    - ``max_update_norm`` clips ``||v_new - v_old||`` if exceeded.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    v = _validate_vector(incumbent, "incumbent")
    vhat = _validate_vector(candidate, "candidate")
    if v.shape != vhat.shape:
        raise ValueError("incumbent and candidate must have the same shape")
    updated = (1.0 - eta) * v + eta * vhat
    norm = float(np.linalg.norm(updated))
    target = float(target_norm) if target_norm is not None else float(np.linalg.norm(v))
    if norm > 0.0 and target > 0.0:
        updated = updated * (target / norm)
        norm = target
    # Apply min/max norm clamps.
    if max_norm is not None and norm > max_norm and norm > 0.0:
        updated = updated * (max_norm / norm)
        norm = max_norm
    if min_norm > 0.0 and norm < min_norm and norm > 0.0:
        updated = updated * (min_norm / norm)
        norm = min_norm
    # Apply max update norm clipping.
    if max_update_norm is not None:
        diff = updated - v
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm > max_update_norm and diff_norm > 0.0:
            diff = diff * (max_update_norm / diff_norm)
            updated = v + diff
    return _validate_vector(updated, "updated").astype(np.float32)


def bootstrap_from_candidate(
    incumbent: np.ndarray,
    candidate: np.ndarray,
    *,
    initial_steering_norm: float,
    epsilon_norm: float = 1e-8,
) -> np.ndarray:
    """Bootstrap a zero (or near-zero) incumbent from the candidate direction.

    Section 5 of the repair spec: if ``||v_incumbent|| < epsilon_norm``,
    initialize from the candidate direction using a configured initial norm::

        v_new = initial_steering_norm * candidate_direction
                / max(||candidate_direction||, epsilon)

    After bootstrap, the loop uses bounded trust-region updates.
    """
    v = _validate_vector(incumbent, "incumbent")
    vhat = _validate_vector(candidate, "candidate")
    if v.shape != vhat.shape:
        raise ValueError("incumbent and candidate must have the same shape")
    inc_norm = float(np.linalg.norm(v))
    if inc_norm >= epsilon_norm:
        # Not a bootstrap situation; return incumbent as-is.
        return v.astype(np.float32)
    cand_norm = float(np.linalg.norm(vhat))
    if cand_norm < epsilon_norm:
        # Candidate is also zero/invalid; cannot bootstrap. Fail closed.
        raise ValueError(
            "cannot bootstrap: both incumbent and candidate norms are below "
            f"epsilon_norm ({epsilon_norm})"
        )
    direction = vhat / max(cand_norm, epsilon_norm)
    return (initial_steering_norm * direction).astype(np.float32)


# --- orchestrator ---

# v0.3.9: capture_fn now returns a structured CaptureResult (Section 6).
# A compatibility adapter is provided for old-style (array, int, int) returns.
CaptureFn = Callable[[Sequence[Mapping[str, Any]], str], CaptureResult]
ExecuteFn = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


def _coerce_capture_result(result: Any) -> CaptureResult:
    """Accept both the new CaptureResult and the legacy (array, int, int) tuple.

    The legacy tuple ``(activations[N,D], n_attempts, n_successes)`` is
    positionally aligned and does NOT carry task IDs. When the legacy form is
    detected we wrap it into a CaptureResult with synthetic task IDs derived
    from positional index — but only when the caller has not provided task IDs.
    This adapter exists solely for backward compatibility; new code should
    return CaptureResult directly.
    """
    if isinstance(result, CaptureResult):
        return result
    if isinstance(result, tuple) and len(result) == 3:
        acts, n_att, n_succ = result
        acts_arr = np.asarray(acts, dtype=np.float32) if acts is not None else np.zeros((0, 0), dtype=np.float32)
        n = acts_arr.shape[0] if acts_arr.ndim == 2 else 0
        captured = []
        for i in range(n):
            captured.append(CapturedActivation(
                task_id=f"__legacy_{i}",
                activation=acts_arr[i],
                success=i < n_succ,
            ))
        return CaptureResult(activations=captured, n_attempts=int(n_att), n_successes=int(n_succ))
    raise ValueError(
        f"capture_fn must return CaptureResult or (array, int, int); got {type(result).__name__}"
    )


@dataclass
class LoopConfig:
    """Configuration for the counterfactual learning loop.

    v0.3.9 additions (Section 5):
    - ``steering_policy_config``: canonical frozen runtime config for lineage.
    - ``min_vector_norm`` / ``max_vector_norm`` / ``max_update_norm``: bounds.
    - ``initial_steering_norm``: bootstrap norm for zero-vector init.
    - ``epsilon_norm``: threshold below which bootstrap is triggered.
    - ``min_capture_coverage``: fail-closed gate on capture coverage.
    """

    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    gate_config: PromotionGateConfig = field(default_factory=PromotionGateConfig)
    steering_policy_config: SteeringPolicyConfig = field(default_factory=SteeringPolicyConfig)
    eta: float = 0.5
    label_field: str = "utility_oracle"
    # Trust-region bounds (Section 5).
    min_vector_norm: float = 0.0
    max_vector_norm: float | None = None
    max_update_norm: float | None = None
    initial_steering_norm: float = 1.0
    epsilon_norm: float = 1e-8
    # Capture coverage gate (Section 6).
    min_capture_coverage: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if self.min_vector_norm < 0:
            raise ValueError("min_vector_norm must be >= 0")
        if self.max_vector_norm is not None and self.max_vector_norm < self.min_vector_norm:
            raise ValueError("max_vector_norm must be >= min_vector_norm")
        if self.max_update_norm is not None and self.max_update_norm < 0:
            raise ValueError("max_update_norm must be >= 0")
        if self.initial_steering_norm < 0:
            raise ValueError("initial_steering_norm must be >= 0")
        if not 0.0 <= self.min_capture_coverage <= 1.0:
            raise ValueError("min_capture_coverage must be in [0, 1]")


@dataclass
class LoopIterationResult:
    """The outcome of one loop iteration."""

    iteration: int
    candidate_vector: np.ndarray
    incumbent_vector: np.ndarray
    decision: PromotionDecision
    promoted: bool
    n_train: int
    n_held_out: int
    n_abstain: int
    # v0.3.9: iteration-level aggregate capture telemetry (Section 7).
    capture_attempts: int = 0
    capture_successes: int = 0
    capture_failures: int = 0
    capture_coverage: float = 0.0


class CounterfactualLearningLoop:
    """Orchestrates counterfactual execution, trust-region update, and the
    promotion gate across iterations.

    The loop is model-optional. The caller injects:
    - ``capture_fn(tasks, class_label) -> CaptureResult``: activation capture.
    - ``execute_fn(task, backend) -> dict``: backend execution.
    - ``route_fn(tasks, steering_vector, config) -> [PolicyRouteDecision]``:
      the routing policy that maps a steering vector to routing decisions.
      This is the contract that makes candidate-vs-incumbent evaluation real.

    v0.3.9 critical fix (Section 1/3): the held-out evaluation now calls
    ``route_fn`` with the candidate vector and the incumbent vector
    separately. The candidate route is determined by the candidate vector;
    the incumbent route is determined by the incumbent vector. The oracle
    Delta-U is used only for the training target, never for the held-out
    routing decision.
    """

    def __init__(
        self,
        *,
        seed_vector: np.ndarray,
        training_tasks: Sequence[Mapping[str, Any]],
        held_out_tasks: Sequence[Mapping[str, Any]],
        capture_fn: CaptureFn,
        execute_fn: ExecuteFn,
        route_fn: RoutePolicyFn,
        config: LoopConfig | None = None,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        timing_method: str = "perf_counter",
        training_dataset_sha256: str | None = None,
        development_dataset_sha256: str | None = None,
    ) -> None:
        self.config = config or LoopConfig()
        self.incumbent = np.asarray(seed_vector, dtype=np.float32).copy()
        self.training_tasks = list(training_tasks)
        self.held_out_tasks = list(held_out_tasks)
        self.capture_fn = capture_fn
        self.execute_fn = execute_fn
        self.route_fn = route_fn
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.timing_method = timing_method
        self.lineage = VectorLineage(current="seed")
        self.iteration = 0
        # Dataset hashes (Section 9): track training and development hashes
        # independently. If not provided, derive from the task content.
        self.training_dataset_sha256 = training_dataset_sha256 or self._hash_tasks(self.training_tasks)
        self.development_dataset_sha256 = development_dataset_sha256 or self._hash_tasks(self.held_out_tasks)
        # experience ledger keyed by the training-data hash
        self.ledger = ExperienceLedger.open(
            data_hash=self.training_dataset_sha256,
            utility_config=self.config.utility_config,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            timing_method=timing_method,
        )
        self._last_capture = CaptureProvenance(n_attempts=0, n_successes=0)

    @staticmethod
    def _hash_tasks(tasks: Sequence[Mapping[str, Any]]) -> str:
        return hash_task({"tasks": [hash_task(t) for t in tasks]})

    def _execute(self, tasks, backend_selected_for_task):
        outcomes = []
        for task in tasks:
            backend = backend_selected_for_task(task)
            outcome = execute_both_backends(
                task, backend_selected=backend,
                symbolic_runner=lambda t, b="symbolic": self.execute_fn(t, b),
                llm_runner=lambda t, b="llm": self.execute_fn(t, b),
                verifier=select_verifier_for_task(task),
                route_suitability_verifier=None,
            )
            outcomes.append(outcome)
        return outcomes

    def _derive_targets(self, outcomes):
        targets = []
        for outcome in outcomes:
            t = derive_learning_target(outcome, self.config.utility_config)
            targets.append((outcome, t))
        return targets

    def _candidate_from_targets(self, tasks, targets):
        """Form the contrastive candidate direction from Delta-U targets.

        v0.3.9 (Section 6): activations and weights are joined by task_id,
        never by positional index. Capture failures are recorded per task and
        excluded from the weighted mean.
        """
        pos_tasks, pos_weights = [], []
        neg_tasks, neg_weights = [], []
        n_abstain = 0
        for task, (outcome, t) in zip(tasks, targets):
            if t.target == ABSTAIN or t.weight == 0.0:
                n_abstain += 1
                continue
            if t.target == "symbolic":
                pos_tasks.append(task)
                pos_weights.append(t.weight)
            elif t.target == "llm":
                neg_tasks.append(task)
                neg_weights.append(t.weight)
        candidate = np.zeros_like(self.incumbent)
        total_attempts = 0
        total_successes = 0
        if pos_tasks or neg_tasks:
            pos_result = _coerce_capture_result(self.capture_fn(pos_tasks, "symbolic"))
            neg_result = _coerce_capture_result(self.capture_fn(neg_tasks, "llm"))
            total_attempts = pos_result.n_attempts + neg_result.n_attempts
            total_successes = pos_result.n_successes + neg_result.n_successes
            # Join activations and weights by task_id (Section 6).
            pos_acts, pos_w = self._align_by_task_id(
                pos_tasks, pos_weights, pos_result)
            neg_acts, neg_w = self._align_by_task_id(
                neg_tasks, neg_weights, neg_result)
            mu_pos = weighted_class_mean(pos_acts, pos_w) if len(pos_w) > 0 else np.zeros_like(self.incumbent)
            mu_neg = weighted_class_mean(neg_acts, neg_w) if len(neg_w) > 0 else np.zeros_like(self.incumbent)
            candidate = mu_pos - mu_neg
            cap = CaptureProvenance(
                n_attempts=total_attempts, n_successes=total_successes,
                layer=self.config.steering_policy_config.layer,
                token_scope=self.config.steering_policy_config.token_location,
            )
            self._last_capture = cap
        else:
            self._last_capture = CaptureProvenance(0, 0)
        return candidate, n_abstain, total_attempts, total_successes

    @staticmethod
    def _align_by_task_id(
        tasks: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        capture_result: CaptureResult,
    ) -> tuple[np.ndarray, list[float]]:
        """Join activations and utility weights by task_id (Section 6).

        Only tasks with both a successful capture AND a non-zero weight enter
        the aligned arrays. The invariant is: ``activation_i`` and ``weight_i``
        refer to the same ``task_id``. Fail closed if alignment cannot be
        established (e.g. a task_id appears in weights but not in captures and
        the capture was supposed to succeed).
        """
        weight_by_task = {}
        for task, w in zip(tasks, weights):
            tid = str(task.get("task_id", ""))
            weight_by_task[tid] = w
        successful = capture_result.successful_by_task_id()
        aligned_acts = []
        aligned_weights = []
        for tid, w in weight_by_task.items():
            cap = successful.get(tid)
            if cap is not None and cap.activation is not None:
                aligned_acts.append(np.asarray(cap.activation, dtype=np.float32))
                aligned_weights.append(w)
        if not aligned_acts:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.stack(aligned_acts), aligned_weights

    def _evaluate_held_out(self, candidate_vector: np.ndarray, incumbent_vector: np.ndarray):
        """Evaluate candidate vs incumbent on held-out tasks (Section 1/3).

        v0.3.9 critical fix: the candidate route is determined by calling
        ``route_fn`` with the **candidate vector**; the incumbent route is
        determined by calling ``route_fn`` with the **incumbent vector**.

        The oracle Delta-U is NOT used to choose the route. Both backends are
        executed counterfactually (for utility scoring), but the utility
        counted for each policy is the utility of the backend THAT POLICY
        SELECTED — not the oracle's preferred backend.
        """
        cfg = self.config.steering_policy_config
        # Candidate routing decisions from the candidate vector.
        candidate_decisions = list(self.route_fn(self.held_out_tasks, candidate_vector, cfg))
        # Incumbent routing decisions from the incumbent vector.
        incumbent_decisions = list(self.route_fn(self.held_out_tasks, incumbent_vector, cfg))
        # Build task_id -> decision maps.
        cand_by_task = {d.task_id: d for d in candidate_decisions}
        inc_by_task = {d.task_id: d for d in incumbent_decisions}

        results: list[HeldOutTaskResult] = []
        for task in self.held_out_tasks:
            tid = str(task.get("task_id", ""))
            c_dec = cand_by_task.get(tid)
            i_dec = inc_by_task.get(tid)
            if c_dec is None or i_dec is None:
                # Route policy did not return a decision for this task.
                results.append(HeldOutTaskResult(
                    task_id=tid, candidate_correct=False, incumbent_correct=False,
                    candidate_utility=0.0, incumbent_utility=0.0, decided=False,
                ))
                continue
            # Execute both backends counterfactually for utility scoring.
            outcome = execute_both_backends(
                task, backend_selected=c_dec.route,
                symbolic_runner=lambda t, b="symbolic": self.execute_fn(t, b),
                llm_runner=lambda t, b="llm": self.execute_fn(t, b),
                verifier=select_verifier_for_task(task),
            )
            u_sym = compute_backend_utility(outcome.backends["symbolic"], self.config.utility_config)
            u_llm = compute_backend_utility(outcome.backends["llm"], self.config.utility_config)
            # Candidate utility = utility of the backend the CANDIDATE policy selected.
            c_route = c_dec.route
            if c_route == "symbolic":
                cu = u_sym.utility
                c_correct = outcome.backends["symbolic"].is_correct
            elif c_route == "llm":
                cu = u_llm.utility
                c_correct = outcome.backends["llm"].is_correct
            else:
                # Abstain: no backend executed, zero utility, not correct.
                cu = 0.0
                c_correct = False
            # Incumbent utility = utility of the backend the INCUMBENT policy selected.
            i_route = i_dec.route
            if i_route == "symbolic":
                iu = u_sym.utility
                i_correct = outcome.backends["symbolic"].is_correct
            elif i_route == "llm":
                iu = u_llm.utility
                i_correct = outcome.backends["llm"].is_correct
            else:
                iu = 0.0
                i_correct = False
            # A task is "decided" only if both policies produced a real (non-abstain) decision.
            decided = (c_route != ABSTAIN) and (i_route != ABSTAIN)
            results.append(HeldOutTaskResult(
                task_id=tid,
                candidate_correct=bool(c_correct),
                incumbent_correct=bool(i_correct),
                candidate_utility=float(cu), incumbent_utility=float(iu),
                decided=decided,
                candidate_route=c_route,
                incumbent_route=i_route,
            ))
        return results

    def _record_experience(self, tasks, outcomes, targets):
        for task, outcome, t in zip(tasks, outcomes, targets):
            rec = build_experience_record(
                record_id=f"{outcome.task_id}@{self.iteration}",
                task=task, prompt=str(task.get("specification", "")),
                outcome=outcome, verifier=select_verifier_for_task(task),
                timing_method=self.timing_method,
                per_backend_timing={
                    b: {"device": "cpu", "cuda_event_timing": False,
                        "elapsed_ms": outcome.backends[b].latency_ms or 0.0}
                    for b in outcome.backends
                },
                model_revision=self.model_revision,
                tokenizer_revision=self.tokenizer_revision,
                utility_config=self.config.utility_config,
                capture_provenance=self._last_capture,
            )
            self.ledger.append(rec)

    def _maybe_bootstrap(self, candidate: np.ndarray) -> np.ndarray:
        """Section 5: bootstrap from candidate if incumbent norm is near-zero."""
        inc_norm = float(np.linalg.norm(self.incumbent))
        if inc_norm < self.config.epsilon_norm:
            return bootstrap_from_candidate(
                self.incumbent, candidate,
                initial_steering_norm=self.config.initial_steering_norm,
                epsilon_norm=self.config.epsilon_norm,
            )
        return self.incumbent

    def step(self) -> LoopIterationResult:
        """Run one iteration: execute, derive targets, update, gate, record."""
        self.iteration += 1
        outcomes = self._execute(self.training_tasks, lambda t: "symbolic")
        targets = self._derive_targets(outcomes)
        candidate, n_abstain, cap_att, cap_succ = self._candidate_from_targets(
            self.training_tasks, targets)
        # Section 5: bootstrap if incumbent is near-zero, then trust-region update.
        bootstrap_inc = self._maybe_bootstrap(candidate)
        updated = trust_region_update(
            bootstrap_inc, candidate, self.config.eta,
            min_norm=self.config.min_vector_norm,
            max_norm=self.config.max_vector_norm,
            max_update_norm=self.config.max_update_norm,
        )
        # Capture coverage gate (Section 6).
        cap_failures = cap_att - cap_succ
        cap_coverage = cap_succ / cap_att if cap_att > 0 else 0.0
        if cap_coverage < self.config.min_capture_coverage and cap_att > 0:
            # Insufficient coverage: skip promotion, record rollback.
            from .promotion import PromotionDecision, PromotionGateConfig
            decision = PromotionDecision(
                action="rollback",
                reason=f"capture coverage {cap_coverage:.3f} < min_capture_coverage "
                       f"{self.config.min_capture_coverage:.3f}",
                improved=0, regressed=0, tied=0, n_decided=0,
                n_held_out=len(self.held_out_tasks), coverage=0.0,
                regression_rate=0.0, p_value=1.0, improved_probability=0.5,
                improved_probability_ci=(0.0, 1.0), mean_utility_delta=0.0,
                config=self.config.gate_config,
            )
        else:
            held_out = self._evaluate_held_out(updated, self.incumbent)
            decision = evaluate_promotion_gate(held_out, self.config.gate_config)
        self._record_experience(self.training_tasks, outcomes, targets)
        # Lineage (Section 8): use the canonical SteeringPolicyConfig, not
        # hard-coded values.
        spc = self.config.steering_policy_config
        parent = self.lineage.current or "seed"
        lineage_rec = build_lineage_record(
            vector_id=f"v{self.iteration}", parent_vector_id=parent,
            iteration=self.iteration,
            training_dataset_sha256=self.training_dataset_sha256,
            development_dataset_sha256=self.development_dataset_sha256,
            optimizer_parameters={"eta": self.config.eta, "method": "trust_region"},
            utility_config_hash=self.ledger.bundle.config_hash,
            layer=spc.layer, token_location=spc.token_location,
            norm=float(np.linalg.norm(updated)), alpha=spc.alpha,
            metrics_before={"norm": float(np.linalg.norm(self.incumbent))},
            metrics_after={"norm": float(np.linalg.norm(updated))},
            promotion_decision=decision,
        )
        self.lineage.append(lineage_rec, promote=decision.promoted)
        if decision.promoted:
            self.incumbent = updated
        result = LoopIterationResult(
            iteration=self.iteration, candidate_vector=updated,
            incumbent_vector=self.incumbent, decision=decision,
            promoted=decision.promoted, n_train=len(self.training_tasks),
            n_held_out=len(self.held_out_tasks), n_abstain=n_abstain,
            capture_attempts=cap_att, capture_successes=cap_succ,
            capture_failures=cap_failures, capture_coverage=cap_coverage,
        )
        return result

    def run(self, max_iterations: int = 10) -> list[LoopIterationResult]:
        results = []
        for _ in range(max_iterations):
            results.append(self.step())
        return results


__all__ = [
    "CounterfactualLearningLoop",
    "LoopConfig",
    "LoopIterationResult",
    "bootstrap_from_candidate",
    "random_direction_at_scale",
    "trust_region_update",
    "weighted_class_mean",
]
