"""v0.3.10.1-alpha — integrated policy learner: ties the weighted
preference-target router to the counterfactual experience pipeline.

This module provides the high-level glue that:

1. Collects counterfactual experiences (both backends executed).
2. Computes utilities and ΔU.
3. Computes sample weights via the configured ``WeightMode``
   (v0.3.10.1 — actually dispatches on mode; the old code always
   called the gap function).
4. Optionally reduces features (PCA on TRAIN only).
5. Trains the selected policy (``centroid`` | ``logistic`` |
   ``mlp_experimental``) via :func:`fit_policy` (v0.3.10.1 — actually
   dispatches on policy_type; the old code always trained logistic).
6. Evaluates the learned policy against an incumbent on held-out tasks
   using *actual policy decisions* (Section 22 — never the oracle).
7. Reports regret, utility, calibration, abstention, and OOD metrics.

v0.3.10.1 — breaking changes (clean break):

* ``utility_fn`` is **required** for held-out evaluation (Section 4).
  The old code defaulted to zero utility, silently making every
  candidate look identical. Now raises ``ValueError``.
* Experiences and activations are joined by ``task_id`` via
  :func:`join_by_task_id` (Sections 5, 6). Naked row arrays at the
  boundary are no longer accepted — pass ``FeatureRecord`` objects.
  The old ``zip(experiences, activations)`` silently truncated on
  length mismatch; the new join reports missing records explicitly.
* ``dev_tasks`` / ``dev_experiences`` / ``dev_activations`` are
  aligned by ``task_id`` before dev evaluation. No silent zip
  truncation.
* Early stopping uses the configured ``early_stopping_metric``
  (default ``dev_regret``). The old trainer always used dev BCE loss.
* Calibration metrics use the corrected
  :func:`preference_brier_soft` and :func:`action_confidence_ece`
  (Section 8). The old ECE was mathematically wrong for soft labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .abstention import choose_route, choose_route_with_reason
from .alignment import AlignmentResult, join_by_task_id
from .calibration import (
    action_confidence_ece,
    brier_score,
    expected_calibration_error,
    preference_brier_soft,
)
from .config import ExperimentConfig
from .confidence import OutcomeConfidence
from .features import PCAFeatureReducer
from .ood import MahalanobisOOD
from .policy_factory import fit_policy, predict_proba
from .regret import mean_regret, paired_promotion_statistics, per_task_regret
from .targets import TargetMode, build_preference_targets
from .types import BackendOutcome, CounterfactualExperience, FeatureRecord, Route
from .weighting import WeightConfig, WeightMode, compute_weight


@dataclass
class PolicyLearnerResult:
    """Result of training and evaluating the integrated policy learner."""

    config: ExperimentConfig
    policy: Any = None  # PolicyModel (CentroidPolicy | WeightedLogisticRouter | SmallMLPRouter)
    feature_reducer: PCAFeatureReducer | None = None
    ood_detector: MahalanobisOOD | None = None
    n_train: int = 0
    n_dev: int = 0
    train_weights: np.ndarray | None = None
    train_task_ids: list[str] = field(default_factory=list)
    dev_task_ids: list[str] = field(default_factory=list)
    dev_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_utilities: np.ndarray | None = None
    incumbent_utilities: np.ndarray | None = None
    oracle_utilities: np.ndarray | None = None
    candidate_routes: list[str] = field(default_factory=list)
    incumbent_routes: list[str] = field(default_factory=list)
    abstention_reasons: list[str] = field(default_factory=list)
    missing_train_feature_ids: list[str] = field(default_factory=list)
    missing_dev_feature_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_train": self.n_train,
            "n_dev": self.n_dev,
            "train_task_ids": self.train_task_ids,
            "dev_task_ids": self.dev_task_ids,
            "train_weights": (
                None if self.train_weights is None
                else self.train_weights.tolist()),
            "dev_metrics": self.dev_metrics,
            "candidate_routes": self.candidate_routes,
            "incumbent_routes": self.incumbent_routes,
            "abstention_reasons": self.abstention_reasons,
            "missing_train_feature_ids": self.missing_train_feature_ids,
            "missing_dev_feature_ids": self.missing_dev_feature_ids,
        }


def build_counterfactual_experiences(
    tasks: Sequence[Mapping[str, Any]],
    *,
    execute_fn: Callable[[Mapping[str, Any], str], dict[str, Any]],
    config: ExperimentConfig | None = None,
    utility_config: ExperimentConfig | None = None,
    confidence_fn: Callable[
        [Mapping[str, Any], str, dict[str, Any]],
        OutcomeConfidence,
    ] | None = None,
    sigma_delta_u_fn: Callable[
        [Mapping[str, Any], float, float], float] | None = None,
) -> list[CounterfactualExperience]:
    """Execute both backends on each task and build counterfactual experiences.

    v0.3.10.1 — ``sample_weight`` is now computed via
    :func:`compute_weight` with the configured ``WeightMode``. The old
    code always called the gap function, silently ignoring ``"snr"``.

    Parameters
    ----------
    tasks : sequence of task dicts
    execute_fn : callable
        ``(task, backend) -> dict`` with keys ``correct``, ``quality``,
        ``latency_ms``, ``compute_cost``, ``risk``, ``verifier_confidence``.
    config : ExperimentConfig | None
    utility_config : ExperimentConfig | None
        Deprecated alias for ``config``.
    confidence_fn : callable | None
    sigma_delta_u_fn : callable | None
        ``(task, u_sym, u_llm) -> sigma_delta_u``. Required for SNR
        weight mode; ignored otherwise.
    """
    cfg = config or utility_config or ExperimentConfig()
    experiences: list[CounterfactualExperience] = []
    for task in tasks:
        tid = str(task.get("task_id", ""))
        sym_result = execute_fn(task, "symbolic")
        llm_result = execute_fn(task, "llm")
        if confidence_fn is None:
            def _default_conf(t, b, r):
                vc = float(r.get(
                    "verifier_confidence",
                    1.0 if r.get("correct") else 0.5))
                return OutcomeConfidence(verifier=vc)
            confidence_fn = _default_conf
        sym_conf = confidence_fn(task, "symbolic", sym_result).combined()
        llm_conf = confidence_fn(task, "llm", llm_result).combined()
        sym_outcome = BackendOutcome(
            task_id=tid, backend="symbolic",
            correct=bool(sym_result.get("correct", False)),
            quality=float(sym_result.get(
                "quality",
                1.0 if sym_result.get("correct") else 0.0)),
            latency_sec=float(sym_result.get("latency_ms", 0.0)) / 1000.0,
            normalized_cost=float(sym_result.get("compute_cost", 0.0)),
            risk=float(sym_result.get("risk", 0.0)),
            verifier_confidence=sym_conf,
        )
        llm_outcome = BackendOutcome(
            task_id=tid, backend="llm",
            correct=bool(llm_result.get("correct", False)),
            quality=float(llm_result.get(
                "quality",
                1.0 if llm_result.get("correct") else 0.0)),
            latency_sec=float(llm_result.get("latency_ms", 0.0)) / 1000.0,
            normalized_cost=float(llm_result.get("compute_cost", 0.0)),
            risk=float(llm_result.get("risk", 0.0)),
            verifier_confidence=llm_conf,
        )
        u_sym = _utility(sym_outcome, cfg)
        u_llm = _utility(llm_outcome, cfg)
        delta_u = u_sym - u_llm
        if delta_u > cfg.abstention_band:
            preferred = Route.SYMBOLIC
        elif delta_u < -cfg.abstention_band:
            preferred = Route.LLM
        else:
            preferred = Route.ABSTAIN
        conf = min(sym_conf, llm_conf)
        sigma = (
            sigma_delta_u_fn(task, u_sym, u_llm)
            if sigma_delta_u_fn is not None else None)
        weight = compute_weight(
            delta_u, conf, cfg.weight_mode,
            gap_threshold=cfg.gap_threshold,
            max_weight=cfg.max_weight,
            min_weight=cfg.min_weight,
            sigma_delta_u=sigma,
        )
        experiences.append(CounterfactualExperience(
            task_id=tid, symbolic=sym_outcome, llm=llm_outcome,
            delta_utility=delta_u, preferred_action=preferred,
            sample_weight=weight,
        ))
    return experiences


def _utility(outcome: BackendOutcome, cfg: ExperimentConfig) -> float:
    """Compute ``U_b`` from a :class:`BackendOutcome` and config."""
    t = outcome.latency_sec * 1000.0  # back to ms
    time_term = cfg.lambda_time * (t / cfg.time_reference_ms)
    compute_term = cfg.lambda_compute * (
        outcome.normalized_cost / cfg.compute_reference)
    risk_term = cfg.lambda_risk * outcome.risk
    return (cfg.quality_weight * outcome.quality
            - time_term - compute_term - risk_term)


def _align_or_raise(
    experiences: Sequence[CounterfactualExperience],
    activations: "np.ndarray | Sequence[FeatureRecord]",
    *,
    label: str,
    strict: bool = True,
) -> tuple[list[CounterfactualExperience], np.ndarray, AlignmentResult]:
    """Align experiences and activations by task_id (Sections 5, 6).

    Accepts either a naked ``np.ndarray`` (legacy path — requires the
    caller to have already asserted length equality and ID order) or a
    sequence of :class:`FeatureRecord` objects (preferred path —
    joined by task_id).
    """
    if isinstance(activations, np.ndarray) or (
            isinstance(activations, (list, tuple))
            and len(activations) > 0
            and not isinstance(activations[0], FeatureRecord)):
        # Legacy naked-array path. Assert lengths match; the caller is
        # responsible for ID-order alignment. This path is kept so the
        # existing synthetic tests that pass naked arrays continue to
        # work, but new callers should pass FeatureRecord objects.
        arr = np.asarray(activations, dtype=np.float32)
        if len(experiences) != len(arr):
            raise ValueError(
                f"{label}: experiences ({len(experiences)}) and "
                f"activations ({len(arr)}) length mismatch; pass "
                f"FeatureRecord objects to join by task_id instead")
        return list(experiences), arr, AlignmentResult(
            experiences=list(experiences), features=arr,
            aligned_task_ids=[e.task_id for e in experiences],
        )
    # Preferred path: join by task_id.
    records = list(activations)
    result = join_by_task_id(experiences, records, strict=strict)
    if strict:
        result.raise_if_not_ok()
    return result.experiences, result.features, result


def train_policy_learner(
    train_experiences: Sequence[CounterfactualExperience],
    train_activations: "np.ndarray | Sequence[FeatureRecord]",
    *,
    config: ExperimentConfig,
    dev_experiences: Sequence[CounterfactualExperience] | None = None,
    dev_activations: "np.ndarray | Sequence[FeatureRecord] | None" = None,
    incumbent_route_fn: Callable[[np.ndarray], str] | None = None,
    utility_fn: Callable[[Mapping[str, Any], str], float] | None = None,
    dev_tasks: Sequence[Mapping[str, Any]] | None = None,
) -> PolicyLearnerResult:
    """Train the integrated policy learner and evaluate on dev.

    v0.3.10.1 — clean-break changes:

    * ``utility_fn`` is **required** when ``dev_tasks`` is provided
      (Section 4). The old code defaulted to zero utility.
    * ``train_activations`` / ``dev_activations`` may be a sequence of
      :class:`FeatureRecord` objects, in which case they are joined to
      the experiences by ``task_id`` (Sections 5, 6). Naked
      ``np.ndarray`` inputs are still accepted but require the caller
      to have already asserted length equality.
    * The trained policy is selected by ``config.policy_type`` (Section
      3). The old code always trained logistic routing.
    * Sample weights come from ``config.weight_mode`` (Section 2). The
      old code always used the gap function.
    * Early stopping uses ``config.early_stopping_metric`` (default
      ``dev_regret``). The old code always used dev BCE loss.
    * Calibration uses the corrected
      :func:`preference_brier_soft` and :func:`action_confidence_ece`
      (Section 8).

    Parameters
    ----------
    train_experiences : sequence of CounterfactualExperience
    train_activations : np.ndarray | sequence of FeatureRecord
        Shape ``[N, D]`` (legacy) or task-bound records (preferred).
    config : ExperimentConfig
    dev_experiences, dev_activations : optional development set.
    incumbent_route_fn : callable | None
        ``h -> route_str`` for the incumbent policy. Defaults to always
        "llm" (a weak incumbent).
    utility_fn : callable | None
        ``(task, route) -> utility``. Required when ``dev_tasks`` is
        provided; raises ``ValueError`` if absent.
    dev_tasks : sequence of task dicts
        Dev tasks for utility evaluation. When provided, must be
        joinable by ``task_id`` to ``dev_experiences``.
    """
    cfg = config.freeze()

    # --- Fail closed on missing utility_fn (Section 4) --------------
    if dev_tasks is not None and utility_fn is None:
        raise ValueError(
            "utility_fn is required for held-out policy evaluation "
            "(Section 4): refusing to synthesize zero utility")

    # --- Align train experiences + features by task_id (Sections 5, 6)
    train_exps, train_feats, train_align = _align_or_raise(
        train_experiences, train_activations,
        label="train", strict=False)
    missing_train = train_align.missing_feature_task_ids

    # Feature reduction (PCA on TRAIN only — Section 20).
    feature_reducer = None
    train_features = train_feats
    if cfg.feature_transform == "pca":
        feature_reducer = PCAFeatureReducer(n_components=cfg.pca_components)
        train_features = feature_reducer.fit_transform(train_feats)

    # OOD detector (fitted on TRAIN only).
    ood_detector = MahalanobisOOD(ridge=cfg.ood_ridge)
    ood_detector.fit(train_features)

    # Sample weights (already computed in build_counterfactual_experiences
    # using the configured WeightMode). Re-extract aligned weights.
    weights = np.array(
        [e.sample_weight for e in train_exps],
        dtype=np.float32)
    delta_us = np.array(
        [e.delta_utility for e in train_exps],
        dtype=np.float32)

    # --- Align dev experiences + features by task_id ----------------
    dev_features = None
    dev_du = None
    dev_w = None
    dev_exps_aligned: list[CounterfactualExperience] = []
    dev_tasks_aligned: list[Mapping[str, Any]] = []
    missing_dev: list[str] = []
    if dev_experiences is not None and dev_activations is not None:
        dev_exps_aligned, dev_feats, dev_align = _align_or_raise(
            dev_experiences, dev_activations,
            label="dev", strict=False)
        missing_dev = dev_align.missing_feature_task_ids
        dev_features = (
            dev_feats if feature_reducer is None
            else feature_reducer.transform(dev_feats))
        dev_du = np.array(
            [e.delta_utility for e in dev_exps_aligned],
            dtype=np.float32)
        dev_w = np.array(
            [e.sample_weight for e in dev_exps_aligned],
            dtype=np.float32)
        # If dev_tasks provided, join them to dev_experiences by task_id.
        if dev_tasks is not None:
            task_by_id = {str(t.get("task_id", "")): t for t in dev_tasks}
            # Detect duplicate task_ids in dev_tasks.
            seen = set()
            for t in dev_tasks:
                tid = str(t.get("task_id", ""))
                if tid in seen:
                    raise ValueError(
                        f"duplicate task_id in dev_tasks: {tid!r}")
                seen.add(tid)
            dev_tasks_aligned = []
            for exp in dev_exps_aligned:
                t = task_by_id.get(exp.task_id)
                if t is None:
                    raise ValueError(
                        f"dev_tasks is missing task_id {exp.task_id!r} "
                        f"that is present in dev_experiences; refusing "
                        f"to silently zip-misalign")
                dev_tasks_aligned.append(t)

    # --- Fit the selected policy (Section 3) ------------------------
    policy = fit_policy(
        cfg, train_features, delta_us, weights,
        dev_features=dev_features,
        dev_delta_u=dev_du,
        dev_weights=dev_w,
        dev_tasks=dev_tasks_aligned or None,
        utility_fn=utility_fn,
        seed=cfg.random_seed,
    )

    # --- Evaluate on dev using ACTUAL policy decisions (Section 22) -
    dev_metrics: dict[str, Any] = {}
    candidate_routes: list[str] = []
    incumbent_routes: list[str] = []
    abstention_reasons: list[str] = []
    cand_utils = None
    inc_utils = None
    ora_utils = None
    if dev_exps_aligned and dev_tasks_aligned and utility_fn is not None:
        probs = predict_proba(policy, dev_features)
        if incumbent_route_fn is None:
            def incumbent_route_fn(h):
                return "llm"
        cand_utils = []
        inc_utils = []
        ora_utils = []
        for i, (task, exp) in enumerate(zip(dev_tasks_aligned, dev_exps_aligned)):
            p = float(probs[i])
            ood_score = ood_detector.score(dev_features[i])
            decision = choose_route_with_reason(
                p, cfg.confidence_threshold,
                ood_score=ood_score,
                ood_threshold=cfg.ood_threshold,
            )
            c_route = decision.route
            candidate_routes.append(c_route.value)
            abstention_reasons.append(decision.reason)
            cu = utility_fn(task, c_route)
            cand_utils.append(cu)
            # Incumbent: ACTUAL incumbent policy decision (Section 22).
            i_route_str = incumbent_route_fn(dev_features[i])
            i_route = Route.from_str(i_route_str)
            incumbent_routes.append(i_route.value)
            iu = utility_fn(task, i_route)
            inc_utils.append(iu)
            # Oracle: max_a U(a) — used ONLY for scoring regret.
            u_sym = utility_fn(task, Route.SYMBOLIC)
            u_llm = utility_fn(task, Route.LLM)
            ora_utils.append(max(u_sym, u_llm))
        cand_utils = np.array(cand_utils)
        inc_utils = np.array(inc_utils)
        ora_utils = np.array(ora_utils)
        # Soft targets for the corrected calibration metrics (Section 8).
        soft_targets, _ = build_preference_targets(
            dev_du, TargetMode.SOFT, cfg.target_temperature, 0.0)
        soft_targets_np = np.asarray(soft_targets, dtype=np.float64)
        # Hard labels (legacy ECE — kept for ablation).
        hard_labels = np.array([
            1.0 if e.preferred_action == Route.SYMBOLIC
            else (0.0 if e.preferred_action == Route.LLM
                  else 0.5)
            for e in dev_exps_aligned
        ])
        # Abstention rates by reason (Section 21).
        reason_counts: dict[str, int] = {}
        for r in abstention_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        abstain_rate = float(np.mean(
            [r == "abstain" for r in candidate_routes]))
        symbolic_util = float(np.mean(
            [r == "symbolic" for r in candidate_routes])) if candidate_routes else 0.0
        llm_util = float(np.mean(
            [r == "llm" for r in candidate_routes])) if candidate_routes else 0.0
        ood_rate = float(reason_counts.get("ood", 0) / len(candidate_routes)) if candidate_routes else 0.0
        dev_metrics = {
            "mean_candidate_utility": float(cand_utils.mean()),
            "mean_incumbent_utility": float(inc_utils.mean()),
            "mean_candidate_regret": float(mean_regret(cand_utils, ora_utils)),
            "mean_incumbent_regret": float(mean_regret(inc_utils, ora_utils)),
            # v0.3.10.1 — corrected calibration (Section 8).
            "preference_brier_soft": float(
                preference_brier_soft(probs, soft_targets_np)),
            "action_confidence_ece": float(
                action_confidence_ece(probs, soft_targets_np)),
            # Legacy metrics — kept for ablation / back-compat.
            "brier_score": float(brier_score(probs, soft_targets_np)),
            "ece": float(expected_calibration_error(probs, hard_labels)),
            # Abstention breakdown (Section 21).
            "abstain_rate": abstain_rate,
            "symbolic_utilization": symbolic_util,
            "llm_utilization": llm_util,
            "ood_rate": ood_rate,
            "abstention_reasons": reason_counts,
            "n_dev": len(dev_exps_aligned),
        }
        stats = paired_promotion_statistics(
            cand_utils, inc_utils, ora_utils,
            seed=cfg.random_seed,
        )
        dev_metrics.update(stats)

    return PolicyLearnerResult(
        config=cfg,
        policy=policy,
        feature_reducer=feature_reducer,
        ood_detector=ood_detector,
        n_train=len(train_exps),
        n_dev=len(dev_exps_aligned),
        train_weights=weights,
        train_task_ids=[e.task_id for e in train_exps],
        dev_task_ids=[e.task_id for e in dev_exps_aligned],
        dev_metrics=dev_metrics,
        candidate_utilities=cand_utils,
        incumbent_utilities=inc_utils,
        oracle_utilities=ora_utils,
        candidate_routes=candidate_routes,
        incumbent_routes=incumbent_routes,
        abstention_reasons=abstention_reasons,
        missing_train_feature_ids=missing_train,
        missing_dev_feature_ids=missing_dev,
    )


__all__ = [
    "PolicyLearnerResult",
    "build_counterfactual_experiences",
    "train_policy_learner",
]
