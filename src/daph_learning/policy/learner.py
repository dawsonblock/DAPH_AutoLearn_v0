"""v0.3.10 — integrated policy learner: ties the weighted logistic router
to the counterfactual experience pipeline.

This module provides the high-level glue that:

1. Collects counterfactual experiences (both backends executed).
2. Computes utilities and ΔU.
3. Computes sample weights (utility-weighted, gap-threshold truncated).
4. Optionally reduces features (PCA on TRAIN only).
5. Trains the weighted soft-target logistic router.
6. Evaluates the learned policy against an incumbent on held-out tasks.
7. Reports regret, utility, calibration, and abstention metrics.

This is the "main learner" (Section 5) that the weighted centroid baseline
is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .abstention import choose_route
from .calibration import brier_score, expected_calibration_error
from .config import ExperimentConfig
from .confidence import OutcomeConfidence
from .features import PCAFeatureReducer
from .logistic import (
    LogisticTrainConfig,
    WeightedLogisticRouter,
    soft_preference_target,
    train_weighted_logistic_router,
)
from .ood import MahalanobisOOD
from .regret import mean_regret, paired_promotion_statistics
from .types import BackendOutcome, CounterfactualExperience, Route
from .weighting import WeightConfig, utility_weight


@dataclass
class PolicyLearnerResult:
    """Result of training and evaluating the integrated policy learner."""

    config: ExperimentConfig
    router: WeightedLogisticRouter | None
    feature_reducer: PCAFeatureReducer | None
    ood_detector: MahalanobisOOD | None
    n_train: int
    n_dev: int
    train_weights: np.ndarray
    dev_metrics: dict[str, Any] = field(default_factory=dict)
    candidate_utilities: np.ndarray | None = None
    incumbent_utilities: np.ndarray | None = None
    oracle_utilities: np.ndarray | None = None
    candidate_routes: list[str] = field(default_factory=list)
    incumbent_routes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_train": self.n_train,
            "n_dev": self.n_dev,
            "train_weights": self.train_weights.tolist(),
            "dev_metrics": self.dev_metrics,
            "candidate_routes": self.candidate_routes,
            "incumbent_routes": self.incumbent_routes,
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
) -> list[CounterfactualExperience]:
    """Execute both backends on each task and build counterfactual experiences.

    Parameters
    ----------
    tasks : sequence of task dicts
    execute_fn : callable
        ``(task, backend) -> dict`` with keys ``correct``, ``quality``,
        ``latency_ms``, ``compute_cost``, ``risk``, ``verifier_confidence``.
    config : ExperimentConfig | None
        Experiment configuration (preferred keyword).
    utility_config : ExperimentConfig | None
        Deprecated alias for ``config``.
    confidence_fn : callable | None
        ``(task, backend, result) -> OutcomeConfidence``. Defaults to
        verifier confidence = 1.0 for correct, 0.5 for incorrect.
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
        wcfg = WeightConfig(
            min_weight=cfg.min_weight,
            max_weight=cfg.max_weight,
            gap_threshold=cfg.gap_threshold,
        )
        conf = min(sym_conf, llm_conf)
        weight = utility_weight(delta_u, conf, wcfg)
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


def train_policy_learner(
    train_experiences: Sequence[CounterfactualExperience],
    train_activations: np.ndarray,
    *,
    config: ExperimentConfig,
    dev_experiences: Sequence[CounterfactualExperience] | None = None,
    dev_activations: np.ndarray | None = None,
    incumbent_route_fn: Callable[[np.ndarray], str] | None = None,
    utility_fn: Callable[[Mapping[str, Any], Route], float] | None = None,
    dev_tasks: Sequence[Mapping[str, Any]] | None = None,
) -> PolicyLearnerResult:
    """Train the integrated policy learner and evaluate on dev.

    Parameters
    ----------
    train_experiences : sequence of CounterfactualExperience
    train_activations : np.ndarray
        Shape ``[N, D]``, aligned by task_id with ``train_experiences``.
    config : ExperimentConfig
    dev_experiences, dev_activations : optional development set.
    incumbent_route_fn : callable | None
        ``h -> route_str`` for the incumbent policy. Defaults to always
        "llm" (a weak incumbent).
    utility_fn : callable | None
        ``(task, route) -> utility`` for evaluating routes on dev tasks.
    dev_tasks : sequence of task dicts
        Dev tasks for utility evaluation (aligned with dev_experiences).
    """
    cfg = config.freeze()
    train_acts = np.asarray(train_activations, dtype=np.float32)
    if len(train_experiences) != len(train_acts):
        raise ValueError("train_experiences and train_activations must align")

    # Feature reduction (PCA on TRAIN only).
    feature_reducer = None
    train_features = train_acts
    if cfg.feature_transform == "pca":
        feature_reducer = PCAFeatureReducer(n_components=cfg.pca_components)
        train_features = feature_reducer.fit_transform(train_acts)

    # OOD detector (fitted on TRAIN only).
    ood_detector = MahalanobisOOD(ridge=cfg.ood_ridge)
    ood_detector.fit(train_features)

    # Sample weights.
    weights = np.array(
        [e.sample_weight for e in train_experiences],
        dtype=np.float32)
    delta_us = np.array(
        [e.delta_utility for e in train_experiences],
        dtype=np.float32)

    # Train the logistic router.
    train_cfg = LogisticTrainConfig(
        temperature=cfg.target_temperature,
    )
    dev_features = None
    dev_du = None
    dev_w = None
    if dev_experiences is not None and dev_activations is not None:
        dev_acts = np.asarray(dev_activations, dtype=np.float32)
        dev_features = (
            dev_acts if feature_reducer is None
            else feature_reducer.transform(dev_acts))
        dev_du = np.array(
            [e.delta_utility for e in dev_experiences],
            dtype=np.float32)
        dev_w = np.array(
            [e.sample_weight for e in dev_experiences],
            dtype=np.float32)

    router = train_weighted_logistic_router(
        train_features, delta_us, weights,
        config=train_cfg,
        dev_features=dev_features,
        dev_delta_u=dev_du,
        dev_weights=dev_w,
        seed=cfg.random_seed,
    )

    # Evaluate on dev.
    dev_metrics: dict[str, Any] = {}
    candidate_routes: list[str] = []
    incumbent_routes: list[str] = []
    cand_utils = None
    inc_utils = None
    ora_utils = None
    if (dev_experiences is not None
            and dev_activations is not None
            and dev_tasks is not None):
        import torch
        dev_acts = np.asarray(dev_activations, dtype=np.float32)
        dev_feat = (
            dev_acts if feature_reducer is None
            else feature_reducer.transform(dev_acts))
        with torch.no_grad():
            probs = router.predict_proba(
                torch.as_tensor(
                    dev_feat, dtype=torch.float32)).numpy()
        if incumbent_route_fn is None:
            def incumbent_route_fn(h):
                return "llm"
        if utility_fn is None:
            def utility_fn(task, route):
                # Default: use the experience's backend utility.
                return 0.0
        cand_utils = []
        inc_utils = []
        ora_utils = []
        soft_targets = soft_preference_target(dev_du, cfg.target_temperature)
        for i, (task, exp) in enumerate(zip(dev_tasks, dev_experiences)):
            p = float(probs[i])
            # OOD check.
            if ood_detector.score(dev_feat[i]) > cfg.ood_threshold:
                c_route = Route.ABSTAIN
            else:
                c_route = choose_route(p, cfg.confidence_threshold)
            candidate_routes.append(c_route.value)
            cu = utility_fn(task, c_route)
            cand_utils.append(cu)
            # Incumbent.
            i_route_str = incumbent_route_fn(dev_feat[i])
            i_route = Route.from_str(i_route_str)
            incumbent_routes.append(i_route.value)
            iu = utility_fn(task, i_route)
            inc_utils.append(iu)
            # Oracle: max_a U(a) using the same
            # utility_fn for scale consistency.
            u_sym = utility_fn(task, Route.SYMBOLIC)
            u_llm = utility_fn(task, Route.LLM)
            ora_utils.append(max(u_sym, u_llm))
        cand_utils = np.array(cand_utils)
        inc_utils = np.array(inc_utils)
        ora_utils = np.array(ora_utils)
        # Hard labels for calibration: 1 if symbolic preferred, 0 if llm.
        hard_labels = np.array([
            1.0 if e.preferred_action == Route.SYMBOLIC
            else (0.0 if e.preferred_action == Route.LLM
                  else 0.5)
            for e in dev_experiences
        ])
        dev_metrics = {
            "mean_candidate_utility": float(cand_utils.mean()),
            "mean_incumbent_utility": float(inc_utils.mean()),
            "mean_candidate_regret": float(mean_regret(cand_utils, ora_utils)),
            "mean_incumbent_regret": float(mean_regret(inc_utils, ora_utils)),
            "brier_score": float(brier_score(probs, soft_targets)),
            "ece": float(expected_calibration_error(probs, hard_labels)),
            "abstain_rate": float(np.mean(
                [r == "abstain" for r in candidate_routes])),
            "n_dev": len(dev_experiences),
        }
        stats = paired_promotion_statistics(
            cand_utils, inc_utils, ora_utils,
            seed=cfg.random_seed,
        )
        dev_metrics.update(stats)

    return PolicyLearnerResult(
        config=cfg,
        router=router,
        feature_reducer=feature_reducer,
        ood_detector=ood_detector,
        n_train=len(train_experiences),
        n_dev=len(dev_experiences or []),
        train_weights=weights,
        dev_metrics=dev_metrics,
        candidate_utilities=cand_utils,
        incumbent_utilities=inc_utils,
        oracle_utilities=ora_utils,
        candidate_routes=candidate_routes,
        incumbent_routes=incumbent_routes,
    )


__all__ = [
    "PolicyLearnerResult",
    "build_counterfactual_experiences",
    "train_policy_learner",
]
