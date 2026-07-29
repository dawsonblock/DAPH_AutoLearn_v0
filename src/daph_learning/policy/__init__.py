"""v0.3.10.1-alpha — learned policy package: counterfactual utility
learning, weighted latent routing, calibrated abstention, OOD
detection, and causal intervention validation.

This package implements the v0.3.10.1 architecture shift from
"weighted steering-vector extractor" to "counterfactual compute-selection
learner". The weighted centroid remains as a transparent baseline
(:mod:`daph_learning.policy.centroid`); the primary policy learner is the
weighted soft/hard-target logistic router
(:mod:`daph_learning.policy.logistic`).

Submodules
----------
``types``       — BackendOutcome, CounterfactualExperience, CapturedActivation,
                  FeatureRecord, PolicyDecision, Route enum.
``confidence``  — OutcomeConfidence with explicit provenance components.
``targets``     — TargetMode, build_preference_targets (Section 1).
``weighting``   — WeightMode, compute_weight, WeightConfig,
                  gap-threshold tie truncation (Sections 2, 7, 9).
``centroid``    — weighted_mean, weighted_contrastive_mean (BASELINE A/B).
``logistic``    — WeightedLogisticRouter, soft/hard target training,
                  masked weighted_policy_loss (Section 1).
``mlp``         — SmallMLPRouter (Section 30, experimental diagnostic).
``abstention``  — choose_route with calibrated confidence threshold.
``calibration`` — Brier score, ECE, soft Brier, action-confidence ECE,
                  reliability bins, selective-risk curve.
``ood``         — MahalanobisOOD detector with calibrated threshold.
``features``    — PCA feature reduction fitted on TRAIN only.
``config``      — frozen, hashed ExperimentConfig.
``regret``      — per-task regret, mean regret, paired bootstrap CI.
``alignment``   — FeatureRecord, join_by_task_id (Sections 5, 6).
"""

from .types import (
    BackendOutcome,
    CapturedActivation,
    CounterfactualExperience,
    FeatureRecord,
    PolicyDecision,
    Route,
)
from .confidence import OutcomeConfidence
from .targets import TargetMode, build_preference_targets
from .weighting import (
    WeightConfig,
    WeightMode,
    compute_sample_weight,
    compute_weight,
    compute_weights_batch,
    snr_weight,
    utility_weight,
)
from .centroid import (
    weighted_contrastive_mean,
    weighted_mean,
)
from .logistic import (
    HardTargetMode,
    hard_preference_target,
    soft_preference_target,
)

try:
    from .logistic import (
        WeightedLogisticRouter,
        weighted_policy_loss,
    )
except ImportError:  # torch not installed
    WeightedLogisticRouter = None  # type: ignore[assignment, misc]
    weighted_policy_loss = None  # type: ignore[assignment]

try:
    from .mlp import SmallMLPRouter
except ImportError:  # torch not installed
    SmallMLPRouter = None  # type: ignore[assignment]

from .abstention import choose_route, choose_route_with_reason
from .calibration import (
    action_confidence_ece,
    brier_score,
    expected_calibration_error,
    preference_brier_soft,
    predicted_action_correctness_target,
    reliability_bins,
    selective_risk_curve,
    soft_calibration_error,
)
from .ood import MahalanobisOOD
from .features import PCAFeatureReducer
from .config import ExperimentConfig
from .regret import (
    mean_regret,
    paired_bootstrap_mean_ci,
    paired_promotion_statistics,
    per_task_regret,
)
from .alignment import (
    AlignmentResult,
    assert_unique_task_ids,
    join_by_task_id,
)

__all__ = [
    "AlignmentResult",
    "BackendOutcome",
    "CapturedActivation",
    "CounterfactualExperience",
    "ExperimentConfig",
    "FeatureRecord",
    "HardTargetMode",
    "MahalanobisOOD",
    "OutcomeConfidence",
    "PCAFeatureReducer",
    "PolicyDecision",
    "Route",
    "SmallMLPRouter",
    "TargetMode",
    "WeightConfig",
    "WeightMode",
    "WeightedLogisticRouter",
    "action_confidence_ece",
    "assert_unique_task_ids",
    "brier_score",
    "build_preference_targets",
    "choose_route",
    "choose_route_with_reason",
    "compute_sample_weight",
    "compute_weight",
    "compute_weights_batch",
    "expected_calibration_error",
    "hard_preference_target",
    "join_by_task_id",
    "mean_regret",
    "paired_bootstrap_mean_ci",
    "paired_promotion_statistics",
    "per_task_regret",
    "predicted_action_correctness_target",
    "preference_brier_soft",
    "reliability_bins",
    "selective_risk_curve",
    "snr_weight",
    "soft_calibration_error",
    "soft_preference_target",
    "utility_weight",
    "weighted_contrastive_mean",
    "weighted_mean",
    "weighted_policy_loss",
]
