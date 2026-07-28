"""v0.3.10 — learned policy package: counterfactual utility learning,
weighted latent routing, calibrated abstention, OOD detection, and
causal intervention validation.

This package implements the v0.3.10 architecture shift from
"weighted steering-vector extractor" to "counterfactual compute-selection
learner". The weighted centroid remains as a transparent baseline
(:mod:`daph_learning.policy.centroid`); the primary policy learner is the
weighted soft-target logistic router
(:mod:`daph_learning.policy.logistic`).

Submodules
----------
``types``       — BackendOutcome, CounterfactualExperience, CapturedActivation,
                  PolicyDecision, Route enum.
``confidence``  — OutcomeConfidence with explicit provenance components.
``weighting``   — utility_weight, WeightConfig, snr_weight, gap-threshold
                  tie truncation.
``centroid``    — weighted_mean, weighted_contrastive_mean (BASELINE A/B).
``logistic``    — WeightedLogisticRouter, soft_preference_target,
                  weighted_policy_loss, hard-target mode.
``abstention``  — choose_route with calibrated confidence threshold.
``calibration`` — Brier score, ECE, reliability bins, selective-risk curve.
``ood``         — MahalanobisOOD detector.
``features``    — PCA feature reduction fitted on TRAIN only.
``config``      — frozen, hashed ExperimentConfig.
``regret``      — per-task regret, mean regret, paired bootstrap CI.
"""

from .types import (
    BackendOutcome,
    CapturedActivation,
    CounterfactualExperience,
    PolicyDecision,
    Route,
)
from .confidence import OutcomeConfidence
from .weighting import (
    WeightConfig,
    WeightMode,
    compute_sample_weight,
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
from .abstention import choose_route
from .calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
    selective_risk_curve,
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

__all__ = [
    "BackendOutcome",
    "CapturedActivation",
    "CounterfactualExperience",
    "ExperimentConfig",
    "HardTargetMode",
    "MahalanobisOOD",
    "OutcomeConfidence",
    "PCAFeatureReducer",
    "PolicyDecision",
    "Route",
    "WeightConfig",
    "WeightMode",
    "WeightedLogisticRouter",
    "brier_score",
    "choose_route",
    "compute_sample_weight",
    "expected_calibration_error",
    "hard_preference_target",
    "mean_regret",
    "paired_bootstrap_mean_ci",
    "paired_promotion_statistics",
    "per_task_regret",
    "reliability_bins",
    "selective_risk_curve",
    "snr_weight",
    "soft_preference_target",
    "utility_weight",
    "weighted_contrastive_mean",
    "weighted_mean",
    "weighted_policy_loss",
]
