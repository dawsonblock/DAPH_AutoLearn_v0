"""v0.3.10.2-alpha — immutable experiment configuration (Section 33).

A canonical frozen configuration containing all parameters that affect
the experiment outcome. The config is hashed and saved with every
experiment so that recorded results are reproducible and traceable.

v0.3.10.1 — breaking changes (clean break):

* ``soft_targets: bool`` replaced by ``target_mode: str`` ∈
  ``{"soft", "hard"}``. The bool was a *correctness bug*: the trainer
  always produced soft targets regardless of the flag.
* ``weight_mode`` choices replaced: ``"gap" | "snr"`` →
  ``"uniform" | "absolute_gap" | "clipped_gap" | "snr"``. The old
  ``"gap"`` mode was a *correctness bug*: the experience construction
  code always called the gap function, silently ignoring ``"snr"``.
* ``policy_type``: ``"lowrank"`` removed (not in v0.3.10.1 spec);
  ``"mlp_experimental"`` added.
* ``autolearn_version`` bumped to ``"0.3.10.2-alpha"``.
* ``evidence_level`` and ``intervention_alpha_grid`` added to support
  the v0.3.10.1 real-model intervention pipeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

from .targets import TargetMode
from .weighting import WeightMode


# Default intervention alpha grid (Section 15): symmetric dose-response.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

# Evidence level tags (Section 32, v0.3.10.2 refinement).
EVIDENCE_LEVELS: tuple[str, ...] = (
    "UNIT",
    "SYNTHETIC",
    "REAL_MODEL_DEV",
    "REAL_MODEL_LATENT_INTERVENTION",
    "REAL_MODEL_BEHAVIORAL_INTERVENTION",
    "REAL_MODEL_UTILITY_INTERVENTION",
    "REAL_MODEL_FINAL",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Canonical, frozen experiment configuration (Section 33).

    Contains utility weights, reward-gap threshold, weighting mode,
    target mode + temperature, feature transform, selected layer,
    capture location, steering alpha, norm limits, KL limit,
    calibration threshold, OOD threshold, random seed, model revision,
    tokenizer revision, and dataset hashes.
    """

    # Utility weights (Section 1, 16).
    # v0.3.10.3.2 — cost-weighted so both-correct cases are resolved
    # by cost (LLM API call cheaper than symbolic compute), creating
    # real within-subtype crossover.
    quality_weight: float = 1.0
    lambda_time: float = 0.001
    lambda_compute: float = 0.1
    lambda_risk: float = 1.0
    time_reference_ms: float = 1000.0
    compute_reference: float = 1.0
    # Reward-gap threshold / abstention band (Sections 1, 7).
    gap_threshold: float = 0.0
    abstention_band: float = 0.0
    # Weighting (Sections 2, 7, 9). v0.3.10.1 — clean break.
    weight_mode: str = "clipped_gap"
    min_weight: float = 0.0
    max_weight: float = 1.0
    # v0.3.10.3.1 — Section 2: zero effective class-weight behavior.
    # ERROR (default) raises InsufficientEffectiveWeight;
    # UNWEIGHTED_FALLBACK substitutes the unweighted estimator with
    # recorded provenance.
    zero_weight_policy: str = "error"
    # Targets (Section 1). v0.3.10.1 — replaces soft_targets: bool.
    target_mode: str = "soft"
    target_temperature: float = 1.0
    # v0.3.10.3.1 — Section 5: paired confidence combine for gap weighting.
    # product (default) | min | geometric_mean.
    confidence_combine: str = "product"
    # Feature transform (Section 13).
    feature_transform: str = "raw"  # "raw" | "pca"
    pca_components: int = 32
    # Capture / intervention (Sections 14-16, 29).
    selected_layer: int = 24
    capture_location: str = "anchor"
    steering_alpha: float = 1.0
    # Norm / KL limits (Sections 17, 5).
    max_vector_norm: float | None = None
    max_neutral_kl: float = 0.03
    max_capability_drop: float = 0.02
    min_utility_gain: float = 0.01
    # Calibration / abstention / OOD (Sections 10, 12, 19, 21).
    confidence_threshold: float = 0.70
    ood_threshold: float = float("inf")
    ood_ridge: float = 1e-4
    ood_quantile: float = 0.99
    # Reproducibility (Sections 23, 33).
    random_seed: int = 0
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    # Dataset hashes (Section 33).
    train_dataset_sha256: str | None = None
    dev_dataset_sha256: str | None = None
    calibration_dataset_sha256: str | None = None
    final_dataset_sha256: str | None = None
    # Policy type (Section 3). v0.3.10.1 — centroid | logistic | mlp_experimental.
    policy_type: str = "logistic"
    mlp_hidden_dim: int = 64
    # Early stopping (Section 9). v0.3.10.1 — dev_regret default, real utility path.
    early_stopping_metric: str = "dev_regret"  # dev_loss | dev_regret | dev_utility
    # Intervention (Sections 13-15).
    intervention_alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    # Replay (Section 27). Optional.
    prioritized_replay_alpha: float = 0.0  # 0 = uniform / off
    # Version.
    autolearn_version: str = "0.3.10.3.2-alpha"
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.time_reference_ms <= 0:
            raise ValueError("time_reference_ms must be > 0")
        if self.compute_reference <= 0:
            raise ValueError("compute_reference must be > 0")
        if self.gap_threshold < 0:
            raise ValueError("gap_threshold must be >= 0")
        if self.abstention_band < 0:
            raise ValueError("abstention_band must be >= 0")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be > 0")
        if self.pca_components <= 0:
            raise ValueError("pca_components must be > 0")
        if not 0.5 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.5, 1.0]")
        if self.ood_threshold < 0:
            raise ValueError("ood_threshold must be >= 0")
        if not 0.0 < self.ood_quantile <= 1.0:
            raise ValueError("ood_quantile must be in (0, 1]")
        # Validate enums via from_str (raises on unknown values).
        WeightMode.from_str(self.weight_mode)
        TargetMode.from_str(self.target_mode)
        from .weighting import ZeroWeightPolicy
        ZeroWeightPolicy.from_str(self.zero_weight_policy)
        from .confidence import ConfidenceCombine
        ConfidenceCombine.from_str(self.confidence_combine)
        if self.feature_transform not in ("raw", "pca"):
            raise ValueError("feature_transform must be 'raw' or 'pca'")
        if self.policy_type not in (
                "centroid", "logistic", "mlp_experimental"):
            raise ValueError(
                "policy_type must be 'centroid', 'logistic', "
                "or 'mlp_experimental'")
        if self.early_stopping_metric not in (
                "dev_loss", "dev_regret", "dev_utility"):
            raise ValueError(
                "early_stopping_metric must be 'dev_loss', "
                "'dev_regret', or 'dev_utility'")
        if self.mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")
        if not 0.0 <= self.prioritized_replay_alpha <= 1.0:
            raise ValueError("prioritized_replay_alpha must be in [0, 1]")
        if len(self.intervention_alpha_grid) == 0:
            raise ValueError("intervention_alpha_grid must be non-empty")

    def _hash_payload(self) -> str:
        payload = {
            k: v for k, v in asdict(self).items()
            if k != "config_sha256"
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def freeze(self) -> "ExperimentConfig":
        """Return a copy with ``config_sha256`` populated (idempotent)."""
        if self.config_sha256 is not None:
            return self
        return replace(self, config_sha256=self._hash_payload())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.config_sha256 is None:
            d["config_sha256"] = self._hash_payload()
        if d.get("ood_threshold") == float("inf"):
            d["ood_threshold"] = None
        # Convert tuple to list for JSON friendliness.
        d["intervention_alpha_grid"] = list(self.intervention_alpha_grid)
        return d


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "EVIDENCE_LEVELS",
    "ExperimentConfig",
]
