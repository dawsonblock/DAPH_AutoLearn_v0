"""v0.3.10 — immutable experiment configuration (Section 33).

A canonical frozen configuration containing all parameters that affect
the experiment outcome. The config is hashed and saved with every
experiment so that recorded results are reproducible and traceable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Canonical, frozen experiment configuration (Section 33).

    Contains utility weights, reward-gap threshold, weighting mode, target
    temperature, feature transform, selected layer, capture location,
    steering alpha, norm limits, KL limit, calibration threshold, OOD
    threshold, random seed, model revision, tokenizer revision, and
    dataset hashes.
    """

    # Utility weights (Section 1).
    quality_weight: float = 1.0
    lambda_time: float = 0.0
    lambda_compute: float = 0.0
    lambda_risk: float = 1.0
    time_reference_ms: float = 1000.0
    compute_reference: float = 1.0
    # Reward-gap threshold / abstention band (Sections 1, 7).
    gap_threshold: float = 0.0
    abstention_band: float = 0.0
    # Weighting (Sections 3, 7, 9).
    weight_mode: str = "gap"  # "gap" | "snr"
    min_weight: float = 0.0
    max_weight: float = 1.0
    # Soft targets (Section 5).
    target_temperature: float = 1.0
    soft_targets: bool = True
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
    # Calibration / abstention / OOD (Sections 10, 12).
    confidence_threshold: float = 0.70
    ood_threshold: float = float("inf")
    ood_ridge: float = 1e-4
    # Reproducibility (Sections 23, 33).
    random_seed: int = 0
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    # Dataset hashes (Section 33).
    train_dataset_sha256: str | None = None
    dev_dataset_sha256: str | None = None
    calibration_dataset_sha256: str | None = None
    final_dataset_sha256: str | None = None
    # Policy type (Section 25).
    policy_type: str = "logistic"  # "centroid" | "logistic" | "lowrank"
    # Version.
    autolearn_version: str = "0.3.10"
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
        if self.weight_mode not in ("gap", "snr"):
            raise ValueError("weight_mode must be 'gap' or 'snr'")
        if self.feature_transform not in ("raw", "pca"):
            raise ValueError("feature_transform must be 'raw' or 'pca'")
        if self.policy_type not in ("centroid", "logistic", "lowrank"):
            raise ValueError(
                "policy_type must be 'centroid', "
                "'logistic', or 'lowrank'"
            )

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
        return d


__all__ = ["ExperimentConfig"]
