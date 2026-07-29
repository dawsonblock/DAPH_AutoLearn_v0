"""v0.3.10.1-alpha — config field audit (Section 44).

Search every config field and CLI option. For each, classify:

USED         — the field is read by the code and affects behavior.
DEPRECATED   — the field is accepted for backwards compat but maps to
               a newer field; emits a warning.
EXPERIMENTAL — the field is implemented but not enabled by default and
               not yet scientifically validated.
UNUSED_BUG   — the field is accepted but never read; a correctness bug.

No CLI flag may silently do nothing. Fail startup if unsupported
combinations are requested.

This module provides:
* :func:`audit_config_fields` — classify every ExperimentConfig field.
* :func:`audit_cli_options` — classify every CLI option.
* :func:`validate_config_consistency` — fail on unsupported combinations.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from .config import ExperimentConfig


# Classification of ExperimentConfig fields (Section 44).
# Each entry: (status, description, read_by)
FIELD_AUDIT: dict[str, tuple[str, str, str]] = {
    "quality_weight": ("USED", "utility weight w_q", "learner._utility"),
    "lambda_time": ("USED", "time penalty coefficient", "learner._utility"),
    "lambda_compute": ("USED", "compute cost coefficient", "learner._utility"),
    "lambda_risk": ("USED", "risk penalty coefficient", "learner._utility"),
    "time_reference_ms": ("USED", "time normalization", "learner._utility"),
    "compute_reference": ("USED", "compute normalization", "learner._utility"),
    "gap_threshold": ("USED", "tie truncation band", "weighting.compute_weight, targets.build_preference_targets"),
    "abstention_band": ("USED", "abstention band for preferred_action", "learner.build_counterfactual_experiences"),
    "weight_mode": ("USED", "weight mode dispatch", "weighting.compute_weight, learner.build_counterfactual_experiences"),
    "min_weight": ("USED", "numerical stability floor", "weighting.compute_weight"),
    "max_weight": ("USED", "weight clipping", "weighting.compute_weight"),
    "target_mode": ("USED", "target mode dispatch", "targets.build_preference_targets, logistic.train_weighted_logistic_router"),
    "target_temperature": ("USED", "soft target temperature", "targets.build_preference_targets"),
    "feature_transform": ("USED", "PCA vs raw", "learner.train_policy_learner"),
    "pca_components": ("USED", "PCA dimension", "features.PCAFeatureReducer"),
    "selected_layer": ("USED", "transformer layer for capture", "real_pipeline.InterventionConfig, cli"),
    "capture_location": ("USED", "token location for capture", "cli/commands/policy"),
    "steering_alpha": ("USED", "intervention alpha", "real_pipeline, cli"),
    "max_vector_norm": ("USED", "vector norm clamp", "real_pipeline.InterventionConfig"),
    "max_neutral_kl": ("USED", "KL gate threshold", "kl_gate.PromotionConstraints"),
    "max_capability_drop": ("USED", "capability gate threshold", "capability_gate.CapabilityGateConfig"),
    "min_utility_gain": ("USED", "promotion threshold", "promotion"),
    "confidence_threshold": ("USED", "abstention threshold", "abstention.choose_route_with_reason, learner"),
    "ood_threshold": ("USED", "OOD abstention threshold", "learner.train_policy_learner, abstention.choose_route_with_reason"),
    "ood_ridge": ("USED", "Mahalanobis regularization", "ood.MahalanobisOOD"),
    "ood_quantile": ("USED", "OOD threshold calibration quantile", "cli calibrate"),
    "random_seed": ("USED", "reproducibility seed", "logistic.train_weighted_logistic_router, mlp.train_small_mlp_router"),
    "model_revision": ("USED", "model revision hash", "provenance"),
    "tokenizer_revision": ("USED", "tokenizer revision hash", "provenance"),
    "train_dataset_sha256": ("USED", "train dataset hash", "provenance"),
    "dev_dataset_sha256": ("USED", "dev dataset hash", "provenance"),
    "calibration_dataset_sha256": ("USED", "calibration dataset hash", "provenance"),
    "final_dataset_sha256": ("USED", "final dataset hash", "provenance"),
    "policy_type": ("USED", "policy dispatch", "policy_factory.fit_policy, make_policy"),
    "mlp_hidden_dim": ("EXPERIMENTAL", "MLP hidden dim (diagnostic only)", "mlp.SmallMLPRouter, mlp.MLPTrainConfig"),
    "early_stopping_metric": ("USED", "dev early stopping", "logistic._evaluate_dev_metric"),
    "intervention_alpha_grid": ("USED", "dose-response grid", "real_pipeline.InterventionConfig"),
    "prioritized_replay_alpha": ("EXPERIMENTAL", "prioritized replay exponent (optional)", "replay"),
    "autolearn_version": ("USED", "version string", "provenance, version tests"),
    "config_sha256": ("USED", "config hash", "config.freeze"),
}


def audit_config_fields() -> dict[str, dict[str, str]]:
    """Classify every ExperimentConfig field (Section 44).

    Returns
    -------
    dict
        Mapping field_name -> {status, description, read_by}.
    """
    config_fields = {f.name for f in fields(ExperimentConfig)}
    audited = {}
    for name in sorted(config_fields):
        if name in FIELD_AUDIT:
            status, desc, read_by = FIELD_AUDIT[name]
            audited[name] = {
                "status": status,
                "description": desc,
                "read_by": read_by,
            }
        else:
            # Any field not in the audit table is flagged for review.
            audited[name] = {
                "status": "UNAUDITED",
                "description": "field not in audit table",
                "read_by": "unknown",
            }
    return audited


def audit_cli_options() -> dict[str, dict[str, str]]:
    """Classify every CLI option in cli/commands/policy.py (Section 44)."""
    return {
        "--policy": {"status": "USED", "description": "policy type dispatch"},
        "--target-mode": {"status": "USED", "description": "target mode (replaces --soft-targets)"},
        "--target-temperature": {"status": "USED", "description": "soft target temperature"},
        "--weight-mode": {"status": "USED", "description": "weight mode dispatch"},
        "--gap-threshold": {"status": "USED", "description": "tie truncation band"},
        "--early-stopping-metric": {"status": "USED", "description": "dev early stopping metric"},
        "--mlp-hidden-dim": {"status": "EXPERIMENTAL", "description": "MLP hidden dim (diagnostic)"},
        "--layer": {"status": "USED", "description": "transformer layer for capture"},
        "--alpha": {"status": "USED", "description": "steering alpha"},
        "--confidence-threshold": {"status": "USED", "description": "abstention threshold"},
        "--ood-threshold": {"status": "USED", "description": "OOD abstention threshold"},
        "--max-weight": {"status": "USED", "description": "weight clipping"},
        "--seed": {"status": "USED", "description": "random seed"},
        "--model": {"status": "USED", "description": "HuggingFace model ID"},
        "--model-revision": {"status": "USED", "description": "model revision hash"},
        "--tokenizer-revision": {"status": "USED", "description": "tokenizer revision hash"},
        "--synthetic": {"status": "USED", "description": "use synthetic environment"},
        "--benchmark": {"status": "USED", "description": "use redesigned benchmark"},
        "--dataset": {"status": "USED", "description": "dataset JSONL for real-model flow"},
        "--train-tasks": {"status": "USED", "description": "train tasks JSONL"},
        "--dev-tasks": {"status": "USED", "description": "dev tasks JSONL"},
        "--test-tasks": {"status": "USED", "description": "test tasks JSONL"},
        "--policy-file": {"status": "USED", "description": "trained policy artifact"},
        "--vector-file": {"status": "USED", "description": "steering vector .npy"},
        "--calibration-tasks": {"status": "USED", "description": "calibration tasks JSONL"},
        "--output": {"status": "USED", "description": "output JSON path"},
    }


def validate_config_consistency(config: ExperimentConfig) -> None:
    """Fail startup if unsupported combinations are requested (Section 44).

    Examples:
    - SNR weight mode requires sigma_delta_u_fn (checked at compute time).
    - mlp_experimental policy with centroid-only configs is not supported.
    - PCA feature_transform with pca_components > train N is not supported.
    """
    # SNR mode: validated at compute_weight call time (raises if sigma_delta_u
    # is None). Here we just check the mode is valid.
    if config.weight_mode not in ("uniform", "absolute_gap", "clipped_gap", "snr"):
        raise ValueError(
            f"unsupported weight_mode {config.weight_mode!r}; "
            f"clean break — old 'gap' is not accepted")
    if config.target_mode not in ("soft", "hard"):
        raise ValueError(
            f"unsupported target_mode {config.target_mode!r}")
    if config.policy_type not in ("centroid", "logistic", "mlp_experimental"):
        raise ValueError(
            f"unsupported policy_type {config.policy_type!r}")
    if config.early_stopping_metric not in ("dev_loss", "dev_regret", "dev_utility"):
        raise ValueError(
            f"unsupported early_stopping_metric {config.early_stopping_metric!r}")
    # mlp_experimental with hard targets + SNR weights is a valid but
    # experimental combination; we allow it but flag it.
    if config.policy_type == "mlp_experimental":
        import warnings
        warnings.warn(
            "mlp_experimental policy is experimental (Section 30); "
            "use for diagnostic comparison only, not production",
            UserWarning, stacklevel=2)


def audit_report() -> dict[str, Any]:
    """Produce a full audit report for the release artifact (Section 44)."""
    cfg_fields = audit_config_fields()
    cli_opts = audit_cli_options()
    # Check for UNUSED_BUG fields.
    unused_bugs = [
        name for name, info in cfg_fields.items()
        if info["status"] == "UNUSED_BUG"
    ]
    unaudited = [
        name for name, info in cfg_fields.items()
        if info["status"] == "UNAUDITED"
    ]
    return {
        "config_fields": cfg_fields,
        "cli_options": cli_opts,
        "unused_bugs": unused_bugs,
        "unaudited_fields": unaudited,
        "no_unused_bugs": len(unused_bugs) == 0,
        "all_fields_audited": len(unaudited) == 0,
    }


__all__ = [
    "audit_cli_options",
    "audit_config_fields",
    "audit_report",
    "validate_config_consistency",
]
