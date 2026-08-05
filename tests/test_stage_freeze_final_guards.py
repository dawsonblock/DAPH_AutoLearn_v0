"""v0.3.10.3.2-alpha — Section 33-36 / G33: stage machine + freeze +
final guards.

Verifies that the stage machine enforces split access discipline, the
freeze manifest binds final evaluation to the frozen state, and final
access is budget-limited.
"""

from __future__ import annotations

import pytest

from daph_learning.policy.stage import (
    ExperimentStage,
    FinalAccessError,
    FinalAccessLedger,
    FreezeManifest,
    StageGuard,
)


def _complete_freeze_kwargs(**overrides):
    """Return kwargs that satisfy FreezeManifest.assert_complete()."""
    base = dict(
        experiment_id="test_exp",
        source_hash="a" * 64,
        config_hash="b" * 64,
        train_dataset_hash="t",
        dev_dataset_hash="d",
        cal_dataset_hash="c",
        final_dataset_hash="f",
        utility_config_hash="u",
        model_id="test-model",
        representation_hash="r",
        policy_hash="p",
        calibration_hash="cal",
        gate_criteria_hash="g",
    )
    base.update(overrides)
    return base


def test_final_cannot_be_accessed_before_freeze():
    """Section 14: the final split cannot be accessed before FROZEN."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    with pytest.raises(FinalAccessError, match="must be FROZEN"):
        guard.assert_can_access_split("final")


def test_final_can_be_accessed_after_freeze():
    """Section 14: the final split can be accessed after FROZEN."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    guard.freeze(**_complete_freeze_kwargs())
    guard.assert_can_access_split("final")  # should not raise


def test_final_access_budget_is_enforced():
    """Section 15: the final access budget is enforced (default 1)."""
    guard = StageGuard(stage=ExperimentStage.FROZEN)
    guard.request_final_access(command="evaluate", reason="final eval")
    with pytest.raises(FinalAccessError, match="budget exhausted"):
        guard.request_final_access(command="evaluate", reason="second try")


def test_freeze_manifest_records_state():
    """Section 33: the freeze manifest records the exact state at
    freeze time."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    manifest = guard.freeze(**_complete_freeze_kwargs(
        policy_hash="c" * 64,
        calibration_hash="d" * 64,
    ))
    assert manifest.source_tree_sha256 == "a" * 64
    assert manifest.config_sha256 == "b" * 64
    assert manifest.policy_sha256 == "c" * 64
    assert manifest.calibration_sha256 == "d" * 64
    assert manifest.stage == "frozen"
    assert manifest.frozen_at > 0
    assert manifest.train_dataset_sha256 == "t"
    assert manifest.utility_config_sha256 == "u"
    assert manifest.model_id == "test-model"
    assert manifest.gate_criteria_sha256 == "g"


def test_freeze_manifest_verifies_matching_state():
    """Section 33: the freeze manifest verifies that the current state
    matches the frozen state."""
    manifest = FreezeManifest(
        source_tree_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    assert manifest.verify_current_state(
        current_source_hash="a" * 64,
        current_config_hash="b" * 64,
    )


def test_freeze_manifest_detects_mismatch():
    """Section 33: the freeze manifest detects state mismatch."""
    manifest = FreezeManifest(
        source_tree_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    with pytest.raises(RuntimeError, match="source_tree_sha256"):
        manifest.verify_current_state(
            current_source_hash="x" * 64,
            current_config_hash="b" * 64,
        )


def test_freeze_manifest_non_strict_returns_false_on_mismatch():
    """Section 33: non-strict mode returns False on mismatch."""
    manifest = FreezeManifest(
        source_tree_sha256="a" * 64,
    )
    assert not manifest.verify_current_state(
        current_source_hash="x" * 64,
        strict=False,
    )


def test_stage_guard_verify_frozen_state():
    """Section 33: the stage guard verifies frozen state before final
    evaluation."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    guard.freeze(**_complete_freeze_kwargs())
    # Should not raise.
    guard.verify_frozen_state(
        current_source_hash="a" * 64,
        current_config_hash="b" * 64,
        current_train_dataset_hash="t",
        current_dev_dataset_hash="d",
        current_cal_dataset_hash="c",
        current_final_dataset_hash="f",
        current_utility_config_hash="u",
        current_model_id="test-model",
        current_representation_hash="r",
        current_gate_criteria_hash="g",
    )
    # Should raise on mismatch.
    with pytest.raises(RuntimeError, match="source_tree_sha256"):
        guard.verify_frozen_state(
            current_source_hash="x" * 64,
            current_config_hash="b" * 64,
        )


def test_stage_guard_verify_without_freeze_raises():
    """Section 33: verifying frozen state without a freeze manifest
    raises."""
    guard = StageGuard(stage=ExperimentStage.FROZEN)
    with pytest.raises(RuntimeError, match="no freeze manifest"):
        guard.verify_frozen_state(current_source_hash="a" * 64)


def test_ledger_records_final_access():
    """Section 15: the ledger records each final access."""
    guard = StageGuard(stage=ExperimentStage.FROZEN)
    guard.ledger.source_hash = "abc123"
    rec = guard.request_final_access(
        command="evaluate --final",
        reason="qualification final eval",
    )
    assert rec.command == "evaluate --final"
    assert rec.reason == "qualification final eval"
    assert rec.source_hash == "abc123"
    assert guard.ledger.n_accesses == 1


def test_freeze_manifest_save_load_roundtrip(tmp_path):
    """Section 33: freeze manifest can be saved and loaded."""
    manifest = FreezeManifest(
        experiment_id="exp",
        source_tree_sha256="a" * 64,
        config_sha256="b" * 64,
        policy_sha256="c" * 64,
        train_dataset_sha256="t",
        development_dataset_sha256="d",
        calibration_dataset_sha256="c",
        final_dataset_sha256="f",
        utility_config_sha256="u",
        model_id="m",
        representation_config_sha256="r",
        calibration_sha256="cal",
        gate_criteria_sha256="g",
    )
    path = tmp_path / "freeze_manifest.json"
    manifest.save(path)
    loaded = FreezeManifest.load(path)
    assert loaded.source_tree_sha256 == "a" * 64
    assert loaded.config_sha256 == "b" * 64
    assert loaded.policy_sha256 == "c" * 64
    assert loaded.train_dataset_sha256 == "t"
    assert loaded.utility_config_sha256 == "u"
    assert loaded.model_id == "m"
    assert loaded.gate_criteria_sha256 == "g"


def test_freeze_manifest_assert_complete():
    """Section 33: assert_complete raises when required fields are empty."""
    manifest = FreezeManifest(
        experiment_id="exp",
        source_tree_sha256="a",
        config_sha256="b",
        train_dataset_sha256="t",
        development_dataset_sha256="d",
        calibration_dataset_sha256="c",
        final_dataset_sha256="f",
        utility_config_sha256="u",
        model_id="m",
        representation_config_sha256="r",
        policy_sha256="p",
        calibration_sha256="cal",
        gate_criteria_sha256="g",
    )
    manifest.assert_complete()  # should not raise

    incomplete = FreezeManifest(experiment_id="exp")
    with pytest.raises(RuntimeError, match="incomplete"):
        incomplete.assert_complete()


def test_freeze_manifest_verifies_all_fields():
    """Section 33: verify_current_state checks all identity-bearing fields."""
    manifest = FreezeManifest(
        experiment_id="exp",
        source_tree_sha256="a",
        config_sha256="b",
        train_dataset_sha256="t",
        development_dataset_sha256="d",
        calibration_dataset_sha256="c",
        final_dataset_sha256="f",
        utility_config_sha256="u",
        model_id="m",
        representation_config_sha256="r",
        policy_sha256="p",
        calibration_sha256="cal",
        gate_criteria_sha256="g",
    )
    # All match.
    assert manifest.verify_current_state(
        current_source_hash="a", current_config_hash="b",
        current_train_dataset_hash="t", current_dev_dataset_hash="d",
        current_cal_dataset_hash="c", current_final_dataset_hash="f",
        current_utility_config_hash="u", current_model_id="m",
        current_representation_hash="r", current_gate_criteria_hash="g",
    )
    # Model mismatch.
    with pytest.raises(RuntimeError, match="model_id"):
        manifest.verify_current_state(current_model_id="wrong")
    # Dataset mismatch.
    with pytest.raises(RuntimeError, match="train_dataset_sha256"):
        manifest.verify_current_state(current_train_dataset_hash="wrong")


def test_train_dev_calibration_accessible_at_any_stage():
    """Section 14: train/dev/calibration splits are accessible at any
    stage."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    guard.assert_can_access_split("train")
    guard.assert_can_access_split("dev")
    guard.assert_can_access_split("calibration")
    guard.transition_to(ExperimentStage.DEV)
    guard.assert_can_access_split("train")
    guard.assert_can_access_split("dev")
    guard.assert_can_access_split("calibration")
