"""Section 16 — extended final-access state machine tests."""

from __future__ import annotations

import pytest

from daph_learning.policy.stage import (
    ExperimentStage,
    FinalAccessError,
    FinalAccessLedger,
    StageGuard,
)


def test_new_stages_exist():
    assert ExperimentStage.CREATED.value == "created"
    assert ExperimentStage.DATASET_FROZEN.value == "dataset_frozen"
    assert ExperimentStage.EXPERIENCE_COLLECTED.value == "experience_collected"
    assert ExperimentStage.DEVELOPMENT_COMPLETE.value == "development_complete"
    assert ExperimentStage.CALIBRATED.value == "calibrated"
    assert ExperimentStage.POLICY_FROZEN.value == "policy_frozen"
    assert ExperimentStage.FINAL_RESERVED.value == "final_reserved"
    assert ExperimentStage.FINAL_EVALUATED.value == "final_evaluated"
    assert ExperimentStage.REPORTED.value == "reported"


def test_from_str_accepts_new_stages():
    assert ExperimentStage.from_str("created") == ExperimentStage.CREATED
    assert ExperimentStage.from_str("policy_frozen") == ExperimentStage.POLICY_FROZEN


def test_valid_forward_transition():
    guard = StageGuard(stage=ExperimentStage.CREATED)
    guard.transition_to(ExperimentStage.DATASET_FROZEN)
    assert guard.stage == ExperimentStage.DATASET_FROZEN


def test_invalid_backward_transition_raises():
    guard = StageGuard(stage=ExperimentStage.POLICY_FROZEN)
    with pytest.raises(FinalAccessError, match="invalid stage transition"):
        guard.transition_to(ExperimentStage.CREATED)


def test_full_lifecycle():
    guard = StageGuard(stage=ExperimentStage.CREATED)
    guard.transition_to(ExperimentStage.DATASET_FROZEN)
    guard.transition_to(ExperimentStage.EXPERIENCE_COLLECTED)
    guard.transition_to(ExperimentStage.DEVELOPMENT_COMPLETE)
    guard.transition_to(ExperimentStage.CALIBRATED)
    guard.transition_to(ExperimentStage.POLICY_FROZEN)
    guard.transition_to(ExperimentStage.FINAL_RESERVED)
    guard.transition_to(ExperimentStage.FINAL_EVALUATED)
    guard.transition_to(ExperimentStage.REPORTED)
    assert guard.stage == ExperimentStage.REPORTED


def test_reported_is_terminal():
    guard = StageGuard(stage=ExperimentStage.REPORTED)
    with pytest.raises(FinalAccessError):
        guard.transition_to(ExperimentStage.CREATED)


def test_final_access_budget_exhausted():
    ledger = FinalAccessLedger(max_accesses=1)
    ledger.request_access(command="test", reason="first")
    with pytest.raises(FinalAccessError, match="budget exhausted"):
        ledger.request_access(command="test", reason="second")


def test_final_access_before_frozen_raises():
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    with pytest.raises(FinalAccessError, match="must be FROZEN"):
        guard.assert_can_access_split("final")


def test_legacy_stages_still_work():
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    guard.transition_to(ExperimentStage.DEV)
    guard.transition_to(ExperimentStage.CALIBRATION)
    guard.transition_to(ExperimentStage.FROZEN)
    assert guard.stage == ExperimentStage.FROZEN
    guard.assert_can_access_split("final")  # should not raise
