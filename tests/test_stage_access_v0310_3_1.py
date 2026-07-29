"""v0.3.10.3.1-alpha — Section 14-15 / G10: stage access control and
final-test access ledger."""

from __future__ import annotations

import pytest

from daph_learning.policy.stage import (
    ExperimentStage,
    FinalAccessError,
    FinalAccessLedger,
    StageGuard,
)


def test_stage_enum_values():
    assert ExperimentStage.TRAIN.value == "train"
    assert ExperimentStage.FROZEN.value == "frozen"
    assert ExperimentStage.FINAL.value == "final"


def test_final_inaccessible_before_freeze():
    """Section 14 / G10: final split cannot be accessed before FROZEN."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    with pytest.raises(FinalAccessError, match="FROZEN"):
        guard.assert_can_access_split("final")


def test_final_inaccessible_at_dev():
    guard = StageGuard(stage=ExperimentStage.DEV)
    with pytest.raises(FinalAccessError):
        guard.assert_can_access_split("final")


def test_final_inaccessible_at_calibration():
    guard = StageGuard(stage=ExperimentStage.CALIBRATION)
    with pytest.raises(FinalAccessError):
        guard.assert_can_access_split("final")


def test_final_accessible_after_freeze():
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    guard.freeze()
    guard.assert_can_access_split("final")  # no raise


def test_final_access_ledger_records_access():
    """Section 15: every final access writes a ledger entry."""
    ledger = FinalAccessLedger(
        max_accesses=1, source_hash="abc", config_hash="cfg",
        policy_hash="pol", calibration_hash="cal")
    rec = ledger.request_access(command="daph-evaluate-routes", reason="final")
    assert rec.source_hash == "abc"
    assert rec.command == "daph-evaluate-routes"
    assert ledger.n_accesses == 1


def test_final_access_ledger_exhausts_after_one():
    """Section 15: default max_accesses=1; second access rejected."""
    ledger = FinalAccessLedger(max_accesses=1)
    ledger.request_access(command="c1", reason="r1")
    with pytest.raises(FinalAccessError, match="exhausted"):
        ledger.request_access(command="c2", reason="r2")


def test_no_hyperparameter_updates_after_final_access():
    """Section 15: after final access, no further final evaluations."""
    guard = StageGuard()
    guard.freeze()
    guard.request_final_access(command="eval", reason="final")
    assert guard.ledger.exhausted
    with pytest.raises(FinalAccessError):
        guard.request_final_access(command="eval2", reason="again")


def test_ledger_persists_to_disk(tmp_path):
    ledger = FinalAccessLedger(source_hash="s", config_hash="c",
                               policy_hash="p", calibration_hash="k")
    ledger.request_access(command="eval", reason="final")
    path = tmp_path / "ledger.json"
    ledger.save(path)
    import json
    data = json.loads(path.read_text())
    assert data["n_accesses"] == 1
    assert data["records"][0]["source_hash"] == "s"


def test_train_dev_calibration_accessible_before_freeze():
    """Non-final splits are accessible during their own stage."""
    guard = StageGuard(stage=ExperimentStage.TRAIN)
    guard.assert_can_access_split("train")
    guard.transition_to(ExperimentStage.DEV)
    guard.assert_can_access_split("dev")
    guard.transition_to(ExperimentStage.CALIBRATION)
    guard.assert_can_access_split("calibration")
