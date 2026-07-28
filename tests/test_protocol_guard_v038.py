from __future__ import annotations

import pytest

from daph_learning.evaluation.protocol import (
    ProtocolPurpose,
    ProtocolViolation,
    assert_split_access,
    load_final_test_access,
    reserve_final_test,
)


@pytest.mark.parametrize(
    "purpose",
    [
        ProtocolPurpose.VECTOR_CAPTURE,
        ProtocolPurpose.HYPERPARAMETER_TUNING,
        ProtocolPurpose.CALIBRATION,
        ProtocolPurpose.MODEL_SELECTION,
        ProtocolPurpose.ENGINEERING_EVAL,
    ],
)
def test_final_test_rejected_for_adaptive_purpose(purpose) -> None:
    with pytest.raises(ProtocolViolation):
        assert_split_access("final_test", purpose)


def test_final_evaluation_requires_frozen_configuration() -> None:
    with pytest.raises(ProtocolViolation, match="frozen"):
        assert_split_access("final_test", ProtocolPurpose.FINAL_EVALUATION)


def test_final_test_ledger_is_one_shot(tmp_path) -> None:
    path = tmp_path / "final_test_access.json"
    reserve_final_test(
        path,
        run_id="run-1",
        dataset_sha256="a" * 64,
        configuration_sha256="b" * 64,
    )
    assert load_final_test_access(path)["run_id"] == "run-1"
    with pytest.raises(ProtocolViolation, match="retired"):
        reserve_final_test(
            path,
            run_id="run-2",
            dataset_sha256="a" * 64,
            configuration_sha256="b" * 64,
        )


def test_final_test_ledger_rejects_non_hash_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="hexadecimal SHA-256"):
        reserve_final_test(
            tmp_path / "invalid.json",
            run_id="run-invalid",
            dataset_sha256="not-a-hash".ljust(64, "x"),
            configuration_sha256="b" * 64,
        )
