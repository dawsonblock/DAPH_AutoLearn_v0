from __future__ import annotations

from daph_learning.autolearn import classify_outcome
from daph_learning.evaluation.verification import (
    NumericVerifier,
    VerificationStatus,
)


def test_numeric_verifier_rejects_substring_false_positive() -> None:
    result = NumericVerifier().verify({"expected": 12}, "FINAL: 312")
    assert result.status is VerificationStatus.VERIFIED_INCORRECT
    assert result.observed == 312


def test_numeric_verifier_rejects_unstructured_prose() -> None:
    result = NumericVerifier().verify({"expected": 12}, "I saw 12 birds")
    assert result.status is VerificationStatus.INVALID_OUTPUT


def test_numeric_verifier_accepts_exact_or_final_integer() -> None:
    assert (
        NumericVerifier().verify({"expected": -7}, "-7").status
        is VerificationStatus.VERIFIED_CORRECT
    )
    assert (
        NumericVerifier().verify({"expected": -7}, "FINAL: -7").status
        is VerificationStatus.VERIFIED_CORRECT
    )


def test_route_raw_is_not_reused_as_an_answer() -> None:
    task = {
        "task_id": "t1",
        "expected": 42,
        "capability_ids": ["integer_arithmetic"],
    }
    assert (
        classify_outcome(
            task,
            "llm",
            None,
            route_raw="LLM. The answer is 42.",
        )
        == "unverifiable"
    )


def test_symbolic_without_expected_value_is_unverifiable() -> None:
    task = {
        "task_id": "t1",
        "expected": None,
        "capability_ids": ["integer_arithmetic"],
    }
    assert classify_outcome(task, "symbolic", "FINAL: 42") == "unverifiable"
