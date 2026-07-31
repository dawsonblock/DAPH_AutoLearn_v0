"""Section 7/12 — tests for BackendExecution + canonical verification.

Verifies that both symbolic and LLM backends pass through the same
canonical verifier, that the FINAL_ANSWER format is enforced, and that
utilities are computed via compute_utility.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.execution.real_backends import (
    BackendExecution,
    BackendName,
    ExecutionStatus,
    MockLLMBackend,
    build_canonical_counterfactual_experience,
    execute_symbolic_canonical,
    execution_to_outcome,
    verify_backend_execution,
)
from daph_learning.evaluation.canonical_verifier import VerificationStatus
from daph_learning.policy.utility import (
    GATE_A_ACCURACY_PRIMARY,
    compute_utility,
)


def _make_arithmetic_task(a=12, b=30, op="+", expected=None):
    if expected is None:
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op == "*":
            expected = a * b
    return {
        "task_id": f"arith_{a}_{b}_{op}",
        "prompt": f"What is {a} {op} {b}? Answer with just the number.",
        "specification": f"{a} {op} {b}",
        "expected": expected,
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": a, "b": b, "op": op},
    }


def test_symbolic_canonical_wraps_as_final_answer():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = execute_symbolic_canonical(task)
    assert exec_result.backend == BackendName.SYMBOLIC
    assert exec_result.execution_status == ExecutionStatus.SUCCESS
    assert exec_result.canonical_answer == 42
    assert "FINAL_ANSWER: 42" in exec_result.canonical_output


def test_symbolic_canonical_unsupported():
    task = {
        "task_id": "unsupported",
        "prompt": "What is the meaning of life?",
        "specification": "meaning_of_life",
        "expected": 42,
        "capability_ids": ["unknown_capability"],
    }
    exec_result = execute_symbolic_canonical(task)
    assert exec_result.execution_status == ExecutionStatus.UNSUPPORTED
    assert exec_result.canonical_answer is None


def test_llm_canonical_extracts_final_answer():
    task = _make_arithmetic_task(12, 30, "+", 42)
    backend = MockLLMBackend(accuracy=1.0, seed=42)
    exec_result = backend.execute(task)
    assert exec_result.backend == BackendName.LLM
    assert exec_result.execution_status == ExecutionStatus.SUCCESS
    assert exec_result.canonical_answer == 42


def test_verify_backend_execution_correct():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = BackendExecution(
        backend=BackendName.SYMBOLIC,
        raw_output="FINAL_ANSWER: 42",
        canonical_answer=42,
        latency_ms=1.0,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={},
    )
    status = verify_backend_execution(task, exec_result)
    assert status == VerificationStatus.CORRECT


def test_verify_backend_execution_incorrect():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = BackendExecution(
        backend=BackendName.SYMBOLIC,
        raw_output="FINAL_ANSWER: 99",
        canonical_answer=99,
        latency_ms=1.0,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={},
    )
    status = verify_backend_execution(task, exec_result)
    assert status == VerificationStatus.INCORRECT


def test_verify_backend_execution_unverifiable():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = BackendExecution(
        backend=BackendName.LLM,
        raw_output="I don't know",
        canonical_answer=None,
        latency_ms=1.0,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={},
    )
    status = verify_backend_execution(task, exec_result)
    assert status == VerificationStatus.UNVERIFIABLE


def test_verify_backend_execution_execution_error():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = BackendExecution(
        backend=BackendName.SYMBOLIC,
        raw_output="",
        canonical_answer=None,
        latency_ms=1.0,
        execution_status=ExecutionStatus.EXECUTION_ERROR,
        metadata={"failure_reason": "overflow"},
    )
    status = verify_backend_execution(task, exec_result)
    assert status == VerificationStatus.EXECUTION_ERROR


def test_execution_to_outcome_correct():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = BackendExecution(
        backend=BackendName.SYMBOLIC,
        raw_output="FINAL_ANSWER: 42",
        canonical_answer=42,
        latency_ms=1.0,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={},
    )
    outcome = execution_to_outcome(
        task, exec_result, VerificationStatus.CORRECT)
    assert outcome.backend == "symbolic"
    assert outcome.correct is True
    assert outcome.quality == 1.0
    assert outcome.verifier_status == "verified_correct"
    assert outcome.verifier_confidence == 1.0


def test_execution_to_outcome_incorrect():
    task = _make_arithmetic_task(12, 30, "+", 42)
    exec_result = BackendExecution(
        backend=BackendName.LLM,
        raw_output="FINAL_ANSWER: 99",
        canonical_answer=99,
        latency_ms=100.0,
        execution_status=ExecutionStatus.SUCCESS,
        metadata={},
    )
    outcome = execution_to_outcome(
        task, exec_result, VerificationStatus.INCORRECT)
    assert outcome.correct is False
    assert outcome.quality == 0.0
    assert outcome.verifier_status == "verified_incorrect"


def test_build_canonical_counterfactual_experience():
    task = _make_arithmetic_task(12, 30, "+", 42)
    sym_exec = execute_symbolic_canonical(task)
    llm_backend = MockLLMBackend(accuracy=1.0, seed=42)
    llm_exec = llm_backend.execute(task)

    experience, sym_status, llm_status = build_canonical_counterfactual_experience(
        task, sym_exec, llm_exec, GATE_A_ACCURACY_PRIMARY)

    assert experience.task_id == task["task_id"]
    assert sym_status == VerificationStatus.CORRECT
    assert llm_status == VerificationStatus.CORRECT
    # Both correct → delta_u should be small (cost difference only).
    assert abs(experience.delta_utility) < 0.5


def test_build_canonical_counterfactual_experience_mixed():
    task = _make_arithmetic_task(12, 30, "+", 42)
    sym_exec = execute_symbolic_canonical(task)  # correct
    llm_backend = MockLLMBackend(accuracy=0.0, seed=42)  # always wrong
    llm_exec = llm_backend.execute(task)

    experience, sym_status, llm_status = build_canonical_counterfactual_experience(
        task, sym_exec, llm_exec, GATE_A_ACCURACY_PRIMARY)

    assert sym_status == VerificationStatus.CORRECT
    assert llm_status == VerificationStatus.INCORRECT
    # Symbolic correct, LLM wrong → symbolic preferred.
    assert experience.preferred_action.value == "symbolic"
    assert experience.delta_utility > 0


def test_compute_utility_through_canonical_path():
    """Utilities must be computed via compute_utility, not duplicated."""
    task = _make_arithmetic_task(12, 30, "+", 42)
    sym_exec = execute_symbolic_canonical(task)
    llm_backend = MockLLMBackend(accuracy=0.0, seed=42)
    llm_exec = llm_backend.execute(task)

    experience, _, _ = build_canonical_counterfactual_experience(
        task, sym_exec, llm_exec, GATE_A_ACCURACY_PRIMARY)

    # Verify utilities match compute_utility.
    u_sym = compute_utility(experience.symbolic, GATE_A_ACCURACY_PRIMARY)
    u_llm = compute_utility(experience.llm, GATE_A_ACCURACY_PRIMARY)
    assert experience.delta_utility == pytest.approx(u_sym - u_llm)


def test_mock_llm_deterministic():
    """MockLLMBackend with same seed produces same results."""
    task = _make_arithmetic_task(12, 30, "+", 42)
    b1 = MockLLMBackend(accuracy=0.5, seed=123)
    b2 = MockLLMBackend(accuracy=0.5, seed=123)
    r1 = b1.execute(task)
    r2 = b2.execute(task)
    assert r1.canonical_answer == r2.canonical_answer
