"""V039-001/002/004 — separated repair-outcome semantics.

Acceptance tests for Item 1 of the recommended repair sequence:

* an unexecuted backend can never be correct
* backend_selected / backend_executed / verification_status / route_suitability
  are distinct and independently inspectable
* the structural invariant is enforced (raises on construction)
* the legacy label bridge is conservative
"""

from __future__ import annotations

import pytest

from daph_learning.autolearn.outcome import (
    BACKENDS,
    BackendExecutionRecord,
    OutcomeSemantics,
    build_outcome_semantics,
)
from daph_learning.evaluation.verification import (
    NumericVerifier,
    RouteSuitabilityVerifier,
    VerificationResult,
    VerificationStatus,
)


def _task(expected=42, oracle="symbolic"):
    return {
        "task_id": "t1",
        "expected": expected,
        "capability_ids": ["integer_arithmetic"],
        "utility_oracle": oracle,
    }


# --- Invariant: an unexecuted backend can never be correct ---

def test_unexecuted_backend_record_is_never_correct():
    rec = BackendExecutionRecord(
        backend="llm",
        selected=False,
        executed=False,
    )
    assert rec.is_correct is False
    assert rec.verification.status is VerificationStatus.NOT_EXECUTED


def test_unexecuted_backend_cannot_be_marked_verified_correct():
    """Constructing an unexecuted record with VERIFIED_CORRECT raises."""
    with pytest.raises(ValueError, match="protocol invariant"):
        BackendExecutionRecord(
            backend="llm",
            selected=True,
            executed=False,
            verification=VerificationResult(
                status=VerificationStatus.VERIFIED_CORRECT,
                verifier="NumericVerifier",
            ),
        )


def test_unexecuted_backend_must_carry_not_executed_status():
    """An unexecuted backend cannot carry a stale verification status."""
    with pytest.raises(ValueError, match="protocol invariant"):
        BackendExecutionRecord(
            backend="llm",
            selected=True,
            executed=False,
            verification=VerificationResult(
                status=VerificationStatus.UNVERIFIABLE,
                verifier="NumericVerifier",
            ),
        )


def test_skipped_backend_never_marked_correct_via_build():
    """Acceptance test: a skipped backend is never marked correct.

    The router selected symbolic; the LLM backend was skipped (not in the
    executed_backends mapping). The LLM record must be NOT_EXECUTED and
    is_correct must be False — never True.
    """
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True},
        },
    )
    llm_rec = outcome.backends["llm"]
    assert llm_rec.executed is False
    assert llm_rec.verification.status is VerificationStatus.NOT_EXECUTED
    assert llm_rec.is_correct is False
    assert outcome.backend_correct("llm") is False


def test_selected_but_unexecuted_backend_never_correct():
    """Router selected LLM but LLM was skipped -> selected_backend_correct False."""
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="llm",
        executed_backends={},  # nothing executed
    )
    assert outcome.selected_backend_executed is False
    assert outcome.selected_backend_correct is False


# --- The four semantics are distinct ---

def test_four_semantics_are_distinct_and_inspectable():
    task = _task(expected=42, oracle="symbolic")
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True,
                         "latency_ms": 1.0, "compute_cost": 0.0},
            "llm": {"output": "FINAL: 99", "execution_succeeded": True,
                    "latency_ms": 50.0, "compute_cost": 0.01},
        },
    )
    # 1. backend_selected
    assert outcome.backend_selected == "symbolic"
    # 2. backend_executed (per-backend)
    assert outcome.backends["symbolic"].executed is True
    assert outcome.backends["llm"].executed is True
    assert outcome.both_backends_executed is True
    # 3. verification_status (per-backend, independent)
    assert outcome.backends["symbolic"].verification.status is VerificationStatus.VERIFIED_CORRECT
    assert outcome.backends["llm"].verification.status is VerificationStatus.VERIFIED_INCORRECT
    # 4. route_suitability (oracle-based, separate from correctness)
    assert outcome.route_suitability.status is VerificationStatus.VERIFIED_CORRECT
    assert outcome.route_suitability.expected == "symbolic"
    assert outcome.route_suitability.observed == "symbolic"


def test_route_suitability_separate_from_backend_correctness():
    """A misrouted-but-correct task: route suitability INCORRECT, backend
    verification CORRECT. The two must not collapse."""
    task = _task(expected=42, oracle="llm")  # oracle says LLM
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",  # but router picked symbolic
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True},
        },
    )
    # backend correctness: symbolic executed and correct
    assert outcome.backends["symbolic"].is_correct is True
    # route suitability: symbolic != llm oracle -> incorrect
    assert outcome.route_suitability.status is VerificationStatus.VERIFIED_INCORRECT


# --- Both backends executed exactly once (counterfactual) ---

def test_both_backends_executed_exactly_once():
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True},
            "llm": {"output": "FINAL: 42", "execution_succeeded": True},
        },
    )
    assert set(outcome.backends) == set(BACKENDS)
    assert outcome.both_backends_executed is True
    # each backend appears exactly once
    assert len(outcome.backends) == 2


# --- Legacy label bridge is conservative ---

def test_legacy_label_correct_only_when_selected_executed_verified():
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True},
        },
    )
    assert outcome.to_legacy_label() == "correct"


def test_legacy_label_unverifiable_when_selected_backend_skipped():
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="llm",
        executed_backends={},  # llm skipped
    )
    assert outcome.to_legacy_label() == "unverifiable"


def test_legacy_label_misrouted_when_selected_backend_wrong():
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 99", "execution_succeeded": True},
        },
    )
    assert outcome.to_legacy_label() == "misrouted"


# --- Abstain route executes neither backend ---

def test_abstain_route_executes_neither_backend():
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="abstain",
        executed_backends={},
    )
    assert outcome.backend_selected == "abstain"
    assert outcome.both_backends_executed is False
    assert all(not outcome.backends[b].executed for b in BACKENDS)
    assert outcome.selected_backend_record is None
    # neither backend correct
    assert outcome.backend_correct("symbolic") is False
    assert outcome.backend_correct("llm") is False


# --- Verifier independence: the router must not be its own verifier ---

def test_custom_verifier_used_when_supplied():
    class AlwaysIncorrect:
        def verify(self, task, output, *, execution_succeeded=True):
            return VerificationResult(
                status=VerificationStatus.VERIFIED_INCORRECT,
                verifier="AlwaysIncorrect",
                expected=task.get("expected"),
                observed=output,
            )

    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True},
        },
        verifier=AlwaysIncorrect(),  # type: ignore[arg-type]
    )
    assert outcome.backends["symbolic"].verification.verifier == "AlwaysIncorrect"
    assert outcome.backends["symbolic"].is_correct is False


def test_default_verifier_is_numeric_not_router():
    """The default verifier is NumericVerifier, never the router itself."""
    task = _task()
    outcome = build_outcome_semantics(
        task,
        backend_selected="symbolic",
        executed_backends={
            "symbolic": {"output": "FINAL: 42", "execution_succeeded": True},
        },
    )
    assert outcome.backends["symbolic"].verification.verifier == "NumericVerifier"


# --- Changing the expected_field / oracle_field changes the result ---

def test_changing_oracle_field_changes_route_suitability():
    task = {
        "task_id": "t1",
        "expected": 42,
        "capability_ids": ["integer_arithmetic"],
        "utility_oracle": "llm",
        "accuracy_oracle": "symbolic",
    }
    # default oracle_field = utility_oracle -> suitability correct for llm
    o1 = build_outcome_semantics(
        task, backend_selected="llm",
        executed_backends={"llm": {"output": "FINAL: 42", "execution_succeeded": True}},
    )
    # oracle_field = accuracy_oracle -> suitability incorrect for llm
    o2 = build_outcome_semantics(
        task, backend_selected="llm",
        executed_backends={"llm": {"output": "FINAL: 42", "execution_succeeded": True}},
        route_suitability_verifier=RouteSuitabilityVerifier(oracle_field="accuracy_oracle"),
    )
    assert o1.route_suitability.status is VerificationStatus.VERIFIED_CORRECT
    assert o2.route_suitability.status is VerificationStatus.VERIFIED_INCORRECT
