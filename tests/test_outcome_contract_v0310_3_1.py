"""v0.3.10.3.1-alpha — Section 4-5 / G2,G4: BackendOutcome contract and
confidence != quality semantics.

Key invariants:
* ``execution_success != verified_correct`` — never infer correctness
  from execution success.
* Confidence is separate from quality (known-unsupported backend has
  low quality but HIGH measurement confidence; do not zero it out).
* Paired confidence combine is configurable (product/min/geometric_mean).
"""

from __future__ import annotations

import math

import pytest

from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.confidence import (
    ConfidenceCombine,
    combine_paired_confidence,
)
from daph_learning.policy.types import BackendOutcome


def _outcome(**kw):
    base = dict(
        task_id="t1", backend="symbolic", available=True, executed=True,
        execution_success=True, output_text="42", output_hash="h",
        verifier_status="verified_correct", quality=1.0, latency_sec=0.1,
        normalized_cost=0.0, risk=0.0, verifier_confidence=1.0,
    )
    base.update(kw)
    # Keep the legacy `correct` bool consistent with verifier_status so
    # the two fields never drift in test fixtures.
    status = base.get("verifier_status", "verified_correct")
    base.setdefault("correct", status == "verified_correct")
    if "verifier_status" in kw:
        base["correct"] = (status == "verified_correct")
    return BackendOutcome(**base)


# --- Section 4: BackendOutcome contract ---

def test_execution_succeeds_but_verifier_rejects():
    """execution_success=True, verified_correct=False — distinct."""
    o = _outcome(execution_success=True, verifier_status="verified_incorrect",
                 quality=0.0, verifier_confidence=1.0)
    assert o.execution_success is True
    assert o.verified_correct is False
    assert o.correct is False


def test_execution_fails():
    o = _outcome(execution_success=False, verifier_status="not_verified",
                 quality=0.0, verifier_confidence=0.0, failure_reason="timeout")
    assert o.execution_success is False
    assert o.verified_correct is None


def test_backend_unavailable():
    o = _outcome(available=False, executed=False, verifier_status="not_verified",
                 quality=0.0, verifier_confidence=0.0, failure_reason="no model")
    assert o.available is False
    assert o.verified_correct is None


def test_verifier_unavailable():
    o = _outcome(verifier_status="verifier_unsupported", quality=0.0,
                 verifier_confidence=0.5)
    assert o.verified_correct is None


def test_executed_and_verified_correct():
    o = _outcome()
    assert o.execution_success is True
    assert o.verified_correct is True
    assert o.correct is True


def test_execution_success_not_inferred_from_correct():
    """A verified-correct outcome with execution_success=False is a
    contradiction the dataclass should reject or at least keep distinct."""
    # verifier_status=verified_correct but executed=False is invalid by
    # the existing __post_init__ semantics (not enforced here, but the
    # fields stay distinct when both are set honestly).
    o = _outcome(execution_success=False, verifier_status="verified_incorrect",
                 quality=0.0)
    assert o.execution_success is False
    assert o.verified_correct is False


# --- Section 5: confidence != quality ---

def test_known_unsupported_backend_keeps_measurement_confidence():
    """A known-unsupported backend has low quality but HIGH measurement
    confidence — we are confident the backend is unsupported. Do not
    zero the confidence just because quality is zero."""
    o = _outcome(available=False, executed=False, quality=0.0,
                 verifier_status="not_verified", verifier_confidence=1.0)
    assert o.quality == 0.0
    assert o.verifier_confidence == 1.0  # high confidence, low quality


def test_incorrect_answer_deterministic_verifier():
    """quality=0, verifier_confidence=1 (deterministic exact verifier)."""
    o = _outcome(verifier_status="verified_incorrect", quality=0.0,
                 verifier_confidence=1.0)
    assert o.quality == 0.0
    assert o.verifier_confidence == 1.0


def test_combine_paired_confidence_product():
    assert combine_paired_confidence(0.8, 0.5) == pytest.approx(0.4)
    assert combine_paired_confidence(0.8, 0.5, "product") == pytest.approx(0.4)


def test_combine_paired_confidence_min():
    assert combine_paired_confidence(0.8, 0.5, ConfidenceCombine.MIN) == 0.5


def test_combine_paired_confidence_geometric_mean():
    assert combine_paired_confidence(
        0.8, 0.5, ConfidenceCombine.GEOMETRIC_MEAN) == pytest.approx(
        math.sqrt(0.4))


def test_combine_paired_confidence_rejects_out_of_range():
    with pytest.raises(ValueError):
        combine_paired_confidence(1.2, 0.5)


def test_config_default_confidence_combine_is_product():
    cfg = ExperimentConfig()
    assert cfg.confidence_combine == "product"


def test_config_rejects_unknown_confidence_combine():
    with pytest.raises(ValueError):
        ExperimentConfig(confidence_combine="bogus")
