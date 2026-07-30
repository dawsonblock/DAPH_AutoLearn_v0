"""Section 13 — baselines tests."""

from __future__ import annotations

import pytest

from daph_learning.evaluation.baselines import (
    BASELINE_NAMES,
    StructuredTaskFeatures,
    always_llm,
    always_symbolic,
    extract_structured_features,
    hand_router,
    oracle,
    structured_feature_logistic,
    subtype_only_logistic,
    surface_tfidf_logistic,
)
from daph_learning.policy.types import BackendOutcome, CounterfactualExperience, Route


def _outcome(backend="symbolic", correct=True):
    quality = 1.0 if correct else 0.0
    return BackendOutcome(
        task_id="t", backend=backend, available=True, executed=True,
        execution_success=True, output_text="42", output_hash="x",
        verifier_status="verified_correct" if correct else "verified_incorrect",
        correct=correct, quality=quality, latency_sec=0.01,
        normalized_cost=0.05, risk=0.0, verifier_confidence=1.0)


def _experience(sym_correct=True, llm_correct=False):
    sym = _outcome("symbolic", sym_correct)
    llm = _outcome("llm", llm_correct)
    du = sym.quality - llm.quality
    if du > 0.02:
        preferred = Route.SYMBOLIC
        weight = du
    elif du < -0.02:
        preferred = Route.LLM
        weight = -du
    else:
        preferred = Route.ABSTAIN
        weight = 0.0
    return CounterfactualExperience(
        task_id="t", symbolic=sym, llm=llm,
        delta_utility=du, preferred_action=preferred,
        sample_weight=weight)


def _task(subtype="A", a=100, b=200, op="+"):
    return {
        "task_id": "t1",
        "specification": f"Compute {a} {op} {b}.",
        "inputs": {"a": a, "b": b, "op": op},
        "metadata": {"subtype": subtype},
    }


def test_all_10_baselines_exist():
    assert len(BASELINE_NAMES) == 10
    for name in BASELINE_NAMES:
        assert name  # non-empty


def test_always_llm():
    assert always_llm(_task(), _experience()) == Route.LLM


def test_always_symbolic():
    assert always_symbolic(_task(), _experience()) == Route.SYMBOLIC


def test_oracle_picks_higher_quality():
    exp = _experience(sym_correct=True, llm_correct=False)
    assert oracle(_task(), exp) == Route.SYMBOLIC
    exp2 = _experience(sym_correct=False, llm_correct=True)
    assert oracle(_task(), exp2) == Route.LLM


def test_hand_router_routes():
    # Just check it returns a Route.
    result = hand_router(_task(), _experience())
    assert result in (Route.SYMBOLIC, Route.LLM, Route.ABSTAIN)


def test_subtype_only_logistic():
    weights = {"A": {"symbolic": 0.8, "llm": 0.2}}
    assert subtype_only_logistic(_task("A"), _experience(), weights=weights) == Route.SYMBOLIC
    weights2 = {"A": {"symbolic": 0.2, "llm": 0.8}}
    assert subtype_only_logistic(_task("A"), _experience(), weights=weights2) == Route.LLM


def test_surface_tfidf_logistic():
    weights = {"compute": 1.0}
    vocab = {"compute": 0}
    assert surface_tfidf_logistic(_task(), _experience(), weights=weights, vocab=vocab) == Route.SYMBOLIC


def test_structured_feature_logistic():
    # Large operands → symbolic (positive weight on max_operand_digits)
    weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    task = _task(a=99999, b=88888)
    assert structured_feature_logistic(task, _experience(), weights=weights) == Route.SYMBOLIC


def test_extract_structured_features():
    feats = extract_structured_features(_task(a=99999, b=88888, op="%"))
    assert feats.max_operand_digits == 5
    assert feats.has_modular is True
    assert feats.operand_magnitude_bucket == 2


def test_structured_features_to_vector():
    feats = StructuredTaskFeatures(
        n_operands=2, max_operand_digits=5, has_modular=True,
        has_comparison=False, is_natural_language=False,
        n_steps=1, operand_magnitude_bucket=2)
    v = feats.to_vector()
    assert len(v) == 7
    assert v[0] == 2.0
    assert v[2] == 1.0  # has_modular
