"""Section 13 — tests for surface and structured-feature baselines.

Tests the 10 required baselines at the lower level (not through the
full pipeline) to verify they produce correct routes and that
hidden-state baselines work with real feature vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.evaluation.baselines import (
    BASELINE_NAMES,
    StructuredTaskFeatures,
    always_llm,
    always_symbolic,
    extract_structured_features,
    hand_router,
    hidden_state_centroid,
    hidden_state_logistic,
    oracle,
    sham_hidden_state_logistic,
    structured_feature_logistic,
    subtype_only_logistic,
    surface_tfidf_logistic,
)
from daph_learning.policy.types import BackendOutcome, CounterfactualExperience, Route


def _make_experience(sym_correct=True, llm_correct=False) -> CounterfactualExperience:
    sym = BackendOutcome(
        task_id="t1", backend="symbolic", available=True, executed=True,
        execution_success=True, output_text="42", output_hash="h",
        verifier_status="verified_correct" if sym_correct else "verified_incorrect",
        correct=sym_correct, quality=1.0 if sym_correct else 0.0,
        latency_sec=0.01, normalized_cost=0.05, risk=0.0,
        verifier_confidence=1.0,
    )
    llm = BackendOutcome(
        task_id="t1", backend="llm", available=True, executed=True,
        execution_success=True, output_text="42", output_hash="h",
        verifier_status="verified_correct" if llm_correct else "verified_incorrect",
        correct=llm_correct, quality=1.0 if llm_correct else 0.0,
        latency_sec=0.1, normalized_cost=0.2, risk=0.0,
        verifier_confidence=1.0,
    )
    return CounterfactualExperience(
        task_id="t1", symbolic=sym, llm=llm,
        delta_utility=0.9 if sym_correct and not llm_correct else 0.0,
        preferred_action=Route.SYMBOLIC if sym_correct else Route.LLM,
        sample_weight=1.0,
    )


def _make_task(subtype="A", op="+", a=12, b=30):
    return {
        "task_id": "t1",
        "specification": f"{a} {op} {b}",
        "prompt": f"What is {a} {op} {b}?",
        "capability_ids": ["integer_arithmetic"],
        "inputs": {"a": a, "b": b, "op": op},
        "metadata": {"subtype": subtype},
    }


def test_baseline_names_count():
    assert len(BASELINE_NAMES) == 10


def test_always_llm():
    exp = _make_experience()
    assert always_llm(_make_task(), exp) == Route.LLM


def test_always_symbolic():
    exp = _make_experience()
    assert always_symbolic(_make_task(), exp) == Route.SYMBOLIC


def test_oracle_picks_higher_utility():
    exp = _make_experience(sym_correct=True, llm_correct=False)
    assert oracle(_make_task(), exp) == Route.SYMBOLIC
    exp2 = _make_experience(sym_correct=False, llm_correct=True)
    assert oracle(_make_task(), exp2) == Route.LLM


def test_hand_router_arithmetic():
    task = _make_task(subtype="A")
    route = hand_router(task, _make_experience())
    assert route in (Route.SYMBOLIC, Route.LLM)


def test_subtype_only_logistic():
    task = _make_task(subtype="A")
    weights = {"A": {"symbolic": 1.0, "llm": 0.0}}
    assert subtype_only_logistic(task, _make_experience(), weights=weights) == Route.SYMBOLIC
    weights2 = {"A": {"symbolic": 0.0, "llm": 1.0}}
    assert subtype_only_logistic(task, _make_experience(), weights=weights2) == Route.LLM


def test_surface_tfidf_logistic():
    task = _make_task()
    vocab = {"what": 0, "is": 1}
    weights = {"what": 1.0, "is": -1.0}
    route = surface_tfidf_logistic(task, _make_experience(), weights=weights, vocab=vocab)
    assert route in (Route.SYMBOLIC, Route.LLM)


def test_structured_feature_logistic():
    task = _make_task()
    feats = extract_structured_features(task)
    assert isinstance(feats, StructuredTaskFeatures)
    vec = feats.to_vector()
    assert len(vec) == 7
    route = structured_feature_logistic(
        task, _make_experience(), weights=[1.0] * 7, bias=0.0)
    assert route in (Route.SYMBOLIC, Route.LLM)


def test_extract_structured_features():
    task = _make_task(subtype="A", op="+", a=12, b=30)
    feats = extract_structured_features(task)
    assert feats.n_operands == 2
    assert feats.max_operand_digits == 2
    assert feats.has_modular is False
    assert feats.is_natural_language is False


def test_extract_structured_features_nl():
    task = _make_task(subtype="B")
    feats = extract_structured_features(task)
    assert feats.is_natural_language is True


def test_hidden_state_centroid():
    h = np.array([1.0, 0.0, 0.0])
    centroid_sym = np.array([1.0, 0.0, 0.0])
    centroid_llm = np.array([0.0, 1.0, 0.0])
    route = hidden_state_centroid(
        _make_task(), _make_experience(),
        centroid_sym=centroid_sym, centroid_llm=centroid_llm, h=h)
    assert route == Route.SYMBOLIC  # closer to symbolic centroid


def test_hidden_state_centroid_llm():
    h = np.array([0.0, 1.0, 0.0])
    centroid_sym = np.array([1.0, 0.0, 0.0])
    centroid_llm = np.array([0.0, 1.0, 0.0])
    route = hidden_state_centroid(
        _make_task(), _make_experience(),
        centroid_sym=centroid_sym, centroid_llm=centroid_llm, h=h)
    assert route == Route.LLM


def test_hidden_state_centroid_no_features():
    route = hidden_state_centroid(
        _make_task(), _make_experience())
    assert route == Route.LLM  # fallback


def test_hidden_state_logistic():
    h = np.array([1.0, -1.0, 0.5])
    weights = np.array([1.0, -1.0, 0.0])
    route = hidden_state_logistic(
        _make_task(), _make_experience(), weights=weights, bias=0.0, h=h)
    # score = 1*1 + (-1)*(-1) + 0*0.5 = 2 > 0 → SYMBOLIC
    assert route == Route.SYMBOLIC


def test_hidden_state_logistic_no_features():
    route = hidden_state_logistic(_make_task(), _make_experience())
    assert route == Route.LLM  # fallback


def test_sham_hidden_state_logistic_no_labels():
    """Without sham labels, routes randomly (signal destroyed)."""
    route = sham_hidden_state_logistic(_make_task(), _make_experience())
    assert route in (Route.SYMBOLIC, Route.LLM)


def test_sham_hidden_state_logistic_with_labels():
    """With sham labels, behaves like hidden_state_logistic."""
    h = np.array([1.0, -1.0])
    weights = np.array([1.0, -1.0])
    route = sham_hidden_state_logistic(
        _make_task(), _make_experience(),
        weights=weights, bias=0.0, h=h, sham_labels=True)
    assert route == Route.SYMBOLIC
