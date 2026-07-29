"""v0.3.10.3.1-alpha — Section 20, 26 / G20-G21: KL release gate and
fail-closed behavior."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.interventions.kl_gate import (
    PromotionConstraints,
    evaluate_kl_capability_gate,
    kl_divergence,
    mean_kl_neutral,
)


# --- Section 26: neutral KL release gate ---

def test_kl_gate_utility_up_kl_exceeds_rejects():
    """Case A: utility improves, KL exceeds threshold -> reject."""
    r = evaluate_kl_capability_gate(
        mean_utility_gain=0.05, neutral_kl=0.10,
        n_samples=200, constraints=PromotionConstraints(max_neutral_kl=0.03))
    assert r.promoted is False
    assert "KL" in r.reason
    assert r.kl_threshold == 0.03


def test_kl_gate_utility_up_kl_below_allows():
    """Case B: utility improves, KL below threshold -> allowed."""
    r = evaluate_kl_capability_gate(
        mean_utility_gain=0.05, neutral_kl=0.01,
        n_samples=200, constraints=PromotionConstraints(max_neutral_kl=0.03))
    assert r.promoted is True
    assert r.mean_neutral_kl == 0.01


def test_kl_gate_kl_unavailable_fails_closed():
    """Case C: KL unavailable in qualified mode -> fail closed."""
    r = evaluate_kl_capability_gate(
        mean_utility_gain=0.05, neutral_kl=None,
        n_samples=200, constraints=PromotionConstraints(max_neutral_kl=0.03))
    assert r.promoted is False
    assert "N/A" in r.reason or "KL" in r.reason


def test_kl_gate_records_p95_and_threshold():
    r = evaluate_kl_capability_gate(
        mean_utility_gain=0.05, neutral_kl=0.01, p95_neutral_kl=0.025,
        n_samples=200, constraints=PromotionConstraints(max_neutral_kl=0.03))
    assert r.p95_neutral_kl == 0.025
    assert r.kl_threshold == 0.03


def test_kl_divergence_identical_distributions():
    p = np.array([0.5, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)


def test_mean_kl_neutral():
    base = [np.array([0.5, 0.5]), np.array([0.3, 0.7])]
    steered = [np.array([0.5, 0.5]), np.array([0.3, 0.7])]
    assert mean_kl_neutral(base, steered) == pytest.approx(0.0, abs=1e-6)


# --- Section 20: fail closed on missing utility/verifier/policy/executor ---

def test_utility_for_route_rejects_unknown_route():
    """Section 20: unknown route raises, not returns 0.0."""
    from daph_learning.policy.utility import utility_for_route
    from daph_learning.policy.types import BackendOutcome, CounterfactualExperience, Route
    from daph_learning.policy.config import ExperimentConfig
    cfg = ExperimentConfig()
    sym = BackendOutcome(task_id="t", backend="symbolic", quality=1.0)
    llm = BackendOutcome(task_id="t", backend="llm", quality=0.5)
    exp = CounterfactualExperience(
        task_id="t", symbolic=sym, llm=llm, delta_utility=0.5,
        preferred_action=Route.SYMBOLIC, sample_weight=0.5)
    with pytest.raises(ValueError):
        utility_for_route(exp, "bogus", cfg)


def test_steering_eval_requires_real_executors():
    """Section 20: the steering utility eval calls the provided
    executors; it does not silently substitute zeros."""
    from daph_learning.steering.utility_eval import evaluate_steering_utility
    from daph_learning.data.crossover_benchmark import (
        generate_crossover_split, simulate_llm_output)
    from daph_learning.execution.real_backends import (
        execute_symbolic_backend, verify_arithmetic)
    from daph_learning.policy.config import ExperimentConfig
    tasks = generate_crossover_split(split="dev", n_per_subtype=2, seed=0)
    called = {"sym": 0, "llm": 0}

    def sym_fn(t):
        called["sym"] += 1
        return execute_symbolic_backend(t)

    def llm_fn(t):
        called["llm"] += 1
        return simulate_llm_output(t)

    def prob_fn(t, alpha):
        return 0.5

    evaluate_steering_utility(
        tasks, policy_proba_fn=prob_fn, execute_symbolic_fn=sym_fn,
        execute_llm_fn=llm_fn, verify_fn=verify_arithmetic,
        config=ExperimentConfig())
    assert called["sym"] == len(tasks)
    assert called["llm"] == len(tasks)
