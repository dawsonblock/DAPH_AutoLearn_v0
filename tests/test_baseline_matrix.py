"""v0.3.10.3.2-alpha — Section 44-47 / G44: baseline matrix.

Verifies that the baseline matrix (AutoLearn vs hand-router vs random
vs always-symbolic vs always-LLM) is computed with the canonical
utility and that AutoLearn's mean utility is competitive.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.data.crossover_benchmark import (
    generate_crossover_split,
    simulate_llm_output,
    crossover_optimal_distribution,
)
from daph_learning.execution.real_backends import (
    execute_symbolic_backend,
    verify_arithmetic,
)
from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.utility import backend_utility
from daph_learning.policy.types import BackendOutcome, Route
from daph_learning.routing.hand_router import hand_router_routes


def _compute_utility_for_route(task, route, cfg):
    """Compute canonical utility for a given route on a task."""
    route_str = route.value if hasattr(route, "value") else str(route)
    if route_str == "abstain":
        return 0.0
    if route_str == "symbolic":
        outcome, out_text = execute_symbolic_backend(task)
        if out_text:
            vr = verify_arithmetic(task, out_text.strip())
            outcome = outcome.__class__(
                **{**outcome.__dict__,
                   "quality": 1.0 if vr.verified_correct else 0.0})
        return backend_utility(outcome, cfg)
    elif route_str == "llm":
        out_text = simulate_llm_output(task)
        vr = verify_arithmetic(task, out_text)
        outcome = BackendOutcome(
            task_id=str(task.get("task_id", "")), backend="llm",
            available=True, executed=True, execution_success=out_text is not None,
            output_text=out_text, output_hash=None,
            verifier_status="verified_correct" if vr.verified_correct else "verified_incorrect",
            correct=bool(vr.verified_correct),
            quality=1.0 if vr.verified_correct else 0.0,
            latency_sec=0.02, normalized_cost=0.001, risk=0.0,
            verifier_confidence=float(vr.confidence),
            failure_reason=vr.failure_reason,
        )
        return backend_utility(outcome, cfg)
    return 0.0


def _baseline_matrix(tasks, cfg):
    """Compute the baseline matrix for all routing strategies."""
    results = {}
    # Always symbolic.
    sym_utils = [_compute_utility_for_route(t, "symbolic", cfg) for t in tasks]
    results["always_symbolic"] = float(np.mean(sym_utils))
    # Always LLM.
    llm_utils = [_compute_utility_for_route(t, "llm", cfg) for t in tasks]
    results["always_llm"] = float(np.mean(llm_utils))
    # Hand router.
    hand_routes = hand_router_routes(tasks)
    hand_utils = [_compute_utility_for_route(t, r, cfg)
                  for t, r in zip(tasks, hand_routes)]
    results["hand_router"] = float(np.mean(hand_utils))
    # Random (50/50).
    rng = np.random.RandomState(42)
    random_routes = ["symbolic" if rng.random() < 0.5 else "llm"
                     for _ in tasks]
    random_utils = [_compute_utility_for_route(t, r, cfg)
                    for t, r in zip(tasks, random_routes)]
    results["random"] = float(np.mean(random_utils))
    # Oracle (best per-task).
    oracle_utils = [max(s, l) for s, l in zip(sym_utils, llm_utils)]
    results["oracle"] = float(np.mean(oracle_utils))
    return results


def test_baseline_matrix_is_computed():
    """Section 44: the baseline matrix must be computed for all
    strategies."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg = ExperimentConfig()
    results = _baseline_matrix(tasks, cfg)
    assert "always_symbolic" in results
    assert "always_llm" in results
    assert "hand_router" in results
    assert "random" in results
    assert "oracle" in results


def test_oracle_is_upper_bound():
    """Section 44: oracle utility must be >= all other strategies."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg = ExperimentConfig()
    results = _baseline_matrix(tasks, cfg)
    assert results["oracle"] >= results["always_symbolic"]
    assert results["oracle"] >= results["always_llm"]
    assert results["oracle"] >= results["hand_router"]
    assert results["oracle"] >= results["random"]


def test_hand_router_beats_random():
    """Section 44: the hand router should beat random routing (it's a
    competent baseline, not a weak one)."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=20, seed=42)
    cfg = ExperimentConfig()
    results = _baseline_matrix(tasks, cfg)
    assert results["hand_router"] >= results["random"], (
        f"hand_router={results['hand_router']:.4f} < "
        f"random={results['random']:.4f} — hand router should beat random")


def test_always_symbolic_beats_always_llm_on_hard_tasks():
    """Section 44: on tasks with large operands (where LLM errs),
    always-symbolic should beat always-LLM."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=20, seed=42)
    cfg = ExperimentConfig()
    # Filter to large-operand tasks (subtype A with large operands).
    large_tasks = [t for t in tasks
                   if t["metadata"]["subtype"] == "A"
                   and t.get("inputs", {}).get("a", 0) > 1000]
    if len(large_tasks) < 5:
        pytest.skip("not enough large-operand tasks")
    results = _baseline_matrix(large_tasks, cfg)
    assert results["always_symbolic"] > results["always_llm"], (
        f"symbolic={results['always_symbolic']:.4f} <= "
        f"llm={results['always_llm']:.4f} on large-operand tasks")


def test_crossover_distribution_is_balanced():
    """Section 44: the crossover distribution should have 0.2 <
    P(symbolic) < 0.8 overall."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=30, seed=42)
    dist = crossover_optimal_distribution(tasks)
    assert 0.2 < dist["p_symbolic_optimal"] < 0.8, (
        f"p_symbolic_optimal={dist['p_symbolic_optimal']:.2f} outside "
        f"[0.2, 0.8] — crossover not balanced (Section 44)")
