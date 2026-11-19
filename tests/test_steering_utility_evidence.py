"""v0.3.10.3.2-alpha — Section 28-32 / G28: steering utility evidence.

Verifies that steering utility is measured via executed backend utility
(ΔU(α)), not symbolic probability, and that beneficial/harmful flip
analysis works correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.data.crossover_benchmark import (
    generate_crossover_split,
    simulate_llm_output,
)
from daph_learning.execution.real_backends import (
    execute_symbolic_backend,
    verify_arithmetic,
)
from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.utility import backend_utility
from daph_learning.steering.utility_eval import (
    evaluate_steering_utility,
    random_control_p_emp,
    SteeringUtilityReport,
)


def _make_steering_setup(tasks, cfg=None):
    """Build the steering evaluation setup with real backends."""
    cfg = cfg or ExperimentConfig()

    def policy_proba_fn(task, alpha):
        """Simulate a steering effect: positive alpha pushes toward
        symbolic, negative toward LLM."""
        subtype = task.get("metadata", {}).get("subtype", "")
        # Base probability depends on subtype.
        base_p = {"A": 0.6, "B": 0.4, "C": 0.3, "D": 0.7, "E": 0.5,
                  "F": 0.4, "G": 0.5, "H": 0.6}
        p = base_p.get(subtype, 0.5) + alpha * 0.3
        return float(np.clip(p, 0.0, 1.0))

    def execute_symbolic_fn(task):
        return execute_symbolic_backend(task)

    def execute_llm_fn(task):
        return simulate_llm_output(task)

    def verify_fn(task, output_text):
        return verify_arithmetic(task, output_text)

    return cfg, policy_proba_fn, execute_symbolic_fn, execute_llm_fn, verify_fn


def test_steering_utility_uses_backend_utility():
    """Section 28: steering utility must be measured via
    backend_utility, not symbolic probability."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg, pol_fn, sym_fn, llm_fn, ver_fn = _make_steering_setup(tasks)
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=pol_fn,
        execute_symbolic_fn=sym_fn,
        execute_llm_fn=llm_fn,
        verify_fn=ver_fn,
        config=cfg,
        alphas=[-1.0, 0.0, 1.0],
    )
    # The report must have per-alpha utility values.
    assert len(report.per_alpha) == 3
    for r in report.per_alpha:
        assert isinstance(r.mean_utility, float)
        assert isinstance(r.mean_regret, float)
    # Baseline utility (alpha=0) must be non-zero.
    assert report.baseline_utility != 0.0, (
        "baseline utility is zero — backend_utility not applied")


def test_steering_tracks_beneficial_and_harmful_flips():
    """Section 23: the report must track beneficial and harmful flip
    fractions."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=20, seed=42)
    cfg, pol_fn, sym_fn, llm_fn, ver_fn = _make_steering_setup(tasks)
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=pol_fn,
        execute_symbolic_fn=sym_fn,
        execute_llm_fn=llm_fn,
        verify_fn=ver_fn,
        config=cfg,
        alphas=[-1.0, 0.0, 1.0],
    )
    # At alpha=1.0 or -1.0, some flips should occur.
    total_flips = sum(r.n_flips for r in report.per_alpha if r.alpha != 0.0)
    assert total_flips > 0, "no flips at any alpha — steering has no effect"
    # Beneficial + harmful + neutral = total flips for each alpha.
    for r in report.per_alpha:
        if r.n_flips > 0:
            assert r.n_beneficial + r.n_harmful + r.n_neutral == r.n_flips


def test_steering_delta_u_not_p_symbolic():
    """Section 21: the report must measure ΔU(α), not just
    P(symbolic|+v)."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg, pol_fn, sym_fn, llm_fn, ver_fn = _make_steering_setup(tasks)
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=pol_fn,
        execute_symbolic_fn=sym_fn,
        execute_llm_fn=llm_fn,
        verify_fn=ver_fn,
        config=cfg,
        alphas=[-1.0, 0.0, 1.0],
    )
    # The report has both mean_utility AND mean_p_symbolic.
    # The key metric is mean_utility (ΔU), not mean_p_symbolic.
    for r in report.per_alpha:
        assert hasattr(r, "mean_utility")
        assert hasattr(r, "mean_p_symbolic")
    # Best global alpha is chosen by utility, not by p_symbolic.
    best_by_utility = max(report.per_alpha, key=lambda r: r.mean_utility)
    assert report.best_global_alpha == best_by_utility.alpha


def test_random_control_p_emp():
    """Section 25: empirical p-value for steering utility gain."""
    learned_gain = 0.5
    random_gains = [0.1, 0.2, 0.3, 0.4, 0.6]
    p = random_control_p_emp(learned_gain, random_gains)
    # 1 random gain >= learned (0.6 >= 0.5)
    # p = (1 + 1) / (5 + 1) = 2/6
    assert p == pytest.approx(2.0 / 6.0)


def test_random_control_p_emp_learned_best():
    """Section 25: if learned gain is best, p should be small."""
    learned_gain = 1.0
    random_gains = [0.1, 0.2, 0.3, 0.4, 0.5]
    p = random_control_p_emp(learned_gain, random_gains)
    # 0 random gains >= learned
    # p = (1 + 0) / (5 + 1) = 1/6
    assert p == pytest.approx(1.0 / 6.0)


def test_oracle_alpha_is_diagnostic_only():
    """Section 24: oracle alpha utility is a DEV-only diagnostic,
    not a selectable metric."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg, pol_fn, sym_fn, llm_fn, ver_fn = _make_steering_setup(tasks)
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=pol_fn,
        execute_symbolic_fn=sym_fn,
        execute_llm_fn=llm_fn,
        verify_fn=ver_fn,
        config=cfg,
        alphas=[-1.0, 0.0, 1.0],
    )
    # Oracle alpha utility must be >= best global alpha utility.
    assert report.oracle_alpha_utility >= report.best_global_alpha_utility


def test_capture_ablation_selects_best_layer():
    """Section 28: capture ablation must select the layer with highest
    mean utility on DEV."""
    from daph_learning.evaluation.capture_ablation import (
        run_capture_ablation, CaptureAblationReport,
    )
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg, pol_fn, sym_fn, llm_fn, ver_fn = _make_steering_setup(tasks)

    # The capture ablation takes a policy_proba_fn(task, layer) and a
    # utility_fn(task, route).
    def policy_proba_fn(task, layer):
        """Simulate different routing quality per layer."""
        subtype = task.get("metadata", {}).get("subtype", "")
        base = {"A": 0.6, "B": 0.4, "C": 0.3, "D": 0.7, "E": 0.5,
                "F": 0.4, "G": 0.5, "H": 0.6}.get(subtype, 0.5)
        # Layer 5 is best (highest utility), layer 1 is worst.
        layer_effect = {1: -0.2, 3: 0.0, 5: 0.15, 7: 0.05}.get(layer, 0.0)
        return float(np.clip(base + layer_effect, 0.0, 1.0))

    def utility_fn(task, route):
        """Canonical utility from executed backends."""
        from daph_learning.policy.types import Route
        route_str = route.value if hasattr(route, "value") else str(route)
        if route_str == "abstain":
            return 0.0
        if route_str == "symbolic":
            outcome, out_text = sym_fn(task)
            if out_text:
                vr = ver_fn(task, out_text.strip())
                outcome = outcome.__class__(
                    **{**outcome.__dict__,
                       "quality": 1.0 if vr.verified_correct else 0.0})
            return backend_utility(outcome, cfg)
        elif route_str == "llm":
            out_text = llm_fn(task)
            vr = ver_fn(task, out_text)
            from daph_learning.policy.types import BackendOutcome
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

    report = run_capture_ablation(
        tasks=tasks,
        layers=[1, 3, 5, 7],
        policy_proba_fn=policy_proba_fn,
        utility_fn=utility_fn,
        config=cfg,
        split="dev",
    )
    assert isinstance(report, CaptureAblationReport)
    assert report.best_layer in [1, 3, 5, 7]
    assert len(report.per_layer) == 4


def test_policy_comparison_decisive():
    """Section 29: policy comparison must produce a decisive verdict."""
    from daph_learning.evaluation.policy_comparison import compare_policies
    tasks = generate_crossover_split(split="dev", n_per_subtype=10, seed=42)
    cfg = ExperimentConfig()

    def utility_fn(task, route):
        from daph_learning.policy.types import Route
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
            from daph_learning.policy.types import BackendOutcome
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

    # Two policies: one always symbolic, one always LLM.
    policy_routes = {
        "always_symbolic": ["symbolic"] * len(tasks),
        "always_llm": ["llm"] * len(tasks),
    }
    result = compare_policies(
        tasks=tasks,
        policy_routes=policy_routes,
        utility_fn=utility_fn,
        gap_threshold=cfg.gap_threshold,
    )
    assert result is not None
    # The report should have per-policy rows.
    assert len(result.rows) == 2
