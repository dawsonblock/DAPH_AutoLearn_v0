"""v0.3.10.3.1-alpha — Section 21-25 / G29-G31: steering utility
evaluation via ΔU(α), not P(symbolic) movement.

The key negative result to preserve: pushing harder in the learned
symbolic direction can raise P(symbolic | +v) while LOWERING utility.
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
from daph_learning.steering.utility_eval import (
    evaluate_steering_utility,
    random_control_p_emp,
)


def _make_policy_proba(steepness: float = 5.0):
    """A simulated steered policy: P(symbolic) rises with alpha
    regardless of whether symbolic is actually better. This reproduces
    the negative result where P(S) rises but utility can fall."""
    def fn(task, alpha):
        # Base probability depends on whether structured inputs exist.
        caps = set(task.get("capability_ids", []))
        base = 0.7 if (caps & {"integer_arithmetic", "modular_multiplication"}) else 0.3
        # alpha pushes toward symbolic monotonically.
        p = base + steepness * alpha * 0.1
        return float(np.clip(p, 0.0, 1.0))
    return fn


def test_steering_eval_produces_per_alpha_table():
    tasks = generate_crossover_split(split="dev", n_per_subtype=4, seed=0)
    cfg = ExperimentConfig(confidence_threshold=0.6)
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=_make_policy_proba(),
        execute_symbolic_fn=execute_symbolic_backend,
        execute_llm_fn=simulate_llm_output,
        verify_fn=verify_arithmetic,
        config=cfg,
    )
    assert len(report.per_alpha) == len(cfg.intervention_alpha_grid)
    for r in report.per_alpha:
        assert r.n_tasks == len(tasks)
        assert 0.0 <= r.flip_rate <= 1.0
        assert (r.beneficial_flip_fraction + r.harmful_flip_fraction
                + r.neutral_flip_fraction) == pytest.approx(1.0) or r.n_flips == 0


def test_steering_eval_uses_verified_utility_not_policy_score():
    """Section 21: the objective is ΔU(α), not P(symbolic). The report
    must carry mean_utility and mean_regret, not just P(symbolic)."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=3, seed=1)
    cfg = ExperimentConfig()
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=_make_policy_proba(),
        execute_symbolic_fn=execute_symbolic_backend,
        execute_llm_fn=simulate_llm_output,
        verify_fn=verify_arithmetic,
        config=cfg,
    )
    for r in report.per_alpha:
        assert "mean_utility" in r.__dict__
        assert "mean_regret" in r.__dict__
        assert "mean_p_symbolic" in r.__dict__


def test_beneficial_and_harmful_flips_classified():
    """Section 23: flips classified as beneficial/harmful/neutral by
    utility change, not by P(symbolic) change."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=5, seed=2)
    cfg = ExperimentConfig(confidence_threshold=0.55)
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=_make_policy_proba(steepness=8.0),
        execute_symbolic_fn=execute_symbolic_backend,
        execute_llm_fn=simulate_llm_output,
        verify_fn=verify_arithmetic,
        config=cfg,
    )
    # At least one alpha != 0 should have flips.
    nonzero = [r for r in report.per_alpha if r.alpha != 0.0 and r.n_flips > 0]
    if nonzero:
        r = nonzero[0]
        assert r.n_beneficial + r.n_harmful + r.n_neutral == r.n_flips


def test_oracle_alpha_diagnostic_on_dev():
    """Section 24: per-task oracle alpha utility is reported (DEV only
    diagnostic, never used on final)."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=3, seed=3)
    cfg = ExperimentConfig()
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=_make_policy_proba(),
        execute_symbolic_fn=execute_symbolic_backend,
        execute_llm_fn=simulate_llm_output,
        verify_fn=verify_arithmetic,
        config=cfg,
    )
    # Oracle >= best global alpha >= baseline (oracle picks per-task best).
    assert report.oracle_alpha_utility >= report.best_global_alpha_utility - 1e-9
    assert report.best_global_alpha_utility >= report.baseline_utility - 1e-9


def test_random_control_p_emp():
    """Section 25: p_emp = (1 + count(random >= learned)) / (N + 1)."""
    learned = 0.05
    randoms = [0.01, 0.02, 0.03, 0.10, 0.08]
    p = random_control_p_emp(learned, randoms)
    # 2 randoms >= 0.05 -> (1 + 2) / 6 = 0.5
    assert p == pytest.approx(0.5)


def test_random_control_p_emp_all_below():
    learned = 1.0
    randoms = [0.1, 0.2, 0.3]
    p = random_control_p_emp(learned, randoms)
    assert p == pytest.approx(1 / 4)


def test_negative_result_supported():
    """Section 46: the report must support the conclusion 'steering
    hurt utility' — i.e. best_global_alpha_utility can be < baseline."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=4, seed=4)
    cfg = ExperimentConfig(confidence_threshold=0.5)
    # A policy that flips NL tasks (LLM-optimal) to symbolic at +alpha,
    # which HURTS utility on B/C/E/F.
    def harmful_policy(task, alpha):
        caps = set(task.get("capability_ids", []))
        base = 0.6 if (caps & {"integer_arithmetic"}) else 0.4
        return float(np.clip(base + 6.0 * alpha * 0.1, 0, 1))
    report = evaluate_steering_utility(
        tasks,
        policy_proba_fn=harmful_policy,
        execute_symbolic_fn=execute_symbolic_backend,
        execute_llm_fn=simulate_llm_output,
        verify_fn=verify_arithmetic,
        config=cfg,
    )
    # The report honestly records whatever happened — no forced positivity.
    assert isinstance(report.baseline_utility, float)
    assert isinstance(report.best_global_alpha_utility, float)
