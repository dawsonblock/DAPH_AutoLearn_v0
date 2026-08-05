"""v0.3.10.3.1-alpha — Section 28, 29-31: capture ablation + decisive
policy comparison + tie-aware metrics + ESS."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.data.crossover_benchmark import (
    generate_crossover_split,
    simulate_llm_output,
)
from daph_learning.evaluation.capture_ablation import (
    run_capture_ablation,
)
from daph_learning.evaluation.policy_comparison import (
    compare_policies,
)
from daph_learning.execution.real_backends import (
    execute_symbolic_backend,
    verify_arithmetic,
)
from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.utility import backend_utility
from daph_learning.data.crossover_benchmark import _apply_verification


def _utility_fn_factory(tasks):
    """Build a utility_fn(task, route) that uses pre-executed backends."""
    cache = {}
    for t in tasks:
        tid = t["task_id"]
        sym_out, sym_text = execute_symbolic_backend(t)
        sym_vr = verify_arithmetic(t, sym_text)
        sym_out = _apply_verification(sym_out, sym_vr)
        llm_text = simulate_llm_output(t)
        llm_vr = verify_arithmetic(t, llm_text)
        from daph_learning.policy.types import BackendOutcome
        from daph_learning.data.crossover_benchmark import _verifier_status
        llm_out = BackendOutcome(
            task_id=tid, backend="llm", available=True, executed=True,
            execution_success=llm_text is not None, output_text=llm_text,
            output_hash=None, verifier_status=_verifier_status(llm_vr),
            correct=bool(llm_vr.verified_correct is True),
            quality=1.0 if llm_vr.verified_correct is True else 0.0,
            latency_sec=0.05, normalized_cost=0.1, risk=0.0,
            verifier_confidence=float(llm_vr.confidence),
            failure_reason=llm_vr.failure_reason)
        cache[tid] = (sym_out, llm_out)
    cfg = ExperimentConfig()

    def fn(task, route):
        tid = task["task_id"]
        sym_out, llm_out = cache[tid]
        if route == "symbolic":
            return backend_utility(sym_out, cfg)
        if route == "llm":
            return backend_utility(llm_out, cfg)
        if route == "abstain":
            return 0.0
        raise ValueError(f"unknown route {route!r}")
    return fn


# --- Section 28: capture representation ablation ---

def test_capture_ablation_selects_best_on_dev():
    tasks = generate_crossover_split(split="dev", n_per_subtype=5, seed=0)
    util_fn = _utility_fn_factory(tasks)

    def prob_fn(task, layer):
        # Simulate: higher layers are better for structured tasks.
        caps = set(task.get("capability_ids", []))
        base = 0.7 if caps else 0.3
        return float(np.clip(base + 0.02 * layer, 0, 1))

    report = run_capture_ablation(
        tasks, layers=[0, 5, 10, 15], policy_proba_fn=prob_fn,
        utility_fn=util_fn, config=ExperimentConfig(), split="dev")
    assert len(report.per_layer) == 4
    assert report.best_layer in [0, 5, 10, 15]
    assert report.split == "dev"


def test_capture_ablation_rejects_final_split():
    tasks = generate_crossover_split(split="dev", n_per_subtype=2, seed=0)
    util_fn = _utility_fn_factory(tasks)
    with pytest.raises(ValueError, match="FINAL is forbidden"):
        run_capture_ablation(
            tasks, layers=[0], policy_proba_fn=lambda t, l: 0.5,
            utility_fn=util_fn, config=ExperimentConfig(), split="final")


def test_capture_ablation_records_per_layer_metrics():
    tasks = generate_crossover_split(split="dev", n_per_subtype=3, seed=1)
    util_fn = _utility_fn_factory(tasks)
    report = run_capture_ablation(
        tasks, layers=[0, 10], policy_proba_fn=lambda t, l: 0.5,
        utility_fn=util_fn, config=ExperimentConfig())
    for r in report.per_layer:
        assert r.n_tasks == len(tasks)
        assert 0.0 <= r.abstention_rate <= 1.0


# --- Section 29-31: decisive policy comparison ---

def test_compare_policies_decisive_winner():
    tasks = generate_crossover_split(split="dev", n_per_subtype=5, seed=2)
    util_fn = _utility_fn_factory(tasks)
    # "oracle" always picks the better route; "random" picks randomly.
    from daph_learning.routing.hand_router import hand_router_route
    oracle_routes = []
    for t in tasks:
        u_sym = util_fn(t, "symbolic")
        u_llm = util_fn(t, "llm")
        oracle_routes.append("symbolic" if u_sym >= u_llm else "llm")
    hand_routes = [hand_router_route(t) for t in tasks]
    rng = np.random.default_rng(0)
    random_routes = [
        rng.choice(["symbolic", "llm"]) for _ in tasks]
    report = compare_policies(
        tasks,
        policy_routes={"oracle": oracle_routes, "hand": hand_routes,
                        "random": random_routes},
        utility_fn=util_fn, gap_threshold=0.01, n_bootstrap=500)
    assert len(report.rows) == 3
    # Oracle should have the highest mean utility.
    oracle_row = next(r for r in report.rows if r.policy_id == "oracle")
    hand_row = next(r for r in report.rows if r.policy_id == "hand")
    assert oracle_row.mean_utility >= hand_row.mean_utility


def test_compare_policies_tie_aware():
    """Section 30: when |ΔU| <= gap, verdict is 'tie'."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=3, seed=3)
    util_fn = _utility_fn_factory(tasks)
    # Two identical policies -> all ties.
    routes = ["symbolic"] * len(tasks)
    report = compare_policies(
        tasks,
        policy_routes={"A": routes, "B": routes},
        utility_fn=util_fn, gap_threshold=0.01, n_bootstrap=100)
    for p in report.pairwise:
        assert p.verdict == "tie"


def test_compare_policies_reports_ess():
    """Section 31: each row reports ESS."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=3, seed=4)
    util_fn = _utility_fn_factory(tasks)
    report = compare_policies(
        tasks, policy_routes={"p": ["symbolic"] * len(tasks)},
        utility_fn=util_fn, n_bootstrap=100)
    assert report.rows[0].ess > 0


def test_compare_policies_best_not_decisive_flagged():
    """Section 29: if best is tied with another, best_is_decisive=False."""
    tasks = generate_crossover_split(split="dev", n_per_subtype=3, seed=5)
    util_fn = _utility_fn_factory(tasks)
    routes = ["symbolic"] * len(tasks)
    report = compare_policies(
        tasks, policy_routes={"A": routes, "B": routes},
        utility_fn=util_fn, n_bootstrap=100)
    assert not report.best_is_decisive
    assert any("not decisively" in n for n in report.notes)
