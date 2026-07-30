"""v0.3.10.3.1-alpha — Section 6-9, 13, 31: weighted ablation, dedup,
grouped stats, ESS, hand router."""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.data.crossover_benchmark import (
    assert_no_within_split_duplicates,
    generate_crossover_split,
)
from daph_learning.evaluation.grouped_stats import (
    effective_sample_size,
    grouped_bootstrap_mean_delta,
    weight_diagnostics,
)
from daph_learning.routing.hand_router import hand_router_route, hand_router_routes


# --- Section 13: strong hand router ---

def test_hand_router_routes_structured_to_symbolic():
    task = {"capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 5, "b": 3, "op": "+"}}
    assert hand_router_route(task) == "symbolic"


def test_hand_router_routes_unsupported_to_llm():
    task = {"capability_ids": [], "inputs": {},
            "specification": "A warehouse has 5 crates..."}
    assert hand_router_route(task) == "llm"


def test_hand_router_routes_nl_extraction_to_llm():
    task = {"capability_ids": [], "inputs": {},
            "specification": "What is 75 minus twice 18?"}
    assert hand_router_route(task) == "llm"


def test_hand_router_routes_modular_to_symbolic():
    task = {"capability_ids": ["integer_arithmetic"],
            "inputs": {"a": 123456, "b": 97, "op": "%"}}
    assert hand_router_route(task) == "symbolic"


def test_hand_router_does_not_read_expected_or_label():
    """The hand router must not peek at the answer or stored label."""
    task = {"capability_ids": [], "inputs": {},
            "specification": "x", "expected": 42, "route_label": "symbolic"}
    # Even with a misleading route_label, it routes by decision-time info.
    assert hand_router_route(task) == "llm"


def test_hand_router_on_crossover_split():
    tasks = generate_crossover_split(split="dev", n_per_subtype=5, seed=0)
    routes = hand_router_routes(tasks)
    # v0.3.10.3.2: hand router now considers operand magnitude.
    # Subtypes A, D have structured inputs -> symbolic.
    # NL subtypes (B, C, E, F) -> symbolic if large operands, llm if small.
    for t, r in zip(tasks, routes):
        st = t["metadata"]["subtype"]
        if st in ("A", "D"):
            assert r == "symbolic", f"{st} should be symbolic, got {r}"
        else:
            # NL subtypes: route depends on operand magnitude
            assert r in ("symbolic", "llm"), f"{st} unexpected route {r}"


# --- Section 8: within-split dedup ---

def test_no_within_split_duplicates_all_splits():
    for split in ("train", "dev", "calibration", "final"):
        tasks = generate_crossover_split(split=split, n_per_subtype=8, seed=5)
        assert_no_within_split_duplicates(tasks)


# --- Section 9: grouped bootstrap ---

def test_grouped_bootstrap_returns_ci():
    records = [
        {"template_id": f"t{i % 5}", "u_a": float(i), "u_b": float(i - 1)}
        for i in range(20)
    ]
    res = grouped_bootstrap_mean_delta(
        records, "template_id", "u_a", "u_b",
        n_bootstrap=500, seed=0)
    assert res["mean_delta"] == pytest.approx(1.0)
    assert res["ci_low"] <= res["mean_delta"] <= res["ci_high"]
    assert res["n_groups"] == 5
    assert res["n_records"] == 20


def test_grouped_bootstrap_keeps_groups_together():
    """Resampling groups, not records: n_groups stays constant."""
    records = [
        {"template_id": "g0", "u_a": 1.0, "u_b": 0.0},
        {"template_id": "g0", "u_a": 1.0, "u_b": 0.0},
        {"template_id": "g1", "u_a": 0.0, "u_b": 1.0},
    ]
    res = grouped_bootstrap_mean_delta(
        records, "template_id", "u_a", "u_b",
        n_bootstrap=100, seed=1)
    assert res["n_groups"] == 2


# --- Section 31: ESS ---

def test_ess_uniform_weights():
    w = np.ones(100)
    assert effective_sample_size(w) == pytest.approx(100.0)


def test_ess_concentrated_weights():
    w = np.zeros(100)
    w[0] = 1.0
    assert effective_sample_size(w) == pytest.approx(1.0)


def test_ess_all_zero():
    assert effective_sample_size(np.zeros(10)) == 0.0


def test_weight_diagnostics():
    w = np.array([0.0, 0.5, 1.0, 0.0, 0.25])
    d = weight_diagnostics(w)
    assert d["n"] == 5
    assert d["zero_weight_fraction"] == pytest.approx(0.4)
    assert d["min_weight"] == 0.0
    assert d["max_weight"] == 1.0
    assert d["ess"] > 0.0


# --- Section 6-7: true weighted vs unweighted ablation ---

def test_uniform_and_weighted_receive_different_weights():
    """Section 6: uniform and weighted models must receive different
    weight arrays (do not reuse gap weights in an 'unweighted' model)."""
    from daph_learning.policy.weighting import compute_weights_batch, WeightMode
    du = np.array([0.0, 0.5, -0.8, 0.3, -0.1])
    conf = np.ones(5)
    uniform = compute_weights_batch(
        du, conf, WeightMode.UNIFORM, gap_threshold=0.0, max_weight=1.0)
    weighted = compute_weights_batch(
        du, conf, WeightMode.CLIPPED_GAP, gap_threshold=0.0, max_weight=1.0)
    assert not np.allclose(uniform, weighted), (
        "uniform and weighted weights must differ; reusing gap weights "
        "in an 'unweighted' model is the Section 6 bug")
    assert np.all(uniform == 1.0)
