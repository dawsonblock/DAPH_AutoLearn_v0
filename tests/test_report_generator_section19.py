"""Section 19 — report generator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daph_learning.evaluation.report import (
    GateDecision,
    evaluate_gates,
    generate_gate_a_results_md,
    generate_report,
)


def _stats(**overrides):
    base = {
        "primary_endpoint": {
            "estimand": "group_weighted",
            "point_estimate": 0.05,
            "ci_low": 0.01,
            "ci_high": 0.09,
        },
        "sham": {
            "p1_utility": 0.05,
            "mean_sham_utility": 0.01,
            "p1_minus_sham_mean": 0.04,
            "p1_minus_sham_ci_low": 0.01,
            "p1_percentile_vs_sham": 95.0,
        },
        "oracle_gap_capture": 0.60,
        "positive_group_fraction": 0.70,
        "worst_subtype_regression": 0.01,
        "final_access_count": 1,
        "source_hash": "abc123",
        "dataset": {
            "n_groups": 60,
            "n_tasks": 480,
            "n_crossover_subtypes": 4,
            "decisive_fraction": 0.40,
        },
        "baselines": {
            "always_llm": {"utility": 0.3},
            "always_symbolic": {"utility": 0.2},
        },
    }
    base.update(overrides)
    return base


def _criteria(**overrides):
    base = {
        "experiment_id": "test_exp",
        "criteria_hash": "hash123",
        "gates": {
            "minimum_point_gain_vs_p0": 0.02,
            "require_lcb_vs_p0_above": 0.0,
            "require_lcb_vs_sham_above": 0.0,
            "minimum_oracle_gap_capture": 0.50,
            "maximum_worst_subtype_regression": 0.03,
            "minimum_positive_group_fraction": 0.60,
            "maximum_final_access_count": 1,
        },
    }
    base.update(overrides)
    return base


def test_evaluate_gates_all_pass():
    decision = evaluate_gates(_stats(), _criteria())
    assert decision.passed
    assert len(decision.gate_verdicts) == 7


def test_evaluate_gates_lcb_fails():
    stats = _stats()
    stats["primary_endpoint"]["ci_low"] = -0.01
    decision = evaluate_gates(stats, _criteria())
    assert not decision.passed
    assert not decision.gate_verdicts["require_lcb_vs_p0_above"]["passed"]


def test_evaluate_gates_oracle_fails():
    stats = _stats()
    stats["oracle_gap_capture"] = 0.30
    decision = evaluate_gates(stats, _criteria())
    assert not decision.passed


def test_generate_md_has_verdict():
    decision = evaluate_gates(_stats(), _criteria())
    md = generate_gate_a_results_md(decision, _stats(), _criteria())
    assert "PASS" in md
    assert "gate-by-gate" in md.lower() or "Gate-by-Gate" in md


def test_generate_md_has_fail_verdict():
    stats = _stats()
    stats["primary_endpoint"]["ci_low"] = -0.01
    decision = evaluate_gates(stats, _criteria())
    md = generate_gate_a_results_md(decision, stats, _criteria())
    assert "FAIL" in md


def test_generate_report_writes_files(tmp_path):
    result = generate_report(
        tmp_path, stats=_stats(), criteria=_criteria())
    assert (tmp_path / "gate_decision.json").exists()
    assert (tmp_path / "GATE_A_RESULTS.md").exists()
    assert (tmp_path / "experiment_results.json").exists()
    decision = json.loads((tmp_path / "gate_decision.json").read_text())
    assert decision["passed"] is True


def test_no_unlabeled_cis_in_report(tmp_path):
    """Section 19: every CI must be labeled by estimator."""
    generate_report(tmp_path, stats=_stats(), criteria=_criteria())
    md = (tmp_path / "GATE_A_RESULTS.md").read_text()
    # The CI should mention the estimand.
    assert "group_weighted" in md
