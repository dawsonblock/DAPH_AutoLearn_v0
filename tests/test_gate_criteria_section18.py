"""Section 18 — frozen gate criteria config tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from daph_learning.evaluation.gate_criteria import (
    GateCriteria, load_gate_criteria,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_real_002_config():
    c = load_gate_criteria(REPO_ROOT / "configs" / "gate_a_real_002.yaml")
    assert c.experiment_id == "daph_gate_a_real_002"
    assert c.primary_endpoint.estimand == "group_weighted"
    assert c.primary_endpoint.confidence_level == 0.95
    assert c.primary_endpoint.bootstrap_iterations == 20000
    assert c.gates.require_lcb_vs_p0_above == 0.0
    assert c.gates.maximum_final_access_count == 1
    assert c.dataset.minimum_groups == 60
    assert c.evidence.require_real_model is True
    assert c.evidence.allow_synthetic is False


def test_load_smoke_config():
    c = load_gate_criteria(REPO_ROOT / "configs" / "gate_a_smoke.yaml")
    assert c.experiment_id == "daph_gate_a_smoke"
    assert c.dataset.minimum_groups == 8


def test_criteria_hash_is_deterministic():
    c = load_gate_criteria(REPO_ROOT / "configs" / "gate_a_real_002.yaml")
    assert c.criteria_hash == c.criteria_hash
    assert len(c.criteria_hash) == 64


def test_real_and_smoke_hashes_differ():
    r = load_gate_criteria(REPO_ROOT / "configs" / "gate_a_real_002.yaml")
    s = load_gate_criteria(REPO_ROOT / "configs" / "gate_a_smoke.yaml")
    assert r.criteria_hash != s.criteria_hash


def test_reject_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "experiment_id: x\nprimary_endpoint: {name: u, estimand: group_weighted, "
        "confidence_level: 0.95, bootstrap_iterations: 100}\n"
        "gates: {minimum_point_gain_vs_p0: 0.0, require_lcb_vs_p0_above: 0.0, "
        "require_lcb_vs_sham_above: 0.0, minimum_oracle_gap_capture: 0.0, "
        "maximum_worst_subtype_regression: 1.0, minimum_positive_group_fraction: 0.0, "
        "maximum_final_access_count: 1}\n"
        "dataset: {minimum_groups: 1, minimum_crossover_subtypes: 1, "
        "minimum_backend_win_fraction: 0.1, minimum_decisive_fraction: 0.1}\n"
        "evidence: {require_real_model: true, allow_synthetic: false, "
        "require_model_revision: true, require_source_hash_match: true, "
        "require_dataset_hashes: true, require_final_access_ledger: true}\n"
        "bogus_key: 123\n")
    with pytest.raises(ValueError, match="unknown"):
        load_gate_criteria(p)


def test_reject_missing_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("experiment_id: x\n")  # missing most keys
    with pytest.raises(ValueError, match="missing"):
        load_gate_criteria(p)


def test_reject_bad_estimand(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "experiment_id: x\nprimary_endpoint: {name: u, estimand: bogus, "
        "confidence_level: 0.95, bootstrap_iterations: 100}\n"
        "gates: {minimum_point_gain_vs_p0: 0.0, require_lcb_vs_p0_above: 0.0, "
        "require_lcb_vs_sham_above: 0.0, minimum_oracle_gap_capture: 0.0, "
        "maximum_worst_subtype_regression: 1.0, minimum_positive_group_fraction: 0.0, "
        "maximum_final_access_count: 1}\n"
        "dataset: {minimum_groups: 1, minimum_crossover_subtypes: 1, "
        "minimum_backend_win_fraction: 0.1, minimum_decisive_fraction: 0.1}\n"
        "evidence: {require_real_model: true, allow_synthetic: false, "
        "require_model_revision: true, require_source_hash_match: true, "
        "require_dataset_hashes: true, require_final_access_ledger: true}\n")
    with pytest.raises(ValueError, match="estimand"):
        load_gate_criteria(p)


def test_reject_contradictory_evidence(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "experiment_id: x\nprimary_endpoint: {name: u, estimand: group_weighted, "
        "confidence_level: 0.95, bootstrap_iterations: 100}\n"
        "gates: {minimum_point_gain_vs_p0: 0.0, require_lcb_vs_p0_above: 0.0, "
        "require_lcb_vs_sham_above: 0.0, minimum_oracle_gap_capture: 0.0, "
        "maximum_worst_subtype_regression: 1.0, minimum_positive_group_fraction: 0.0, "
        "maximum_final_access_count: 1}\n"
        "dataset: {minimum_groups: 1, minimum_crossover_subtypes: 1, "
        "minimum_backend_win_fraction: 0.1, minimum_decisive_fraction: 0.1}\n"
        "evidence: {require_real_model: true, allow_synthetic: true, "
        "require_model_revision: true, require_source_hash_match: true, "
        "require_dataset_hashes: true, require_final_access_ledger: true}\n")
    with pytest.raises(ValueError, match="contradictory"):
        load_gate_criteria(p)


def test_gate_criteria_is_frozen():
    c = load_gate_criteria(REPO_ROOT / "configs" / "gate_a_real_002.yaml")
    with pytest.raises((AttributeError, TypeError)):
        c.experiment_id = "hacked"  # type: ignore[misc]
