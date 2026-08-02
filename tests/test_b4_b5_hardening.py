"""Tests for B4 hardening + B5 adaptive compute modules.

Covers:
* artifact JSON validity
* manifest hashing
* corrupted artifact rejection
* source-hash validation
* task split leakage
* retrieval leakage
* train-only PCA
* frozen config enforcement
* group-local positive-group calculation
* paired group bootstrap
* matched sham generation
* per-task action alignment
* representation task-ID ordering
* deterministic offline reevaluation
* report-vs-JSON consistency
* malformed hidden-state matrix rejection
* nonfinite hidden-state rejection
* action-space consistency
* utility normalization
* cost accounting
* winner-distribution diagnostics
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from daph_learning.executive.artifact_integrity import (
    validate_json_artifact,
    validate_hash_reference,
    validate_required_tree,
    detect_corruption,
    is_placeholder,
    is_zero_hash,
    sha256_file,
    sha256_json,
    B4_REQUIRED_ARTIFACTS,
    B5_REQUIRED_ARTIFACTS,
)
from daph_learning.executive.manifest import (
    ManifestBuilder,
    compute_config_hash,
    write_config_hash,
    capture_environment,
    load_manifest,
)
from daph_learning.executive.leakage import (
    check_task_id_overlap,
    check_exact_prompt_leakage,
    check_retrieval_store_leakage,
    check_pca_train_only,
    check_policy_leakage,
    check_group_leakage,
    check_representation_sanity,
)
from daph_learning.executive.stats import (
    compute_group_local_results,
    positive_group_fraction,
    worst_group_delta,
    paired_group_bootstrap,
    create_matched_sham_utilities,
    gap_capture,
    selection_accuracy,
)
from daph_learning.executive.lifecycle import (
    ExperimentStatus,
    FrozenConfig,
    ExperimentState,
    register_experiment,
    invalidate_experiment,
    load_registry,
)
from daph_learning.executive.b5_actions import (
    b5_action_space,
    B5_ACTION_IDS,
    B5_ACTION_DIRECT_FAST,
    B5_ACTION_DIRECT_THINK,
    B5_ACTION_RETRIEVE,
    B5_ACTION_DECOMPOSE,
    InferencePreset,
    B5_DEFAULT_PRESETS,
)
from daph_learning.executive.b5_dataset import (
    generate_b5_dataset,
    build_b5_retrieval_store,
    compute_winner_distribution,
    B5_FAMILIES,
)
from daph_learning.executive.b5_policies import (
    LinearQPolicy,
    RidgeQPolicy,
    MLPQPolicy,
    compute_surface_features,
)
from daph_learning.executive.b5_qualification import (
    evaluate_gates,
    GateThresholds,
    GateResult,
)


# ──────────────────────────────────────────────────────────────────────
# Artifact integrity tests
# ──────────────────────────────────────────────────────────────────────

class TestArtifactIntegrity:
    """Tests for artifact JSON validity and corruption rejection."""

    def test_valid_json_artifact(self, tmp_path):
        p = tmp_path / "valid.json"
        p.write_text(json.dumps({"key": "value", "num": 42}))
        check = validate_json_artifact(p, required_fields=["key"])
        assert check.passed

    def test_missing_file(self, tmp_path):
        check = validate_json_artifact(tmp_path / "nonexistent.json")
        assert not check.passed
        assert "does not exist" in check.detail

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("")
        check = validate_json_artifact(p)
        assert not check.passed
        assert "empty" in check.detail

    def test_malformed_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}")
        check = validate_json_artifact(p)
        assert not check.passed
        assert "invalid JSON" in check.detail

    def test_ssh_corruption_rejected(self, tmp_path):
        """The exact corruption that invalidated B4 must be rejected."""
        p = tmp_path / "corrupted.json"
        p.write_text("Error: Your SSH client doesn't support PTY")
        check = validate_json_artifact(p)
        assert not check.passed
        assert "corruption" in check.detail.lower()

    def test_placeholder_rejected(self, tmp_path):
        p = tmp_path / "placeholder.json"
        p.write_text(json.dumps({"model_id": "unknown"}))
        check = validate_json_artifact(p, required_fields=["model_id"])
        assert not check.passed
        assert "placeholder" in check.detail

    def test_missing_required_field(self, tmp_path):
        p = tmp_path / "missing.json"
        p.write_text(json.dumps({"a": 1}))
        check = validate_json_artifact(p, required_fields=["a", "b"])
        assert not check.passed
        assert "missing required field" in check.detail

    def test_hash_validation(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"test": true}')
        h = sha256_file(p)
        check = validate_hash_reference(p, h)
        assert check.passed

    def test_hash_mismatch(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"test": true}')
        check = validate_hash_reference(p, "0" * 64)
        assert not check.passed

    def test_zero_hash_rejected(self):
        assert is_zero_hash("0" * 64)
        assert is_zero_hash("")
        assert not is_zero_hash("a" * 64)

    def test_detect_corruption_patterns(self):
        assert detect_corruption("Error: Your SSH client doesn't support PTY") is not None
        assert detect_corruption("Traceback (most recent call last)") is not None
        assert detect_corruption('{"valid": "json"}') is None

    def test_is_placeholder(self):
        assert is_placeholder("unknown")
        assert is_placeholder("TBD")
        assert is_placeholder("")
        assert not is_placeholder("Qwen/Qwen3-8B")
        assert not is_placeholder(42)

    def test_required_tree_missing_artifacts(self, tmp_path):
        report = validate_required_tree(tmp_path, B4_REQUIRED_ARTIFACTS, experiment_id="test")
        assert not report.passed
        assert len(report.failures) > 0


# ──────────────────────────────────────────────────────────────────────
# Manifest tests
# ──────────────────────────────────────────────────────────────────────

class TestManifest:
    """Tests for manifest hashing and validation."""

    def test_config_hash_deterministic(self):
        c1 = {"a": 1, "b": 2}
        c2 = {"b": 2, "a": 1}
        assert compute_config_hash(c1) == compute_config_hash(c2)

    def test_config_hash_differs(self):
        assert compute_config_hash({"a": 1}) != compute_config_hash({"a": 2})

    def test_manifest_builder(self, tmp_path):
        # Create a test artifact
        (tmp_path / "data").mkdir()
        p = tmp_path / "data" / "test.json"
        p.write_text(json.dumps({"value": 42}))

        builder = ManifestBuilder(
            experiment_id="test",
            experiment_family="test_family",
            artifact_root=tmp_path,
        )
        builder.set_config_hash("abc123")
        builder.add_file("data/test.json", schema_version="1.0", section="dataset")
        builder.write()

        manifest = load_manifest(tmp_path / "manifest.json")
        assert manifest["schema_version"] == "1.0"
        assert manifest["experiment_id"] == "test"
        assert manifest["config_hash"] == "abc123"
        assert "test" in manifest["dataset"]

    def test_manifest_rejects_corrupted_file(self, tmp_path):
        (tmp_path / "data").mkdir()
        p = tmp_path / "data" / "corrupt.json"
        p.write_text("Error: Your SSH client doesn't support PTY")

        builder = ManifestBuilder(
            experiment_id="test",
            experiment_family="test",
            artifact_root=tmp_path,
        )
        with pytest.raises(ValueError, match="corruption"):
            builder.add_file("data/corrupt.json")

    def test_capture_environment(self):
        env = capture_environment()
        assert "python_version" in env
        assert "platform" in env


# ──────────────────────────────────────────────────────────────────────
# Leakage check tests
# ──────────────────────────────────────────────────────────────────────

class TestLeakageChecks:
    """Tests for leakage detection."""

    def test_task_id_no_overlap(self):
        check = check_task_id_overlap(
            ["t1", "t2"], ["t3", "t4"], ["t5", "t6"]
        )
        assert check.passed

    def test_task_id_overlap_detected(self):
        check = check_task_id_overlap(
            ["t1", "t2"], ["t2", "t4"], ["t5", "t6"]
        )
        assert not check.passed
        assert "overlap" in check.detail

    def test_exact_prompt_no_overlap(self):
        check = check_exact_prompt_leakage(
            ["What is 1+1?"], ["What is 2+2?"], ["What is 3+3?"]
        )
        assert check.passed

    def test_exact_prompt_overlap_detected(self):
        check = check_exact_prompt_leakage(
            ["What is 1+1?"], ["What is 1+1?"], ["What is 3+3?"]
        )
        assert not check.passed

    def test_retrieval_store_train_only(self):
        check = check_retrieval_store_leakage(
            ["t1", "t2"], ["t1", "t2"], ["t3"], ["t4"]
        )
        assert check.passed

    def test_retrieval_store_contamination(self):
        check = check_retrieval_store_leakage(
            ["t1", "t3"], ["t1"], ["t2"], ["t3", "t4"]
        )
        assert not check.passed
        assert "contaminated" in check.detail

    def test_pca_train_only_valid(self):
        check = check_pca_train_only({
            "fit_split": "train",
            "training_representation_hash": "a" * 64,
        })
        assert check.passed

    def test_pca_train_only_violation(self):
        check = check_pca_train_only({
            "fit_split": "final",
            "training_representation_hash": "a" * 64,
        })
        assert not check.passed

    def test_pca_zero_hash_rejected(self):
        check = check_pca_train_only({
            "fit_split": "train",
            "training_representation_hash": "0" * 64,
        })
        assert not check.passed

    def test_policy_no_final_leakage(self):
        check = check_policy_leakage({
            "selection_split": "dev",
            "used_final_data": False,
        })
        assert check.passed

    def test_policy_final_leakage(self):
        check = check_policy_leakage({
            "selection_split": "final",
            "used_final_data": True,
        })
        assert not check.passed

    def test_group_no_cross_split(self):
        check = check_group_leakage(["g1", "g2"], ["g3"], ["g4"])
        assert check.passed

    def test_group_cross_split_detected(self):
        check = check_group_leakage(["g1", "g2"], ["g1"], ["g3"])
        assert not check.passed

    def test_representation_sanity_valid(self):
        features = np.random.randn(10, 32).astype(np.float32)
        check = check_representation_sanity(features, [f"t{i}" for i in range(10)])
        assert check.passed

    def test_representation_sanity_nan(self):
        features = np.random.randn(10, 32).astype(np.float32)
        features[0, 0] = np.nan
        check = check_representation_sanity(features, [f"t{i}" for i in range(10)])
        assert not check.passed
        assert "NaN" in check.detail

    def test_representation_sanity_inf(self):
        features = np.random.randn(10, 32).astype(np.float32)
        features[0, 0] = np.inf
        check = check_representation_sanity(features, [f"t{i}" for i in range(10)])
        assert not check.passed
        assert "infinite" in check.detail

    def test_representation_sanity_wrong_dim(self):
        features = np.random.randn(10, 32).astype(np.float32)
        check = check_representation_sanity(features, [f"t{i}" for i in range(10)], expected_dim=64)
        assert not check.passed

    def test_representation_sanity_task_id_mismatch(self):
        features = np.random.randn(10, 32).astype(np.float32)
        check = check_representation_sanity(features, [f"t{i}" for i in range(5)])
        assert not check.passed
        assert "task_ids" in check.detail


# ──────────────────────────────────────────────────────────────────────
# Corrected statistics tests
# ──────────────────────────────────────────────────────────────────────

class TestCorrectedStats:
    """Tests for group-local positive-group and paired bootstrap."""

    def test_group_local_paired(self):
        """Group-local delta must be paired, not global baseline."""
        results = compute_group_local_results(
            task_ids=["t1", "t2", "t3", "t4"],
            group_ids=["g1", "g1", "g2", "g2"],
            subtypes=["a", "a", "b", "b"],
            hidden_utilities=[0.8, 0.6, 0.9, 0.7],
            baseline_utilities=[0.5, 0.5, 0.5, 0.5],
        )
        assert len(results) == 2
        # g1: mean(0.8-0.5, 0.6-0.5) = mean(0.3, 0.1) = 0.2
        assert results[0].group_id == "g1"
        assert abs(results[0].paired_delta - 0.2) < 1e-6
        assert results[0].delta_positive
        # g2: mean(0.9-0.5, 0.7-0.5) = mean(0.4, 0.2) = 0.3
        assert results[1].group_id == "g2"
        assert abs(results[1].paired_delta - 0.3) < 1e-6

    def test_positive_group_fraction(self):
        results = compute_group_local_results(
            task_ids=["t1", "t2", "t3", "t4"],
            group_ids=["g1", "g1", "g2", "g2"],
            subtypes=["a", "a", "b", "b"],
            hidden_utilities=[0.8, 0.6, 0.3, 0.2],
            baseline_utilities=[0.5, 0.5, 0.5, 0.5],
        )
        # g1 positive, g2 negative
        assert positive_group_fraction(results) == 0.5

    def test_worst_group_delta(self):
        results = compute_group_local_results(
            task_ids=["t1", "t2", "t3", "t4"],
            group_ids=["g1", "g1", "g2", "g2"],
            subtypes=["a", "a", "b", "b"],
            hidden_utilities=[0.8, 0.6, 0.3, 0.2],
            baseline_utilities=[0.5, 0.5, 0.5, 0.5],
        )
        assert abs(worst_group_delta(results) - (-0.25)) < 1e-6

    def test_paired_group_bootstrap(self):
        """Bootstrap must sample groups, not individual tasks."""
        rng = np.random.RandomState(42)
        n = 100
        utils_a = rng.randn(n) + 0.1
        utils_b = rng.randn(n)
        groups = [f"g{i // 10}" for i in range(n)]  # 10 groups of 10

        result = paired_group_bootstrap(
            utils_a, utils_b, groups,
            n_replicates=1000, seed=42,
        )
        assert result.n_replicates == 1000
        assert -0.5 < result.point_estimate < 0.5
        assert result.lcb_95 <= result.point_estimate <= result.ucb_95
        assert 0.0 <= result.prob_positive <= 1.0

    def test_paired_bootstrap_empty(self):
        result = paired_group_bootstrap(
            np.array([]), np.array([]), [],
            n_replicates=100, seed=42,
        )
        assert result.n_replicates == 0
        assert result.point_estimate == 0.0

    def test_matched_sham_utilities(self):
        """Sham must destroy the feature→utility link."""
        n = 50
        n_actions = 4
        utilities = np.random.randn(n, n_actions)
        subtypes = ["a"] * 25 + ["b"] * 25
        split_ids = np.zeros(n, dtype=int)

        sham = create_matched_sham_utilities(
            utilities, subtypes, split_ids, seed=42
        )
        # Shape preserved
        assert sham.shape == utilities.shape
        # Within each bin, the set of utility rows is the same (just permuted)
        for bin_key in [("a", 0), ("b", 0)]:
            indices = [i for i in range(n) if (subtypes[i], int(split_ids[i])) == bin_key]
            original_rows = set(tuple(utilities[i]) for i in indices)
            sham_rows = set(tuple(sham[i]) for i in indices)
            assert original_rows == sham_rows

    def test_gap_capture(self):
        assert abs(gap_capture(0.8, 0.5, 1.0) - 0.6) < 1e-6
        assert gap_capture(0.5, 0.5, 1.0) == 0.0
        assert gap_capture(0.5, 0.5, 0.5) == 0.0  # no gap

    def test_selection_accuracy(self):
        preds = np.array([0, 1, 2, 1])
        oracle = np.array([0, 1, 1, 1])
        assert selection_accuracy(preds, oracle) == 0.75


# ──────────────────────────────────────────────────────────────────────
# Lifecycle tests
# ──────────────────────────────────────────────────────────────────────

class TestLifecycle:
    """Tests for frozen experiment lifecycle."""

    def test_status_transitions(self):
        state = ExperimentState(experiment_id="test")
        assert state.status == ExperimentStatus.DEVELOPMENT

        state.freeze({"experiment_id": "test", "action_space": {}})
        assert state.status == ExperimentStatus.FROZEN

        state.start_final({"experiment_id": "test", "action_space": {}})
        assert state.status == ExperimentStatus.RUNNING

        state.mark_qualified()
        assert state.status == ExperimentStatus.QUALIFIED

    def test_invalid_transition(self):
        state = ExperimentState(experiment_id="test")
        with pytest.raises(ValueError, match="invalid status transition"):
            state.transition_to(ExperimentStatus.QUALIFIED)

    def test_frozen_config_mismatch(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test", "action_space": {"a": 1}})
        with pytest.raises(ValueError, match="does not match"):
            state.start_final({"experiment_id": "test", "action_space": {"a": 2}})

    def test_frozen_config_match(self):
        config = {"experiment_id": "test", "action_space": {"a": 1}, "extra": "ignored"}
        state = ExperimentState(experiment_id="test")
        state.freeze(config)
        # Same frozen fields, different non-frozen field → should match
        state.start_final({"experiment_id": "test", "action_space": {"a": 1}, "extra": "different"})

    def test_invalidate(self):
        state = ExperimentState(experiment_id="test")
        state.freeze({"experiment_id": "test"})
        state.invalidate("corrupted artifacts")
        assert state.status == ExperimentStatus.INVALIDATED

    def test_registry(self, tmp_path):
        reg_path = tmp_path / "registry.json"
        register_experiment(reg_path, type("E", (), {
            "experiment_id": "exp1", "version": "1.0", "status": "QUALIFIED",
            "artifact_root": "artifacts/exp1", "qualification_summary": {},
            "invalidated_reason": None, "to_dict": lambda self: {
                "experiment_id": "exp1", "version": "1.0", "status": "QUALIFIED",
                "artifact_root": "artifacts/exp1", "qualification_summary": {},
                "invalidated_reason": None,
            }
        })())
        reg = load_registry(reg_path)
        assert len(reg["experiments"]) == 1

        invalidate_experiment(reg_path, "exp1", "corrupted")
        reg = load_registry(reg_path)
        assert reg["experiments"][0]["status"] == "INVALIDATED"


# ──────────────────────────────────────────────────────────────────────
# B5 action space tests
# ──────────────────────────────────────────────────────────────────────

class TestB5ActionSpace:
    """Tests for B5 action space consistency."""

    def test_four_actions(self):
        space = b5_action_space()
        assert space.n_actions == 4

    def test_action_ids(self):
        assert set(B5_ACTION_IDS) == {
            B5_ACTION_DIRECT_FAST,
            B5_ACTION_DIRECT_THINK,
            B5_ACTION_RETRIEVE,
            B5_ACTION_DECOMPOSE,
        }

    def test_fast_vs_think_compute_difference(self):
        """FAST and THINK must differ in inference compute."""
        fast = B5_DEFAULT_PRESETS[B5_ACTION_DIRECT_FAST]
        think = B5_DEFAULT_PRESETS[B5_ACTION_DIRECT_THINK]
        assert fast.reasoning_mode == "off"
        assert think.reasoning_mode == "on"
        assert fast.max_tokens < think.max_tokens
        assert fast.no_think_prefix is True
        assert think.no_think_prefix is False

    def test_cost_estimates_ordered(self):
        """Cost estimates should reflect compute intensity."""
        space = b5_action_space()
        costs = {a.action_id: a.cost_estimate for a in space.actions}
        assert costs[B5_ACTION_DIRECT_FAST] < costs[B5_ACTION_DIRECT_THINK]
        assert costs[B5_ACTION_DIRECT_THINK] < costs[B5_ACTION_DECOMPOSE]


# ──────────────────────────────────────────────────────────────────────
# B5 dataset tests
# ──────────────────────────────────────────────────────────────────────

class TestB5Dataset:
    """Tests for B5 dataset generation and crossovers."""

    def test_dataset_generation(self):
        ds = generate_b5_dataset(n_train=60, n_dev=30, n_final=40, n_ood=20, seed=42)
        assert ds["train"].n_tasks == 60
        assert ds["dev"].n_tasks == 30
        assert ds["final"].n_tasks == 40
        assert ds["final_ood"].n_tasks == 20

    def test_no_task_id_overlap(self):
        ds = generate_b5_dataset(n_train=60, n_dev=30, n_final=40, n_ood=20, seed=42)
        train_ids = {t["task_id"] for t in ds["train"].tasks}
        dev_ids = {t["task_id"] for t in ds["dev"].tasks}
        final_ids = {t["task_id"] for t in ds["final"].tasks}
        assert len(train_ids & dev_ids) == 0
        assert len(train_ids & final_ids) == 0
        assert len(dev_ids & final_ids) == 0

    def test_no_group_overlap(self):
        ds = generate_b5_dataset(n_train=60, n_dev=30, n_final=40, n_ood=20, seed=42)
        train_groups = {t["group_id"] for t in ds["train"].tasks}
        dev_groups = {t["group_id"] for t in ds["dev"].tasks}
        final_groups = {t["group_id"] for t in ds["final"].tasks}
        assert len(train_groups & dev_groups) == 0
        assert len(train_groups & final_groups) == 0

    def test_all_families_present(self):
        ds = generate_b5_dataset(n_train=120, n_dev=60, n_final=80, n_ood=40, seed=42)
        train_families = {t["family"] for t in ds["train"].tasks}
        for fam in B5_FAMILIES:
            assert fam in train_families, f"family {fam} not in train"

    def test_within_family_difficulty_tiers(self):
        ds = generate_b5_dataset(n_train=360, n_dev=120, n_final=240, n_ood=60, seed=42)
        # Each family should have tasks at multiple difficulty tiers
        for fam in B5_FAMILIES[:3]:
            difficulties = {t["difficulty"] for t in ds["train"].tasks if t["family"] == fam}
            assert len(difficulties) >= 2, f"family {fam} only has {difficulties}"

    def test_winner_distribution(self):
        tasks = [{"task_id": f"t{i}", "family": "fam1"} for i in range(10)]
        utilities = {}
        for i, t in enumerate(tasks):
            # Alternate winners
            utilities[t["task_id"]] = {
                B5_ACTION_DIRECT_FAST: 0.9 if i % 2 == 0 else 0.3,
                B5_ACTION_DIRECT_THINK: 0.3 if i % 2 == 0 else 0.9,
                B5_ACTION_RETRIEVE: 0.5,
                B5_ACTION_DECOMPOSE: 0.4,
            }
        dist = compute_winner_distribution(tasks, utilities, B5_ACTION_IDS)
        assert dist["n_families"] == 1
        # Two actions each win 50% → crossover passes
        assert dist["per_family"][0]["crossover_passes"]

    def test_retrieval_store_train_only(self):
        ds = generate_b5_dataset(n_train=60, n_dev=30, n_final=40, n_ood=20, seed=42)
        store = build_b5_retrieval_store(ds["train"].tasks)
        # All store entries should come from train tasks
        train_ids = {t["task_id"] for t in ds["train"].tasks}
        for entry in store:
            assert entry["task_id"] in train_ids


# ──────────────────────────────────────────────────────────────────────
# B5 policy tests
# ──────────────────────────────────────────────────────────────────────

class TestB5Policies:
    """Tests for B5 policy models."""

    def test_linear_q_policy(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 32).astype(np.float32)
        U = rng.randn(100, 4).astype(np.float32)

        policy = LinearQPolicy(action_ids=list(B5_ACTION_IDS), n_iter=100, l2=0.01)
        policy.fit(X, U)
        preds = policy.predict(X)
        assert preds.shape == (100,)
        utils = policy.predict_utilities(X)
        assert utils.shape == (100, 4)

    def test_ridge_q_policy(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 32).astype(np.float32)
        U = rng.randn(100, 4).astype(np.float32)

        policy = RidgeQPolicy(action_ids=list(B5_ACTION_IDS), alpha=1.0)
        policy.fit(X, U)
        preds = policy.predict(X)
        assert preds.shape == (100,)

    def test_ridge_dev_tuning(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 32).astype(np.float32)
        U_train = rng.randn(100, 4).astype(np.float32)
        X_dev = rng.randn(30, 32).astype(np.float32)
        U_dev = rng.randn(30, 4).astype(np.float32)

        policy = RidgeQPolicy(action_ids=list(B5_ACTION_IDS))
        policy.fit_with_dev_tuning(X_train, U_train, X_dev, U_dev,
                                   alphas=[0.1, 1.0, 10.0])
        assert policy.alpha in [0.1, 1.0, 10.0]

    def test_mlp_q_policy(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 32).astype(np.float32)
        U = rng.randn(50, 4).astype(np.float32)

        policy = MLPQPolicy(
            action_ids=list(B5_ACTION_IDS),
            hidden1=32, hidden2=16, n_iter=100, seed=42,
        )
        policy.fit(X, U)
        preds = policy.predict(X)
        assert preds.shape == (50,)

    def test_policy_save_load(self, tmp_path):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 16).astype(np.float32)
        U = rng.randn(50, 4).astype(np.float32)

        policy = LinearQPolicy(action_ids=list(B5_ACTION_IDS), n_iter=50)
        policy.fit(X, U)
        path = str(tmp_path / "policy.json")
        policy.save(path)
        loaded = LinearQPolicy.load(path)
        assert loaded.action_ids == policy.action_ids
        assert np.allclose(loaded.predict_utilities(X), policy.predict_utilities(X))

    def test_predict_margins(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 16).astype(np.float32)
        U = rng.randn(20, 4).astype(np.float32)

        policy = LinearQPolicy(action_ids=list(B5_ACTION_IDS), n_iter=50)
        policy.fit(X, U)
        margins = policy.predict_margins(X)
        assert margins.shape == (20,)
        assert np.all(margins >= 0)  # top - second is always ≥ 0

    def test_surface_features(self):
        tasks = [
            {"prompt": "What is 1+1?", "subtype": "arithmetic_easy"},
            {"prompt": "What is 2+2?", "subtype": "arithmetic_easy"},
            {"prompt": "Compute complex formula", "subtype": "multi_step"},
        ]
        features = compute_surface_features(tasks, feature_types=("subtype", "prompt_length"))
        assert features.shape[0] == 3
        assert features.shape[1] >= 2  # at least 2 subtypes + 1 length


# ──────────────────────────────────────────────────────────────────────
# B5 qualification gate tests
# ──────────────────────────────────────────────────────────────────────

class TestQualificationGates:
    """Tests for B5 qualification gates."""

    def test_all_gates_pass(self):
        from daph_learning.executive.stats import BootstrapResult, ShamComparisonResult, GroupResult

        boot_pass = BootstrapResult(
            comparison="test", n_replicates=1000,
            mean_delta=0.1, median_delta=0.1,
            lcb_95=0.05, ucb_95=0.15,
            std_error=0.02, prob_positive=0.95,
            point_estimate=0.1,
        )
        sham_pass = ShamComparisonResult(
            n_shams=50, real_hidden_regret=0.1,
            sham_mean_regret=0.3, sham_median_regret=0.3,
            sham_best_regret=0.2, sham_worst_regret=0.4,
            hidden_vs_sham_paired_delta=0.2,
            hidden_vs_sham_lcb95=0.05,
            hidden_vs_sham_ucb95=0.35,
            prob_hidden_gt_sham=0.95,
        )
        groups = [GroupResult("g1", "a", 5, 0.8, 0.5, 0.3, True)]

        result = evaluate_gates(
            experiment_id="test",
            boot_hidden_vs_bestfixed=boot_pass,
            boot_hidden_vs_surface=boot_pass,
            sham_result=sham_pass,
            group_results=groups,
            hidden_utility=0.8,
            best_fixed_utility=0.5,
            oracle_utility=1.0,
            selection_accuracy=0.8,
            leakage_passed=True,
            integrity_passed=True,
        )
        assert result.overall_passed
        assert len(result.gates) == 7
        assert len(result.failed_gates) == 0

    def test_gate_failure(self):
        from daph_learning.executive.stats import BootstrapResult, ShamComparisonResult, GroupResult

        boot_fail = BootstrapResult(
            comparison="test", n_replicates=1000,
            mean_delta=-0.1, median_delta=-0.1,
            lcb_95=-0.15, ucb_95=-0.05,
            std_error=0.02, prob_positive=0.05,
            point_estimate=-0.1,
        )
        sham_fail = ShamComparisonResult(
            n_shams=50, real_hidden_regret=0.3,
            sham_mean_regret=0.1, sham_median_regret=0.1,
            sham_best_regret=0.05, sham_worst_regret=0.2,
            hidden_vs_sham_paired_delta=-0.2,
            hidden_vs_sham_lcb95=-0.35,
            hidden_vs_sham_ucb95=-0.05,
            prob_hidden_gt_sham=0.05,
        )
        groups = [GroupResult("g1", "a", 5, 0.3, 0.5, -0.2, False)]

        result = evaluate_gates(
            experiment_id="test",
            boot_hidden_vs_bestfixed=boot_fail,
            boot_hidden_vs_surface=boot_fail,
            sham_result=sham_fail,
            group_results=groups,
            hidden_utility=0.3,
            best_fixed_utility=0.5,
            oracle_utility=1.0,
            selection_accuracy=0.3,
            leakage_passed=False,
            integrity_passed=False,
        )
        assert not result.overall_passed
        assert len(result.failed_gates) == 7


# ──────────────────────────────────────────────────────────────────────
# Synthetic integration test
# ──────────────────────────────────────────────────────────────────────

class TestSyntheticIntegration:
    """End-to-end synthetic pipeline test."""

    def test_full_pipeline_runs(self, tmp_path):
        from daph_learning.executive.synthetic_pipeline import run_synthetic_experiment
        result = run_synthetic_experiment(
            str(tmp_path / "synthetic_exp"),
            n_train=120, n_dev=60, n_final=80, n_ood=40,
            n_shams=5, bootstrap_replicates=200,
        )
        assert "experiment_id" in result
        assert "qualified" in result
        assert "n_gates" in result
        # The pipeline must produce all required artifacts
        root = tmp_path / "synthetic_exp"
        assert (root / "manifest.json").exists()
        assert (root / "qualification" / "qualification.json").exists()
        assert (root / "status.json").exists()

    def test_reproduction_command(self, tmp_path):
        from daph_learning.executive.synthetic_pipeline import run_synthetic_experiment
        from daph_learning.executive.reproduce import reproduce

        root = tmp_path / "repro_exp"
        run_synthetic_experiment(
            str(root),
            n_train=120, n_dev=60, n_final=80, n_ood=40,
            n_shams=5, bootstrap_replicates=200,
        )
        result = reproduce(str(root), experiment_family="b5")
        assert "passed" in result
        assert "steps" in result
        # The reproduction should verify the artifact tree
        steps = {s["step"]: s for s in result["steps"]}
        assert "artifact_tree" in steps

    def test_qualification_json_is_valid(self, tmp_path):
        from daph_learning.executive.synthetic_pipeline import run_synthetic_experiment
        root = tmp_path / "qual_test"
        run_synthetic_experiment(
            str(root),
            n_train=120, n_dev=60, n_final=80, n_ood=40,
            n_shams=5, bootstrap_replicates=200,
        )
        with open(root / "qualification" / "qualification.json") as f:
            qual = json.load(f)
        assert "experiment_id" in qual
        assert "gates" in qual
        assert "overall_passed" in qual
        # No corruption
        assert detect_corruption(json.dumps(qual)) is None

    def test_report_vs_json_consistency(self, tmp_path):
        """Report headline metrics must match JSON values."""
        from daph_learning.executive.synthetic_pipeline import run_synthetic_experiment
        root = tmp_path / "report_test"
        run_synthetic_experiment(
            str(root),
            n_train=120, n_dev=60, n_final=80, n_ood=40,
            n_shams=5, bootstrap_replicates=200,
        )
        with open(root / "qualification" / "qualification.json") as f:
            qual = json.load(f)
        with open(root / "qualification" / "report.md") as f:
            report = f.read()

        # The report must contain the hidden utility value
        hidden_util = qual["primary_metrics"]["hidden_utility"]
        # Check that the value appears in the report (within formatting tolerance)
        assert f"{hidden_util:.4f}" in report or f"{hidden_util:.1%}" in report or \
               f"{hidden_util:.3f}" in report
