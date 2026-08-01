"""DAPH v0.4 — End-to-end integration test for 3-action executive qualification.

This test simulates the full executive qualification pipeline for a 3-action
experiment (direct reasoning vs retrieval vs decomposition):

1. Generate synthetic counterfactual sets with 3 actions
2. Compute per-action utilities using UtilityModel
3. Train an ExecutiveLogisticPolicy on the utilities
4. Generate ActionDecisions for held-out data
5. Build ExecutiveTaskRecords
6. Run evaluate_qualification() to get the full result
7. Generate a markdown report
8. Verify the report contains all expected sections

This is the integration test that proves the generic executive module
can run a complete experiment end-to-end.
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    ExecutiveState,
    ActionExecution,
    CounterfactualSet,
    UtilityModel,
    ExecutiveTaskRecord,
    ExecutiveCentroidPolicy,
    ExecutiveLogisticPolicy,
    evaluate_qualification,
    generate_markdown_report,
    write_report,
    compute_oracle_action,
    compute_oracle_utility,
    compute_always_action_utility,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic 3-action data generation
# ──────────────────────────────────────────────────────────────────────

def _generate_3action_experiment(
    n_train=120, n_test=60, n_groups=6, d=12, seed=42
):
    """Generate a complete synthetic 3-action experiment.

    Creates counterfactual sets with structured utility patterns:
    - Cluster 0 tasks: reasoning.direct is best
    - Cluster 1 tasks: retrieval.vector is best
    - Cluster 2 tasks: reasoning.decompose is best

    Returns
    -------
    train_cf : list[CounterfactualSet]
    test_cf : list[CounterfactualSet]
    train_features : np.ndarray  shape [n_train, d]
    test_features : np.ndarray   shape [n_test, d]
    train_utils : np.ndarray     shape [n_train, 3]
    test_utils : np.ndarray      shape [n_test, 3]
    utility_model : UtilityModel
    action_space : ActionSpace
    """
    rng = np.random.RandomState(seed)

    action_space = ActionSpace(actions=(
        ActionDescriptor(action_id="action.reasoning.direct", cost_estimate=0.15),
        ActionDescriptor(action_id="action.retrieval.vector", cost_estimate=0.10),
        ActionDescriptor(action_id="action.reasoning.decompose", cost_estimate=0.30),
    ))

    utility_model = UtilityModel(
        quality_weight=1.0,
        lambda_time=0.01,
        lambda_compute=0.1,
        lambda_risk=1.0,
        time_reference_ms=1000.0,
        compute_reference=1.0,
    )

    # Cluster centers in feature space
    centers = rng.randn(3, d) * 2.0

    def _generate_set(n, prefix):
        cf_sets = []
        features = []
        utils_matrix = []
        for i in range(n):
            cluster = i % 3
            group = f"g{i % n_groups}"
            subtype = f"subtype_{cluster}"

            # Feature = cluster center + noise
            h = centers[cluster] + rng.randn(d) * 0.5
            features.append(h)

            state = ExecutiveState(
                task_id=f"{prefix}_{i}",
                prompt=f"Solve problem {i}",
                task_metadata={"subtype": subtype, "group_id": group, "split": "test"},
            )

            # Action correctness depends on cluster
            # Cluster 0: direct is correct, others wrong
            # Cluster 1: retrieval is correct, others wrong
            # Cluster 2: decompose is correct, others wrong
            correct = [False, False, False]
            correct[cluster] = True

            # Latencies vary by action
            latencies = [200.0, 100.0, 600.0]
            costs = [0.15, 0.10, 0.30]

            executions = {}
            for j, action_id in enumerate(action_space.action_ids):
                executions[action_id] = ActionExecution(
                    action_id=action_id,
                    selected=(j == cluster),  # oracle selection
                    executed=True,
                    output=str(i) if correct[j] else "wrong",
                    verified_correct=correct[j],
                    verifier_name="numeric",
                    latency_ms=latencies[j] + rng.randn() * 20,
                    compute_cost=costs[j],
                )

            cf = CounterfactualSet(
                state=state,
                executions=executions,
                selected_action=action_space.action_ids[cluster],
            )
            cf_sets.append(cf)

            # Compute utilities
            breakdowns = utility_model.compute_all(cf)
            utils_matrix.append([breakdowns[a].utility for a in action_space.action_ids])

        return cf_sets, np.array(features), np.array(utils_matrix)

    train_cf, train_feats, train_utils = _generate_set(n_train, "train")
    test_cf, test_feats, test_utils = _generate_set(n_test, "test")

    return (train_cf, test_cf, train_feats, test_feats,
            train_utils, test_utils, utility_model, action_space)


# ──────────────────────────────────────────────────────────────────────
# End-to-End Integration Test
# ──────────────────────────────────────────────────────────────────────

class TestThreeActionEndToEnd:
    """Full pipeline: data → utility → policy → qualification → report."""

    def test_full_pipeline_logistic(self):
        """Test the complete pipeline with logistic policy."""
        (train_cf, test_cf, train_feats, test_feats,
         train_utils, test_utils, um, space) = _generate_3action_experiment()

        # 1. Train policy
        policy = ExecutiveLogisticPolicy()
        weights = np.ones(len(train_cf))
        policy.fit(train_feats, train_utils, weights, space, n_iter=300)

        # 2. Predict on test set
        test_probs = policy.predict_proba(test_feats)
        assert test_probs.shape == (len(test_cf), 3)

        # 3. Build ExecutiveTaskRecords
        records = []
        for i, cf in enumerate(test_cf):
            probs = test_probs[i]
            selected_idx = int(np.argmax(probs))
            selected_action = space.action_ids[selected_idx]

            # P0 baseline: always select action.reasoning.direct
            p0_action = space.action_ids[0]
            p0_utility = test_utils[i, 0]

            # P1 realized utility
            p1_utility = test_utils[i, selected_idx]

            # Oracle
            oracle_idx = int(np.argmax(test_utils[i]))
            oracle_action = space.action_ids[oracle_idx]
            oracle_utility = test_utils[i, oracle_idx]

            records.append(ExecutiveTaskRecord(
                task_id=cf.state.task_id,
                group_id=cf.state.group_id or "g0",
                subtype=cf.state.subtype or "subtype_0",
                split="test",
                utilities={space.action_ids[j]: test_utils[i, j] for j in range(3)},
                probabilities={space.action_ids[j]: float(probs[j]) for j in range(3)},
                selected_action=selected_action,
                oracle_action=oracle_action,
                p1_realized_utility=float(p1_utility),
                p0_realized_utility=float(p0_utility),
                oracle_utility=float(oracle_utility),
                regret=float(oracle_utility - p1_utility),
            ))

        assert len(records) == len(test_cf)

        # 4. Evaluate qualification
        result = evaluate_qualification(
            records, space,
            experiment_id="b1_integration_test",
            bootstrap_iterations=1000,
            bootstrap_seed=42,
        )

        assert result.n_tasks == len(test_cf)
        assert result.n_groups > 0
        assert len(result.always_action_utilities) == 3

        # 5. Generate report
        md = generate_markdown_report(result, include_per_group=True, records=records)
        assert "Executive Qualification Report" in md
        assert "b1_integration_test" in md
        assert "action.reasoning.direct" in md
        assert "action.retrieval.vector" in md
        assert "action.reasoning.decompose" in md
        assert "P1 - P0" in md
        assert "Per-Group" in md

        # 6. The policy should do better than P0 (always-direct)
        # Since the data has clear cluster structure, P1 should capture most of it
        assert result.p1_mean_utility >= result.p0_mean_utility - 0.1

    def test_full_pipeline_centroid(self):
        """Test the complete pipeline with centroid policy."""
        (train_cf, test_cf, train_feats, test_feats,
         train_utils, test_utils, um, space) = _generate_3action_experiment()

        policy = ExecutiveCentroidPolicy()
        weights = np.ones(len(train_cf))
        policy.fit(train_feats, train_utils, weights, space)

        test_probs = policy.predict_proba(test_feats)
        assert test_probs.shape == (len(test_cf), 3)

        # Build records
        records = []
        for i, cf in enumerate(test_cf):
            probs = test_probs[i]
            selected_idx = int(np.argmax(probs))
            selected_action = space.action_ids[selected_idx]
            p0_utility = test_utils[i, 0]
            p1_utility = test_utils[i, selected_idx]
            oracle_idx = int(np.argmax(test_utils[i]))
            oracle_utility = test_utils[i, oracle_idx]

            records.append(ExecutiveTaskRecord(
                task_id=cf.state.task_id,
                group_id=cf.state.group_id or "g0",
                subtype=cf.state.subtype or "subtype_0",
                split="test",
                utilities={space.action_ids[j]: test_utils[i, j] for j in range(3)},
                probabilities={space.action_ids[j]: float(probs[j]) for j in range(3)},
                selected_action=selected_action,
                oracle_action=space.action_ids[oracle_idx],
                p1_realized_utility=float(p1_utility),
                p0_realized_utility=float(p0_utility),
                oracle_utility=float(oracle_utility),
                regret=float(oracle_utility - p1_utility),
            ))

        result = evaluate_qualification(
            records, space,
            experiment_id="b1_centroid_test",
            bootstrap_iterations=500,
        )
        assert result.n_tasks == len(test_cf)

    def test_oracle_computation(self):
        """Oracle should select the correct action for each cluster."""
        (train_cf, test_cf, train_feats, test_feats,
         train_utils, test_utils, um, space) = _generate_3action_experiment(n_test=30)

        for i, cf in enumerate(test_cf):
            best_id, best_u = compute_oracle_action(cf, um)
            cluster = i % 3
            expected_action = space.action_ids[cluster]
            assert best_id == expected_action, (
                f"task {i}: oracle={best_id}, expected={expected_action}")

    def test_always_action_utilities(self):
        """Always-action utilities should be computed correctly."""
        (train_cf, test_cf, _, _,
         _, _, um, space) = _generate_3action_experiment(n_test=30)

        for action_id in space.action_ids:
            u = compute_always_action_utility(test_cf, action_id, um)
            assert 0.0 <= u <= 1.0

        # Oracle should be better than any always-action
        oracle_u = compute_oracle_utility(test_cf, um)
        for action_id in space.action_ids:
            always_u = compute_always_action_utility(test_cf, action_id, um)
            assert oracle_u >= always_u - 0.01

    def test_write_report_to_disk(self, tmp_path):
        """Report files should be written correctly."""
        (train_cf, test_cf, train_feats, test_feats,
         train_utils, test_utils, um, space) = _generate_3action_experiment(n_test=30)

        policy = ExecutiveLogisticPolicy()
        policy.fit(train_feats, train_utils, np.ones(len(train_cf)), space, n_iter=100)
        test_probs = policy.predict_proba(test_feats)

        records = []
        for i, cf in enumerate(test_cf):
            probs = test_probs[i]
            selected_idx = int(np.argmax(probs))
            oracle_idx = int(np.argmax(test_utils[i]))
            records.append(ExecutiveTaskRecord(
                task_id=cf.state.task_id,
                group_id=cf.state.group_id or "g0",
                subtype=cf.state.subtype or "subtype_0",
                split="test",
                utilities={space.action_ids[j]: test_utils[i, j] for j in range(3)},
                probabilities={space.action_ids[j]: float(probs[j]) for j in range(3)},
                selected_action=space.action_ids[selected_idx],
                oracle_action=space.action_ids[oracle_idx],
                p1_realized_utility=float(test_utils[i, selected_idx]),
                p0_realized_utility=float(test_utils[i, 0]),
                oracle_utility=float(test_utils[i, oracle_idx]),
                regret=float(test_utils[i, oracle_idx] - test_utils[i, selected_idx]),
            ))

        result = evaluate_qualification(
            records, space, experiment_id="disk_test", bootstrap_iterations=200)
        paths = write_report(result, tmp_path, records=records)

        assert paths["markdown"].exists()
        assert paths["json"].exists()
        md_content = paths["markdown"].read_text()
        assert "disk_test" in md_content
        import json
        json_data = json.loads(paths["json"].read_text())
        assert json_data["experiment_id"] == "disk_test"
        assert json_data["n_tasks"] == 30
