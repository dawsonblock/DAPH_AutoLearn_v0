#!/usr/bin/env python3
"""DAPH v0.4 — B1 experiment with cluster-structured mock data.

This script runs the B1 experiment with a mock generator that simulates
the expected behavior: different actions succeed on different task types.

    Subtype 0 (small numbers):   direct reasoning succeeds
    Subtype 1 (medium numbers):  retrieval succeeds
    Subtype 2 (large numbers):   decomposition succeeds

This produces a non-trivial qualification result where the learned policy
should outperform any single always-action baseline.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    UtilityModel,
    LLMGenerationConfig,
    DirectReasoningExecutor,
    RetrievalVectorExecutor,
    ReasoningDecomposeExecutor,
    ExecutorRegistry,
    build_executive_experiences,
    experiences_to_training_arrays,
    ExecutiveLogisticPolicy,
    evaluate_qualification,
    write_report,
    ExecutiveTaskRecord,
)


def make_cluster_aware_generators():
    """Create generators where each action succeeds on a different subtype.

    Returns a dict of action_id → generate_fn.
    """
    def direct_gen(prompt, config):
        # Direct succeeds on subtype 0 (small numbers)
        if "subtype_0" in prompt:
            return "FINAL_ANSWER: 42", 50.0
        return "FINAL_ANSWER: 0", 50.0

    def retrieval_gen(prompt, config):
        # Retrieval succeeds on subtype 1 (medium numbers)
        if "subtype_1" in prompt:
            return "FINAL_ANSWER: 42", 100.0
        return "FINAL_ANSWER: 0", 100.0

    def decompose_gen(prompt, config):
        # Decompose succeeds on subtype 2 (large numbers)
        if "subtype_2" in prompt:
            return "FINAL_ANSWER: 42", 200.0
        return "FINAL_ANSWER: 0", 200.0

    return {
        "action.reasoning.direct": direct_gen,
        "action.retrieval.lexical": retrieval_gen,
        "action.reasoning.decompose": decompose_gen,
    }


def generate_clustered_tasks(n=90, seed=42):
    """Generate tasks with 3 subtypes, each favoring a different action."""
    rng = np.random.RandomState(seed)
    tasks = []
    for i in range(n):
        subtype_idx = i % 3
        subtype = f"subtype_{subtype_idx}"
        group = f"g{i % 9}"

        if subtype_idx == 0:
            a, b = int(rng.randint(1, 20)), int(rng.randint(1, 20))
        elif subtype_idx == 1:
            a, b = int(rng.randint(10, 100)), int(rng.randint(10, 100))
        else:
            a, b = int(rng.randint(50, 500)), int(rng.randint(50, 500))

        tasks.append({
            "task_id": f"clustered_{i:04d}",
            "prompt": f"What is {a} + {b}? (subtype_{subtype_idx})",
            "answer": 42,  # all generators that succeed return 42
            "subtype": subtype,
            "group_id": group,
        })
    return tasks


def main():
    print("\n" + "=" * 60)
    print("  B1 Experiment: Cluster-Structured Mock Data")
    print("=" * 60 + "\n")

    # Build action space
    space = ActionSpace(actions=(
        ActionDescriptor(action_id="action.reasoning.direct", cost_estimate=0.15),
        ActionDescriptor(action_id="action.retrieval.lexical", cost_estimate=0.10),
        ActionDescriptor(action_id="action.reasoning.decompose", cost_estimate=0.30),
    ))

    # Build utility model
    um = UtilityModel(
        quality_weight=1.0, lambda_time=0.01,
        lambda_compute=0.1, lambda_risk=1.0,
        time_reference_ms=1000.0, compute_reference=1.0,
    )

    # Build executors with cluster-aware generators
    gen_map = make_cluster_aware_generators()
    config = LLMGenerationConfig(model_id="mock")
    registry = ExecutorRegistry()
    registry.register(DirectReasoningExecutor(
        config=config, generate_fn=gen_map["action.reasoning.direct"]))
    registry.register(RetrievalVectorExecutor(
        config=config, examples=[],
        generate_fn=gen_map["action.retrieval.lexical"]))
    registry.register(ReasoningDecomposeExecutor(
        config=config, generate_fn=gen_map["action.reasoning.decompose"]))

    # Generate tasks
    tasks = generate_clustered_tasks(n=90)
    split_idx = int(len(tasks) * 0.6)
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]
    print(f"Tasks: {len(train_tasks)} train, {len(test_tasks)} test")
    print(f"Subtypes: 3 (each favors a different action)\n")

    # Execute
    print("Executing all actions on all tasks...")
    t0 = time.time()
    train_cf = [registry.execute_all(task, space) for task in train_tasks]
    test_cf = [registry.execute_all(task, space) for task in test_tasks]
    print(f"  Done in {time.time() - t0:.1f}s")

    # Build experiences
    train_exp = build_executive_experiences(train_cf, um, space)
    test_exp = build_executive_experiences(test_cf, um, space)

    # Features: encode subtype as one-hot + noise
    def extract_features(cf_sets, d=12):
        rng = np.random.RandomState(123)
        features = []
        for cf in cf_sets:
            subtype = cf.state.subtype or "subtype_0"
            st_idx = int(subtype.split("_")[-1]) if "_" in subtype else 0
            feat = np.zeros(d, dtype=np.float32)
            feat[st_idx] = 1.0
            feat[3:] = rng.randn(d - 3) * 0.1
            features.append(feat)
        return np.array(features)

    train_feats = extract_features(train_cf)
    test_feats = extract_features(test_cf)

    # Train policy
    print("\nTraining logistic policy...")
    arrays = experiences_to_training_arrays(train_exp, space, features=train_feats)
    policy = ExecutiveLogisticPolicy()
    policy.fit(arrays["features"], arrays["utilities"],
               arrays["weights"], space, n_iter=300)
    print(f"  Trained on {policy.n_train_} examples")

    # Predict
    test_probs = policy.predict_proba(test_feats)
    test_arrays = experiences_to_training_arrays(test_exp, space, features=test_feats)

    # Build records
    records = []
    for i, cf in enumerate(test_cf):
        p = test_probs[i]
        sel_idx = int(np.argmax(p))
        oracle_idx = int(np.argmax(test_arrays["utilities"][i]))
        records.append(ExecutiveTaskRecord(
            task_id=cf.task_id,
            group_id=cf.state.group_id or "g0",
            subtype=cf.state.subtype or "subtype_0",
            split="test",
            utilities={space.action_ids[j]: float(test_arrays["utilities"][i, j])
                       for j in range(3)},
            probabilities={space.action_ids[j]: float(p[j]) for j in range(3)},
            selected_action=space.action_ids[sel_idx],
            oracle_action=space.action_ids[oracle_idx],
            p1_realized_utility=float(test_arrays["utilities"][i, sel_idx]),
            p0_realized_utility=float(test_arrays["utilities"][i, 0]),
            oracle_utility=float(test_arrays["utilities"][i, oracle_idx]),
            regret=float(test_arrays["utilities"][i, oracle_idx] -
                         test_arrays["utilities"][i, sel_idx]),
        ))

    # Evaluate
    print("\nRunning qualification (10000 bootstrap iterations)...")
    result = evaluate_qualification(
        records, space,
        experiment_id="b1_cluster_mock",
        bootstrap_iterations=10000,
    )

    print(f"\n{'─' * 60}")
    print(f"  Results:")
    print(f"    P1 mean utility:  {result.p1_mean_utility:.4f}")
    print(f"    P0 mean utility:  {result.p0_mean_utility:.4f}")
    p1_p0 = result.p1_minus_p0
    print(f"    P1 - P0 (point):  {p1_p0.get('point', 0.0):.4f}")
    print(f"    P1 - P0 (LCB):    {p1_p0.get('lcb', 0.0):.4f}")
    print(f"    Oracle utility:   {result.oracle_mean_utility:.4f}")
    print(f"    Oracle gap:       {result.oracle_gap:.4f}")
    print(f"    Gap capture:      {result.oracle_gap_capture:.2%}")
    print(f"    Pos group frac:   {result.positive_group_fraction:.2%}")
    print(f"\n    Always-action utilities:")
    for aid, u in result.always_action_utilities.items():
        print(f"      {aid}: {u:.4f}")
    print(f"{'─' * 60}")

    # Write report
    output_dir = Path("artifacts/executive_b1_cluster")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = write_report(result, output_dir, records=records)
    print(f"\nReports written to {output_dir}")
    print(f"  {paths['markdown'].name}")
    print(f"  {paths['json'].name}")

    # Verify the policy learned something useful
    print(f"\n  Policy learned to route by subtype:")
    for st in ["subtype_0", "subtype_1", "subtype_2"]:
        st_records = [r for r in records if r.subtype == st]
        if st_records:
            selections = [r.selected_action for r in st_records]
            from collections import Counter
            counts = Counter(selections)
            print(f"    {st}: {dict(counts)}")

    print(f"\n{'=' * 60}")
    print("  B1 cluster experiment complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
