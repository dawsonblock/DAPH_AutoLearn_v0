#!/usr/bin/env python3
"""DAPH v0.4 — Executive experiment runner.

Runs a generic N-action executive qualification experiment from a YAML
config. This is the v0.4 equivalent of ``run_gate_a_staged.py`` but
works with arbitrary action spaces via the
:mod:`daph_learning.executive` module.

Usage:
    # With a vLLM server running on localhost:8000
    python scripts/run_executive_experiment.py \\
        --config configs/executive_b1_direct_vs_retrieval_vs_decompose.yaml \\
        --output-dir artifacts/executive_b1/

    # With a mock generator (for testing)
    python scripts/run_executive_experiment.py \\
        --config configs/executive_b1_direct_vs_retrieval_vs_decompose.yaml \\
        --output-dir artifacts/executive_b1/ \\
        --mock

    # With synthetic data (no real model needed)
    python scripts/run_executive_experiment.py \\
        --config configs/executive_b1_direct_vs_retrieval_vs_decompose.yaml \\
        --output-dir artifacts/executive_b1/ \\
        --mock --synthetic

Pipeline:
    1. Load config → build ActionSpace, UtilityModel, executors
    2. Load or generate tasks
    3. Counterfactual execution: run all actions on all tasks
    4. Build executive experiences
    5. Train executive policy (centroid or logistic)
    6. Predict on held-out set
    7. Build ExecutiveTaskRecords
    8. Run evaluate_qualification()
    9. Write reports (markdown + JSON)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daph_learning.executive import (
    ActionDescriptor,
    ActionSpace,
    UtilityModel,
    ExecutiveLogisticPolicy,
    ExecutiveCentroidPolicy,
    make_executive_policy,
    ExecutorRegistry,
    LLMGenerationConfig,
    DirectReasoningExecutor,
    RetrievalVectorExecutor,
    ReasoningDecomposeExecutor,
    build_b1_executors,
    build_executive_experiences,
    experiences_to_training_arrays,
    build_executive_training_targets,
    evaluate_qualification,
    write_report,
    ExecutiveTaskRecord,
    generate_diverse_tasks,
    build_retrieval_store,
    HiddenStateConfig,
    load_model_for_capture,
    capture_hidden_states,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_action_space(config: dict) -> ActionSpace:
    """Build ActionSpace from config."""
    actions_config = config.get("action_space", {})
    actions = []
    for a in actions_config.get("actions", []):
        actions.append(ActionDescriptor(
            action_id=a["action_id"],
            display_name=a.get("display_name", a["action_id"]),
            description=a.get("description", ""),
            cost_estimate=a.get("cost_estimate", 0.0),
            tags=tuple(a.get("tags", [])),
        ))
    return ActionSpace(actions=tuple(actions))


def build_utility_model(config: dict) -> UtilityModel:
    """Build UtilityModel from config."""
    uc = config.get("utility_model", {})
    return UtilityModel(
        quality_weight=uc.get("quality_weight", 1.0),
        lambda_time=uc.get("lambda_time", 0.01),
        lambda_compute=uc.get("lambda_compute", 0.1),
        lambda_risk=uc.get("lambda_risk", 1.0),
        time_reference_ms=uc.get("time_reference_ms", 1000.0),
        compute_reference=uc.get("compute_reference", 1.0),
        abstention_band=uc.get("abstention_band", 0.05),
    )


def build_llm_config(config: dict) -> LLMGenerationConfig:
    """Build LLMGenerationConfig from config."""
    mc = config.get("model", {})
    return LLMGenerationConfig(
        model_id=mc.get("name", ""),
        max_tokens=mc.get("max_tokens", 512),
        temperature=mc.get("temperature", 0.0),
        vllm_port=mc.get("vllm_port", 8000),
        vllm_api_key=mc.get("vllm_api_key", ""),
        vllm_api_key_env=mc.get("vllm_api_key_env", ""),
    )


def build_executors(
    config: dict,
    llm_config: LLMGenerationConfig,
    *,
    mock: bool = False,
    examples: list[dict] | None = None,
) -> ExecutorRegistry:
    """Build executor registry from config."""
    if mock:
        # Use a mock generator that returns a fixed answer
        def mock_gen(prompt, cfg):
            return "FINAL_ANSWER: 42", 50.0
        return build_b1_executors(
            llm_config, examples=examples or [], generate_fn=mock_gen)

    return build_b1_executors(
        llm_config, examples=examples or [])


def generate_synthetic_tasks(
    n_tasks: int = 100,
    n_groups: int = 10,
    n_subtypes: int = 3,
    seed: int = 42,
    *,
    diverse: bool = False,
) -> list[dict]:
    """Generate synthetic arithmetic tasks for testing.

    If ``diverse=True``, uses the diverse task generator with 8 subtypes
    (addition, multiplication, word problems, multi-step, comparison).
    Otherwise uses simple addition with 3 difficulty levels.
    """
    if diverse:
        return generate_diverse_tasks(
            n_tasks=n_tasks, n_groups=n_groups, seed=seed)

    rng = np.random.RandomState(seed)
    tasks = []
    for i in range(n_tasks):
        subtype = f"subtype_{i % n_subtypes}"
        group = f"g{i % n_groups}"

        # Difficulty varies by subtype
        if i % n_subtypes == 0:
            a, b = int(rng.randint(1, 20)), int(rng.randint(1, 20))
        elif i % n_subtypes == 1:
            a, b = int(rng.randint(10, 100)), int(rng.randint(10, 100))
        else:
            a, b = int(rng.randint(50, 500)), int(rng.randint(50, 500))

        tasks.append({
            "task_id": f"synthetic_{i:04d}",
            "prompt": f"What is {a} + {b}?",
            "answer": a + b,
            "subtype": subtype,
            "group_id": group,
        })
    return tasks


def run_experiment(
    config: dict,
    output_dir: str,
    *,
    mock: bool = False,
    synthetic: bool = False,
    n_tasks: int | None = None,
) -> dict:
    """Run the full executive experiment pipeline."""
    t_start = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_id = config.get("experiment_id", "executive_experiment")
    print(f"\n{'='*60}")
    print(f"  DAPH v0.4 Executive Experiment: {experiment_id}")
    print(f"{'='*60}\n")

    # 1. Build action space, utility model, executors
    space = build_action_space(config)
    print(f"Action space: {len(space.action_ids)} actions")
    for a in space.action_ids:
        print(f"  - {a}")

    um = build_utility_model(config)
    llm_config = build_llm_config(config)

    # 2. Load or generate tasks
    data_cfg = config.get("data", {})
    use_diverse = data_cfg.get("diverse", True)
    use_hidden_states = config.get("model", {}).get("capture_hidden_states", False)

    if synthetic:
        n_total = n_tasks or data_cfg.get("n_tasks_train", 400) + data_cfg.get("n_tasks_test", 200)
        n_groups = data_cfg.get("n_groups", 40)
        n_subtypes = data_cfg.get("n_subtypes", 8)
        seed = data_cfg.get("seed", 42)
        tasks = generate_synthetic_tasks(
            n_tasks=n_total,
            n_groups=n_groups,
            n_subtypes=n_subtypes,
            seed=seed,
            diverse=use_diverse,
        )
        # Split into train/test
        n_train = data_cfg.get("n_tasks_train", int(len(tasks) * 0.6))
        n_train = min(n_train, len(tasks) - 20)  # ensure at least 20 test
        train_tasks = tasks[:n_train]
        test_tasks = tasks[n_train:]
        print(f"{'Diverse' if use_diverse else 'Simple'} tasks: "
              f"{len(train_tasks)} train, {len(test_tasks)} test")
    else:
        raise NotImplementedError(
            "Real data loading not yet implemented. Use --synthetic for testing.")

    # 3. Build executors with diverse retrieval store
    if use_diverse:
        examples = build_retrieval_store(train_tasks, n_per_subtype=5)
    else:
        examples = [{"prompt": t["prompt"], "answer": t["answer"]}
                    for t in train_tasks[:20]]
    registry = build_executors(config, llm_config, mock=mock, examples=examples)

    # 4. Counterfactual execution (concurrent)
    print(f"\nExecuting {len(train_tasks)} train tasks × {len(space.action_ids)} actions...")
    train_cf = registry.execute_all_tasks(train_tasks, space, max_concurrent=64)

    print(f"\nExecuting {len(test_tasks)} test tasks × {len(space.action_ids)} actions...")
    test_cf = registry.execute_all_tasks(test_tasks, space, max_concurrent=64)

    # 5. Build experiences
    print("\nBuilding executive experiences...")
    train_experiences = build_executive_experiences(train_cf, um, space)
    test_experiences = build_executive_experiences(test_cf, um, space)
    print(f"  {len(train_experiences)} train, {len(test_experiences)} test")

    # 6. Extract features (hidden states or random fallback)
    mc = config.get("model", {})
    use_hidden_states = mc.get("capture_hidden_states", False)

    if use_hidden_states and not mock:
        print("\nCapturing hidden states from HF model...")
        hs_cfg = HiddenStateConfig(
            model_name=mc.get("name", ""),
            layers=mc.get("capture_layers", [0.5]),
            location=mc.get("capture_location", "last_token"),
            max_length=mc.get("capture_max_length", 512),
            batch_size=mc.get("capture_batch_size", 16),
        )
        device = mc.get("device", "cuda")
        print(f"  Loading model {hs_cfg.model_name} on {device}...")
        cap_model, cap_tokenizer = load_model_for_capture(
            hs_cfg.model_name, device=device, dtype=mc.get("dtype", "auto"))
        print(f"  Capturing train features ({len(train_tasks)} tasks)...")
        train_features = capture_hidden_states(
            train_tasks, cap_model, cap_tokenizer, hs_cfg, device=device)
        print(f"  Capturing test features ({len(test_tasks)} tasks)...")
        test_features = capture_hidden_states(
            test_tasks, cap_model, cap_tokenizer, hs_cfg, device=device)
        # Free model memory
        del cap_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Feature dim: {train_features.shape[1]}")
    else:
        print("\nUsing random features (no hidden state capture)...")
        rng = np.random.RandomState(config.get("data", {}).get("seed", 42))
        d = 16  # feature dimension
        train_features = rng.randn(len(train_experiences), d).astype(np.float32)
        test_features = rng.randn(len(test_experiences), d).astype(np.float32)

    # 7. Build training arrays
    train_arrays = experiences_to_training_arrays(
        train_experiences, space, features=train_features)

    # 8. Train policy
    policy_cfg = config.get("policy", {})
    policy_type = policy_cfg.get("type", "logistic")
    print(f"\nTraining {policy_type} policy...")
    policy = make_executive_policy(policy_type)
    if policy_type == "logistic":
        policy.fit(
            train_arrays["features"], train_arrays["utilities"],
            train_arrays["weights"], space,
            lr=policy_cfg.get("learning_rate", 0.01),
            n_iter=policy_cfg.get("n_iter", 500),
            l2=policy_cfg.get("l2", 0.0001),
        )
    else:
        policy.fit(
            train_arrays["features"], train_arrays["utilities"],
            train_arrays["weights"], space,
        )
    print(f"  Trained on {policy.n_train_} examples")

    # 9. Predict on test set
    test_probs = policy.predict_proba(test_features)
    print(f"  Predicted on {test_probs.shape[0]} test examples")

    # 10. Build ExecutiveTaskRecords
    test_arrays = experiences_to_training_arrays(
        test_experiences, space, features=test_features)

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
                       for j in range(len(space.action_ids))},
            probabilities={space.action_ids[j]: float(p[j])
                           for j in range(len(space.action_ids))},
            selected_action=space.action_ids[sel_idx],
            oracle_action=space.action_ids[oracle_idx],
            p1_realized_utility=float(test_arrays["utilities"][i, sel_idx]),
            p0_realized_utility=float(test_arrays["utilities"][i, 0]),
            oracle_utility=float(test_arrays["utilities"][i, oracle_idx]),
            regret=float(test_arrays["utilities"][i, oracle_idx] -
                         test_arrays["utilities"][i, sel_idx]),
        ))

    # 11. Evaluate qualification
    qual_cfg = config.get("qualification", {})
    print(f"\nRunning qualification evaluation "
          f"({qual_cfg.get('bootstrap_iterations', 10000)} bootstrap iterations)...")
    result = evaluate_qualification(
        records, space,
        experiment_id=experiment_id,
        bootstrap_iterations=qual_cfg.get("bootstrap_iterations", 10000),
        bootstrap_seed=qual_cfg.get("bootstrap_seed", 42),
    )

    print(f"\n{'─'*60}")
    print(f"  Results:")
    print(f"    P1 mean utility:  {result.p1_mean_utility:.4f}")
    print(f"    P0 mean utility:  {result.p0_mean_utility:.4f}")
    p1_p0 = result.p1_minus_p0
    print(f"    P1 - P0 (point):  {p1_p0.get('point', 0.0):.4f}")
    print(f"    P1 - P0 (LCB):    {p1_p0.get('lcb', 0.0):.4f}")
    print(f"    P1 - P0 (UCB):    {p1_p0.get('ucb', 0.0):.4f}")
    print(f"    Oracle utility:   {result.oracle_mean_utility:.4f}")
    print(f"    Oracle gap:       {result.oracle_gap:.4f}")
    print(f"    Gap capture:      {result.oracle_gap_capture:.4f}")
    p1_sham = result.p1_minus_sham
    if p1_sham:
        print(f"    P1 - Sham (point):{p1_sham.get('point', 0.0):.4f}")
    print(f"    Pos group frac:   {result.positive_group_fraction:.4f}")
    print(f"    Always-action utilities:")
    for aid, u in result.always_action_utilities.items():
        print(f"      {aid}: {u:.4f}")
    print(f"{'─'*60}")

    # 12. Write reports
    print(f"\nWriting reports to {output_path}...")
    paths = write_report(result, output_path, records=records)
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON:     {paths['json']}")

    # 13. Save policy
    policy_path = output_path / "policy.json"
    policy.save(str(policy_path))
    print(f"  Policy:   {policy_path}")

    # 14. Save counterfactual sets (provenance)
    cf_path = output_path / "counterfactual_sets.json"
    cf_data = []
    for cf in test_cf:
        cf_data.append({
            "task_id": cf.task_id,
            "executions": {
                aid: {
                    "executed": e.executed,
                    "verified_correct": e.verified_correct,
                    "latency_ms": e.latency_ms,
                    "compute_cost": e.compute_cost,
                    "failure_type": e.failure_type,
                }
                for aid, e in cf.executions.items()
            }
        })
    with open(cf_path, "w") as f:
        json.dump(cf_data, f, indent=2)
    print(f"  CF sets:  {cf_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print(f"  Experiment complete: {experiment_id}")
    print(f"{'='*60}\n")

    return {
        "experiment_id": experiment_id,
        "result": result,
        "output_dir": str(output_path),
        "n_train": len(train_experiences),
        "n_test": len(test_experiences),
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a DAPH v0.4 executive qualification experiment.")
    parser.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML config.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write output artifacts.")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use a mock LLM generator (for testing).")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic tasks instead of loading real data.")
    parser.add_argument(
        "--n-tasks", type=int, default=None,
        help="Override number of synthetic tasks.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(
        config, args.output_dir,
        mock=args.mock,
        synthetic=args.synthetic,
        n_tasks=args.n_tasks,
    )


if __name__ == "__main__":
    main()
