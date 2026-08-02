#!/usr/bin/env python3
"""DAPH v0.4.0a2 — B5 Staged Runner: Adaptive Inference Compute.

This is the canonical real-model experiment entrypoint for B5.

Required stages:
  prepare       — freeze config, generate dataset, run split checks
  counterfactuals — execute all actions on all tasks, save outcomes
  representations — capture hidden states, save NPZ
  train         — train policies, surface baselines, shams
  qualify       — evaluate FINAL once, run gates, generate report
  reproduce     — verify reproduction

Usage:
  python scripts/run_b5_staged.py prepare --config configs/b5.yaml
  python scripts/run_b5_staged.py counterfactuals --resume
  python scripts/run_b5_staged.py representations --resume
  python scripts/run_b5_staged.py train
  python scripts/run_b5_staged.py qualify
  python scripts/run_b5_staged.py reproduce

  python scripts/run_b5_staged.py all --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daph_learning.executive.b5_actions import B5_ACTION_IDS, b5_action_space
from daph_learning.executive.b5_dataset import generate_b5_dataset
from daph_learning.executive.b5_policies import (
    LinearQPolicy, RidgeQPolicy, MLPQPolicy,
    SurfaceFeatureExtractor, SurfaceEnsemblePolicy,
)
from daph_learning.executive.b5_qualification import evaluate_gates, GateThresholds
from daph_learning.executive.b5_diagnostics import (
    empirical_crossover_analysis,
    think_fast_delta_analysis,
    compute_budget_frontier,
)
from daph_learning.executive.error_semantics import (
    ExecutionStatus, ObservedCost, compute_observed_utility,
)
from daph_learning.executive.stats import (
    make_paired_comparison, paired_group_bootstrap,
    compute_group_local_results, positive_group_fraction,
    worst_group_delta, gap_capture, selection_accuracy,
    action_advantage_margin, run_matched_sham_evaluation,
)
from daph_learning.executive.leakage import (
    check_task_id_overlap, check_exact_prompt_leakage,
    check_group_leakage, check_api_key_placeholder,
    run_leakage_checks_from_artifacts,
)
from daph_learning.executive.lifecycle import (
    ExperimentState, ExperimentStatus, FrozenConfig,
)
from daph_learning.executive.manifest import ManifestBuilder
from daph_learning.executive.artifact_integrity import (
    validate_required_tree, B5_REQUIRED_ARTIFACTS,
)


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_output_dir(config: dict) -> Path:
    return Path(config.get("output_dir", "artifacts/executive_b5_adaptive_compute"))


def _config_hash(config: dict) -> str:
    """Compute SHA-256 hash of config."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()


def _task_hash(task: dict) -> str:
    """Compute hash of a task for resume safety."""
    return hashlib.sha256(
        json.dumps(task, sort_keys=True, default=str).encode()
    ).hexdigest()


def _action_config_hash(action_id: str, config: dict) -> str:
    """Compute hash of action configuration."""
    action_config = {
        "action_id": action_id,
        "model": config.get("model", {}),
        "action_params": config.get("actions", {}).get(action_id, {}),
    }
    return hashlib.sha256(
        json.dumps(action_config, sort_keys=True, default=str).encode()
    ).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Stage: prepare
# ──────────────────────────────────────────────────────────────────────

def stage_prepare(config: dict) -> None:
    """Freeze config, generate dataset, run split checks."""
    out = get_output_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    experiment_id = config.get("experiment_id", "executive_b5_adaptive_compute")

    print(f"\n{'='*60}")
    print(f"  B5 Stage: prepare")
    print(f"  Output: {out}")
    print(f"{'='*60}")

    # Check for embedded credentials
    cred_check = check_api_key_placeholder(config)
    if not cred_check.passed:
        print(f"  ERROR: {cred_check.detail}")
        sys.exit(1)

    # Freeze config
    frozen = FrozenConfig(config=config)
    config_dir = out / "config"
    config_dir.mkdir(exist_ok=True)
    with open(config_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(config_dir / "config_hash.txt", "w") as f:
        f.write(frozen.config_hash)

    # Experiment state
    state = ExperimentState(experiment_id=experiment_id)
    state.freeze(frozen)
    state.save(out / "status.json")

    # Generate dataset
    ds_config = config.get("dataset", {})
    dataset = generate_b5_dataset(
        n_train=ds_config.get("n_train", 3000),
        n_dev=ds_config.get("n_dev", 750),
        n_final=ds_config.get("n_final", 1000),
        n_ood=ds_config.get("n_ood", 500),
        seed=ds_config.get("seed", 20260802),
        split_mode=ds_config.get("split_mode", "standard"),
    )

    # Save dataset
    ds_dir = out / "dataset"
    ds_dir.mkdir(exist_ok=True)
    for name, split in dataset.items():
        fname = "final_ood.json" if name == "final_ood" else f"{name}.json"
        with open(ds_dir / fname, "w") as f:
            json.dump(split.tasks, f, indent=2)

    all_groups = {}
    for name, split in dataset.items():
        for g in split.groups:
            all_groups[g] = name
    with open(ds_dir / "groups.json", "w") as f:
        json.dump(all_groups, f, indent=2)

    with open(ds_dir / "dataset_manifest.json", "w") as f:
        json.dump({
            "n_train": len(dataset["train"].tasks),
            "n_dev": len(dataset["dev"].tasks),
            "n_final": len(dataset["final"].tasks),
            "n_ood": len(dataset["final_ood"].tasks),
            "split_mode": ds_config.get("split_mode", "standard"),
        }, f, indent=2)

    # Run split checks
    train_ids = [t["task_id"] for t in dataset["train"].tasks]
    dev_ids = [t["task_id"] for t in dataset["dev"].tasks]
    final_ids = [t["task_id"] for t in dataset["final"].tasks]

    checks = []
    checks.append(check_task_id_overlap(train_ids, dev_ids, final_ids))
    checks.append(check_exact_prompt_leakage(
        [t["prompt"] for t in dataset["train"].tasks],
        [t["prompt"] for t in dataset["dev"].tasks],
        [t["prompt"] for t in dataset["final"].tasks],
    ))
    checks.append(check_group_leakage(
        [t["group_id"] for t in dataset["train"].tasks],
        [t["group_id"] for t in dataset["dev"].tasks],
        [t["group_id"] for t in dataset["final"].tasks],
    ))

    checks_dir = out / "checks"
    checks_dir.mkdir(exist_ok=True)
    all_pass = all(c.passed for c in checks)
    with open(checks_dir / "leakage_checks.json", "w") as f:
        json.dump({
            "passed": all_pass,
            "checks": [c.to_dict() if hasattr(c, "to_dict") else c.__dict__ for c in checks],
        }, f, indent=2)

    if not all_pass:
        print("  ERROR: Leakage checks failed!")
        state.mark_failed("leakage")
        state.save(out / "status.json")
        sys.exit(1)

    # Crossover report
    print(f"  Dataset: {len(dataset['train'].tasks)} train, "
          f"{len(dataset['dev'].tasks)} dev, "
          f"{len(dataset['final'].tasks)} final, "
          f"{len(dataset['final_ood'].tasks)} OOD")
    print(f"  Leakage checks: PASS")
    print(f"  Config hash: {frozen.config_hash[:16]}...")
    print(f"  Status: FROZEN")


# ──────────────────────────────────────────────────────────────────────
# Stage: counterfactuals
# ──────────────────────────────────────────────────────────────────────

def stage_counterfactuals(config: dict, resume: bool = False) -> None:
    """Execute all actions on all tasks, save outcomes.

    Supports resume: loads existing records, verifies config hash,
    skips valid completed executions, reruns only missing/corrupt ones.
    """
    out = get_output_dir(config)
    experiment_id = config.get("experiment_id", "executive_b5_adaptive_compute")
    action_space = b5_action_space()
    action_ids = list(B5_ACTION_IDS)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: counterfactuals (resume={resume})")
    print(f"{'='*60}")

    # Load config hash
    config_hash_file = out / "config" / "config_hash.txt"
    if not config_hash_file.exists():
        print("  ERROR: Config not frozen. Run 'prepare' first.")
        sys.exit(1)
    frozen_hash = config_hash_file.read_text().strip()

    cf_dir = out / "counterfactuals"
    cf_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(out / "dataset" / fname) as f:
            dataset[split_name] = json.load(f)

    # Resume: load existing records
    existing_records = {}
    if resume:
        for split_name in ["train", "dev", "final", "final_ood"]:
            cf_file = cf_dir / f"{split_name}.json"
            if cf_file.exists():
                try:
                    records = json.loads(cf_file.read_text())
                    # Verify config hash matches
                    for tid, actions in records.items():
                        for aid, result in actions.items():
                            if result.get("config_hash") == frozen_hash:
                                existing_records[(split_name, tid, aid)] = result
                except (json.JSONDecodeError, KeyError):
                    pass

    print(f"  Existing valid records: {len(existing_records)}")

    # Execute counterfactuals
    # In real mode, this would call vLLM/HF models.
    # In mock mode, use fake executors.
    mock = config.get("mock", False)

    for split_name in ["train", "dev", "final", "final_ood"]:
        tasks = dataset[split_name]
        results = {}
        n_new = 0

        for task in tasks:
            tid = task["task_id"]
            task_h = _task_hash(task)
            results[tid] = {}

            for action_id in action_ids:
                # Check if we can reuse existing record
                key = (split_name, tid, action_id)
                if resume and key in existing_records:
                    results[tid][action_id] = existing_records[key]
                    continue

                # Execute action
                action_cfg_hash = _action_config_hash(action_id, config)

                if mock:
                    # Mock execution
                    hint = task.get("oracle_action_hint", "")
                    difficulty = task.get("difficulty", "medium")
                    base_prob = {"easy": 0.6, "medium": 0.3, "hard": 0.1}[difficulty]
                    if action_id == hint:
                        prob = {"easy": 0.95, "medium": 0.85, "hard": 0.75}[difficulty]
                    else:
                        prob = base_prob
                    rng = np.random.RandomState(hash(tid + action_id) % (2**32))
                    correct = rng.random() < prob
                    status = ExecutionStatus.CORRECT if correct else ExecutionStatus.INCORRECT
                    cost = ObservedCost(
                        prompt_tokens=100,
                        completion_tokens=rng.randint(50, 500),
                        llm_call_count=1,
                        wall_latency_ms=rng.uniform(200, 5000),
                    )
                else:
                    # Real execution would go here
                    print(f"    ERROR: Real execution not implemented. Use --mock.")
                    sys.exit(1)

                utility = compute_observed_utility(status, cost)
                results[tid][action_id] = {
                    "status": status.value,
                    "verified_correct": status == ExecutionStatus.CORRECT,
                    "utility": utility,
                    "cost": cost.to_dict(),
                    "task_hash": task_h,
                    "action_config_hash": action_cfg_hash,
                    "config_hash": frozen_hash,
                    "model_revision": config.get("model", {}).get("revision", ""),
                }
                n_new += 1

        # Save counterfactuals
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(cf_dir / fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {split_name}: {n_new} new executions, {len(results)} total tasks")

    # Save summary
    with open(cf_dir / "summary.json", "w") as f:
        json.dump({
            "n_train": len(dataset["train"]),
            "n_dev": len(dataset["dev"]),
            "n_final": len(dataset["final"]),
            "n_ood": len(dataset["final_ood"]),
            "action_ids": action_ids,
            "config_hash": frozen_hash,
        }, f, indent=2)

    print(f"  Counterfactuals complete.")


# ──────────────────────────────────────────────────────────────────────
# Stage: representations
# ──────────────────────────────────────────────────────────────────────

def stage_representations(config: dict, resume: bool = False) -> None:
    """Capture hidden states, save NPZ.

    In real mode, this loads the HF model and captures hidden states.
    In mock mode, generates synthetic features.
    """
    out = get_output_dir(config)
    print(f"\n{'='*60}")
    print(f"  B5 Stage: representations (resume={resume})")
    print(f"{'='*60}")

    rep_dir = out / "representations"
    rep_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(out / "dataset" / fname) as f:
            dataset[split_name] = json.load(f)

    mock = config.get("mock", False)
    rep_config = config.get("representation", {})
    layers = rep_config.get("layers", ["0.25", "0.50", "0.75", "final"])
    poolings = rep_config.get("poolings", ["last_content_token", "mean_content"])

    if mock:
        # Generate synthetic hidden states
        hint_to_idx = {aid: i for i, aid in enumerate(B5_ACTION_IDS)}
        raw_dim = 128

        for split_name in ["train", "dev", "final", "final_ood"]:
            tasks = dataset[split_name]
            n = len(tasks)
            features = np.zeros((n, raw_dim), dtype=np.float32)
            for i, task in enumerate(tasks):
                hint = task.get("oracle_action_hint", B5_ACTION_IDS[0])
                hint_idx = hint_to_idx.get(hint, 0)
                features[i, hint_idx * 32:(hint_idx + 1) * 32] = 1.0 + np.random.randn(32) * 0.1
                features[i, 128 - 32:] = np.random.randn(32) * 0.5

            task_ids = np.array([t["task_id"] for t in tasks], dtype=object)
            np.savez_compressed(
                rep_dir / f"{split_name}.npz",
                features=features,
                task_ids=task_ids,
                layer="mock",
                pooling="mock",
                original_dim=raw_dim,
                transformed_dim=raw_dim,
                model_revision="mock",
                tokenizer_revision="mock",
                chat_template_hash="mock",
            )
            print(f"  {split_name}: {n} tasks, {raw_dim} dims")
    else:
        # Real hidden state capture would go here
        print("  ERROR: Real representation capture not implemented. Use --mock.")
        sys.exit(1)

    # Representation manifest
    with open(rep_dir / "representation_manifest.json", "w") as f:
        json.dump({
            "layers": layers,
            "poolings": poolings,
            "model_id": config.get("model", {}).get("model_id", ""),
            "model_revision": config.get("model", {}).get("revision", ""),
        }, f, indent=2)

    print(f"  Representations complete.")


# ──────────────────────────────────────────────────────────────────────
# Stage: train
# ──────────────────────────────────────────────────────────────────────

def stage_train(config: dict) -> None:
    """Train policies, surface baselines, shams."""
    out = get_output_dir(config)
    print(f"\n{'='*60}")
    print(f"  B5 Stage: train")
    print(f"{'='*60}")

    pol_dir = out / "policies"
    pol_dir.mkdir(exist_ok=True)

    # Load data
    dataset = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(out / "dataset" / fname) as f:
            dataset[split_name] = json.load(f)

    cfs = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(out / "counterfactuals" / fname) as f:
            cfs[split_name] = json.load(f)

    reps = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        rep_file = out / "representations" / f"{split_name}.npz"
        if rep_file.exists():
            reps[split_name] = np.load(rep_file, allow_pickle=True)

    action_ids = list(B5_ACTION_IDS)

    # Build utility matrices
    def build_utils(split_name):
        tasks = dataset[split_name]
        n = len(tasks)
        U = np.zeros((n, len(action_ids)), dtype=np.float32)
        for i, task in enumerate(tasks):
            tid = task["task_id"]
            for j, aid in enumerate(action_ids):
                U[i, j] = cfs[split_name].get(tid, {}).get(aid, {}).get("utility", 0.0)
        return U

    U_train = build_utils("train")
    U_dev = build_utils("dev")
    U_final = build_utils("final")

    # Train hidden policies
    train_features = reps["train"]["features"]
    dev_features = reps["dev"]["features"]
    final_features = reps["final"]["features"]

    # PCA
    from daph_learning.executive.dim_reduction import PCAPipeline
    pca = PCAPipeline()
    pca_dim = config.get("representation", {}).get("pca_dim", 64)
    train_reduced = pca.fit_transform(train_features, pca_dim)
    dev_reduced = pca.transform(dev_features)
    final_reduced = pca.transform(final_features)

    # Save PCA
    pca_dir = out / "pca"
    pca_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        pca_dir / "pca_artifact.npz",
        mean=pca.mean, scale=pca.std, components=pca.components,
        explained_variance=pca.explained_variance_ratio,
        input_dim=train_features.shape[1], output_dim=pca_dim,
    )
    with open(pca_dir / "pca_manifest.json", "w") as f:
        json.dump({"fit_split": "train", "input_dim": train_features.shape[1],
                    "output_dim": pca_dim}, f, indent=2)

    # Linear Q policy
    linear = LinearQPolicy(action_ids=action_ids, n_iter=1000, l2=0.001)
    linear.fit(train_reduced, U_train)
    linear.save(str(pol_dir / "hidden_policy.json"))

    # Ridge Q policy
    ridge = RidgeQPolicy(action_ids=action_ids, alpha=1.0)
    ridge.fit(train_reduced, U_train)
    ridge.save(str(pol_dir / "ridge_policy.json"))

    # MLP Q policy
    mlp = MLPQPolicy(action_ids=action_ids, hidden1=128, hidden2=64, dropout=0.1)
    mlp.fit(train_reduced, U_train, epochs=200, lr=0.001)
    mlp.save(str(pol_dir / "mlp_policy.json"))

    # Surface baselines
    surface_extractor = SurfaceFeatureExtractor(
        feature_types=("subtype", "prompt_length", "tfidf")
    )
    train_surface = surface_extractor.fit_transform(dataset["train"])
    final_surface = surface_extractor.transform(dataset["final"])

    surface_ensemble = SurfaceEnsemblePolicy(action_ids=action_ids, alpha=1.0)
    surface_ensemble.fit(train_surface, U_train)

    with open(pol_dir / "surface_baselines.json", "w") as f:
        json.dump({
            "policy_type": "surface_ensemble",
            "features": ["subtype", "prompt_length", "tfidf"],
            "surface_ensemble_utility": float(
                U_final[np.arange(len(U_final)), surface_ensemble.predict(final_surface)].mean()
            ),
        }, f, indent=2)

    # Fixed baselines
    fixed_utils = U_train.mean(axis=0)
    best_fixed_idx = int(np.argmax(fixed_utils))
    with open(pol_dir / "fixed_baselines.json", "w") as f:
        json.dump({
            "action_ids": action_ids,
            "mean_utilities": fixed_utils.tolist(),
            "best_fixed_action": action_ids[best_fixed_idx],
            "best_fixed_idx": best_fixed_idx,
        }, f, indent=2)

    # Policy manifest
    with open(pol_dir / "policy_manifest.json", "w") as f:
        json.dump({
            "policies": ["hidden_policy", "ridge_policy", "mlp_policy",
                         "fixed_baselines", "surface_baselines"],
        }, f, indent=2)

    # Shams
    sham_dir = out / "sham"
    sham_dir.mkdir(exist_ok=True)
    n_shams = config.get("qualification", {}).get("n_shams", 50)

    train_subtypes = [t.get("subtype", "unknown") for t in dataset["train"]]
    final_subtypes = [t.get("subtype", "unknown") for t in dataset["final"]]
    final_groups = [t.get("group_id", "g0") for t in dataset["final"]]
    train_split_ids = np.zeros(len(dataset["train"]), dtype=int)
    final_split_ids = np.full(len(dataset["final"]), 2, dtype=int)

    hidden_preds = linear.predict(final_reduced)
    sham_result = run_matched_sham_evaluation(
        train_features=train_reduced,
        train_utilities=U_train,
        test_features=final_reduced,
        test_utilities=U_final,
        test_group_ids=final_groups,
        train_subtypes=train_subtypes,
        test_subtypes=final_subtypes,
        train_split_ids=train_split_ids,
        test_split_ids=final_split_ids,
        real_hidden_predictions=hidden_preds,
        policy_cls=LinearQPolicy,
        policy_kwargs={"action_ids": action_ids, "n_iter": 500, "l2": 0.01},
        n_shams=n_shams,
    )

    with open(sham_dir / "sham_runs.json", "w") as f:
        json.dump(sham_result.to_dict(), f, indent=2)

    print(f"  Policies trained: linear, ridge, MLP, surface_ensemble")
    print(f"  Shams: {n_shams}")
    print(f"  Best fixed: {action_ids[best_fixed_idx]}")


# ──────────────────────────────────────────────────────────────────────
# Stage: qualify
# ──────────────────────────────────────────────────────────────────────

def stage_qualify(config: dict) -> None:
    """Evaluate FINAL once, run gates, generate report."""
    out = get_output_dir(config)
    experiment_id = config.get("experiment_id", "executive_b5_adaptive_compute")
    action_ids = list(B5_ACTION_IDS)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: qualify")
    print(f"{'='*60}")

    # Load everything
    dataset = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(out / "dataset" / fname) as f:
            dataset[split_name] = json.load(f)

    cfs = {}
    for split_name in ["train", "dev", "final", "final_ood"]:
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
        with open(out / "counterfactuals" / fname) as f:
            cfs[split_name] = json.load(f)

    # Build utility matrices
    def build_utils(split_name):
        tasks = dataset[split_name]
        n = len(tasks)
        U = np.zeros((n, len(action_ids)), dtype=np.float32)
        for i, task in enumerate(tasks):
            tid = task["task_id"]
            for j, aid in enumerate(action_ids):
                U[i, j] = cfs[split_name].get(tid, {}).get(aid, {}).get("utility", 0.0)
        return U

    U_train = build_utils("train")
    U_final = build_utils("final")

    final_groups = [t.get("group_id", "g0") for t in dataset["final"]]
    final_subtypes = [t.get("subtype", "unknown") for t in dataset["final"]]
    final_families = [t.get("family", "unknown") for t in dataset["final"]]
    final_difficulties = [t.get("difficulty", "medium") for t in dataset["final"]]
    final_prompt_lengths = [len(t.get("prompt", "")) for t in dataset["final"]]

    # Load policies
    from daph_learning.executive.dim_reduction import PCAPipeline
    pca = PCAPipeline()
    pca_dir = out / "pca"
    pca_data = np.load(pca_dir / "pca_artifact.npz", allow_pickle=True)
    pca.mean = pca_data["mean"]
    pca.std = pca_data["scale"]
    pca.components = pca_data["components"]

    reps = {}
    for split_name in ["train", "dev", "final"]:
        reps[split_name] = np.load(out / "representations" / f"{split_name}.npz", allow_pickle=True)

    train_reduced = pca.transform(reps["train"]["features"])
    final_reduced = pca.transform(reps["final"]["features"])

    linear = LinearQPolicy.load(str(out / "policies" / "hidden_policy.json"))
    hidden_preds = linear.predict(final_reduced)
    hidden_utils = U_final[np.arange(len(U_final)), hidden_preds]

    # Fixed baselines
    fixed_data = json.loads((out / "policies" / "fixed_baselines.json").read_text())
    best_fixed_idx = fixed_data["best_fixed_idx"]
    best_fixed_action = fixed_data["best_fixed_action"]
    best_fixed_utils = U_final[:, best_fixed_idx]

    # Surface ensemble
    surface_extractor = SurfaceFeatureExtractor(
        feature_types=("subtype", "prompt_length", "tfidf")
    )
    train_surface = surface_extractor.fit_transform(dataset["train"])
    final_surface = surface_extractor.transform(dataset["final"])
    surface_ensemble = SurfaceEnsemblePolicy(action_ids=action_ids, alpha=1.0)
    surface_ensemble.fit(train_surface, U_train)
    surface_preds = surface_ensemble.predict(final_surface)
    surface_utils = U_final[np.arange(len(U_final)), surface_preds]

    # Oracle
    oracle_utils = U_final.max(axis=1)
    oracle_actions = U_final.argmax(axis=1)

    # Canonical comparisons
    comp_hf = make_paired_comparison(
        "hidden", "best_fixed", hidden_utils, best_fixed_utils, final_groups,
        subtypes=final_subtypes,
        n_replicates=config.get("qualification", {}).get("bootstrap_iterations", 10000),
    )
    comp_hs = make_paired_comparison(
        "hidden", "surface_ensemble", hidden_utils, surface_utils, final_groups,
        subtypes=final_subtypes,
        n_replicates=config.get("qualification", {}).get("bootstrap_iterations", 10000),
    )

    # Sham
    sham_data = json.loads((out / "sham" / "sham_runs.json").read_text())

    # Gap capture
    gc = gap_capture(hidden_utils.mean(), best_fixed_utils.mean(), oracle_utils.mean())

    # Selection accuracy
    sel_acc = selection_accuracy(hidden_preds, oracle_actions)

    # B5 diagnostics
    crossover = empirical_crossover_analysis(
        U_final, final_families, action_ids
    )
    think_fast = think_fast_delta_analysis(
        U_final, final_families, final_difficulties, final_prompt_lengths,
        fast_idx=action_ids.index("action.reasoning.direct_fast"),
        think_idx=action_ids.index("action.reasoning.direct_think"),
        decompose_idx=action_ids.index("action.reasoning.decompose"),
        retrieve_idx=action_ids.index("action.retrieval.examples"),
    )
    margin = action_advantage_margin(U_final)

    # Save crossover report
    with open(out / "dataset" / "crossover_report.json", "w") as f:
        json.dump(crossover, f, indent=2)

    # Evaluate gates
    leakage_report = run_leakage_checks_from_artifacts(out, experiment_id=experiment_id)
    integrity_report = validate_required_tree(out, B5_REQUIRED_ARTIFACTS, experiment_id=experiment_id)

    qual_result = evaluate_gates(
        experiment_id=experiment_id,
        boot_hidden_vs_bestfixed=paired_group_bootstrap(
            hidden_utils, best_fixed_utils, final_groups,
            comparison="hidden_vs_bestfixed",
        ),
        boot_hidden_vs_surface=paired_group_bootstrap(
            hidden_utils, surface_utils, final_groups,
            comparison="hidden_vs_surface",
        ),
        sham_result=sham_data,
        group_results=compute_group_local_results(
            task_ids=[t["task_id"] for t in dataset["final"]],
            group_ids=final_groups,
            subtypes=final_subtypes,
            hidden_utilities=hidden_utils.tolist(),
            baseline_utilities=best_fixed_utils.tolist(),
        ),
        hidden_utility=hidden_utils.mean(),
        best_fixed_utility=best_fixed_utils.mean(),
        oracle_utility=oracle_utils.mean(),
        selection_accuracy=sel_acc,
        leakage_passed=leakage_report.passed,
        integrity_passed=integrity_report.passed,
    )

    # Authoritative qualification.json
    qualification = {
        "experiment_id": experiment_id,
        "status": "QUALIFIED" if qual_result.overall_passed else "FAILED_QUALIFICATION",
        "primary_policy": "hidden",
        "best_fixed_policy": best_fixed_action,
        "best_surface_policy": "surface_ensemble",
        "oracle": {"utility": float(oracle_utils.mean())},
        "policies": {
            "hidden": {"utility": float(hidden_utils.mean())},
            "best_fixed": {"name": best_fixed_action, "utility": float(best_fixed_utils.mean())},
            "surface_ensemble": {"utility": float(surface_utils.mean())},
        },
        "comparisons": {
            "hidden_vs_best_fixed": comp_hf.to_dict(),
            "hidden_vs_surface_ensemble": comp_hs.to_dict(),
        },
        "groups": {
            "positive_group_fraction": comp_hf.positive_group_fraction,
            "worst_group_delta": comp_hf.worst_group_delta,
            "group_count": comp_hf.group_count,
        },
        "gates": {g.gate_name: {"passed": g.passed, "value": g.value, "detail": g.detail}
                   for g in qual_result.gates},
        "integrity": {"passed": integrity_report.passed},
        "leakage": {"passed": leakage_report.passed},
        "diagnostics": {
            "crossover": crossover,
            "think_fast_delta": think_fast,
            "action_advantage_margin": margin,
        },
        "n_train": len(dataset["train"]),
        "n_final": len(dataset["final"]),
        "action_ids": action_ids,
    }

    qual_dir = out / "qualification"
    qual_dir.mkdir(exist_ok=True)
    with open(qual_dir / "qualification.json", "w") as f:
        json.dump(qualification, f, indent=2)

    # Update state
    state = ExperimentState(experiment_id=experiment_id)
    state.start_final(FrozenConfig(config=config))
    if qual_result.overall_passed:
        state.mark_qualified()
    else:
        state.mark_failed("qualification")
    state.save(out / "status.json")

    print(f"  Status: {qualification['status']}")
    print(f"  Hidden utility: {hidden_utils.mean():.4f}")
    print(f"  Best fixed utility: {best_fixed_utils.mean():.4f}")
    print(f"  Surface ensemble utility: {surface_utils.mean():.4f}")
    print(f"  Oracle utility: {oracle_utils.mean():.4f}")
    print(f"  Gap capture: {gc:.1%}")
    print(f"  Failed gates: {qual_result.failed_gates}")


# ──────────────────────────────────────────────────────────────────────
# Stage: reproduce
# ──────────────────────────────────────────────────────────────────────

def stage_reproduce(config: dict) -> None:
    """Verify reproduction."""
    out = get_output_dir(config)
    print(f"\n{'='*60}")
    print(f"  B5 Stage: reproduce")
    print(f"{'='*60}")

    from daph_learning.executive.reproduce import reproduce
    result = reproduce(out)
    if result.get("passed"):
        print("  Reproduction: PASS")
    else:
        print(f"  Reproduction: FAIL — {result.get('errors', [])}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B5 Staged Runner")
    parser.add_argument("stage", choices=["prepare", "counterfactuals",
                                           "representations", "train",
                                           "qualify", "reproduce", "all"])
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing records")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock execution (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mock:
        config["mock"] = True

    if args.stage == "all":
        stage_prepare(config)
        stage_counterfactuals(config, resume=args.resume)
        stage_representations(config, resume=args.resume)
        stage_train(config)
        stage_qualify(config)
        stage_reproduce(config)
    else:
        func = globals().get(f"stage_{args.stage}")
        if func is None:
            print(f"Unknown stage: {args.stage}")
            sys.exit(1)
        if args.stage in ("counterfactuals", "representations"):
            func(config, resume=args.resume)
        else:
            func(config)


if __name__ == "__main__":
    main()
