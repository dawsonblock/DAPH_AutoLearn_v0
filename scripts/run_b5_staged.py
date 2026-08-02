#!/usr/bin/env python3
"""DAPH v0.4.0a3 — B5 Staged Runner: Adaptive Inference Compute.

This is the canonical real-model experiment entrypoint for B5.

Required stages (in order):
  prepare                    — freeze config, generate dataset, run split checks
  development-counterfactuals — execute actions on TRAIN + DEV only
  development-representations — capture hidden states for TRAIN + DEV only
  train                      — train policies, surface baselines, shams on TRAIN/DEV
  freeze-policy              — select best policy on DEV, freeze policy manifest
  final-counterfactuals      — execute actions on FINAL + FINAL_OOD (requires FROZEN policy)
  final-representations      — capture hidden states for FINAL + FINAL_OOD
  qualify                    — evaluate FINAL, run gates, generate report
  reproduce                  — verify offline reproduction

Usage:
  python scripts/run_b5_staged.py prepare --config configs/executive_b5.yaml
  python scripts/run_b5_staged.py development-counterfactuals --resume
  python scripts/run_b5_staged.py development-representations --resume
  python scripts/run_b5_staged.py train
  python scripts/run_b5_staged.py freeze-policy
  python scripts/run_b5_staged.py final-counterfactuals --resume
  python scripts/run_b5_staged.py final-representations --resume
  python scripts/run_b5_staged.py qualify
  python scripts/run_b5_staged.py reproduce

  python scripts/run_b5_staged.py all --mock --config configs/executive_b5_mock.yaml

Lifecycle:
  DEVELOPMENT → FROZEN → TRAIN_RUNNING → TRAIN_COMPLETE
      → FINAL_RUNNING → QUALIFIED / FAILED_*
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

from daph_learning.executive.b5_actions import (
    B5_ACTION_IDS, b5_action_space,
    DirectFastExecutor, DirectThinkExecutor,
    B5ExecutorRegistry, build_b5_executors,
    B5_DEFAULT_PRESETS,
)
from daph_learning.executive.b5_dataset import generate_b5_dataset, build_b5_retrieval_store
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
    FinalAccessViolation,
)
from daph_learning.executive.final_access import (
    FinalAccessGuard, check_final_isolation,
)
from daph_learning.executive.atomic_io import (
    atomic_write_json, atomic_write_text, atomic_write_npz,
    atomic_write_jsonl, append_jsonl,
)
from daph_learning.executive.manifest import ManifestBuilder, compute_config_hash
from daph_learning.executive.artifact_integrity import (
    validate_required_tree, B5_REQUIRED_ARTIFACTS,
)


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

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


def _load_state(config: dict) -> ExperimentState:
    """Load persisted experiment state from status.json."""
    out = get_output_dir(config)
    status_path = out / "status.json"
    if not status_path.exists():
        print(f"  ERROR: No status.json found at {status_path}. Run 'prepare' first.")
        sys.exit(1)
    return ExperimentState.load(status_path)


def _save_state(config: dict, state: ExperimentState) -> None:
    """Save experiment state to status.json."""
    out = get_output_dir(config)
    state.save(out / "status.json")


def _verify_config_hash(config: dict) -> str:
    """Load frozen config hash and verify current config matches.

    Uses the same partial-config hash as FrozenConfig (only FROZEN_CONFIG_FIELDS).
    The 'mock' flag is a runtime override and is excluded.
    """
    out = get_output_dir(config)
    config_hash_file = out / "config" / "config_hash.txt"
    if not config_hash_file.exists():
        print("  ERROR: Config not frozen. Run 'prepare' first.")
        sys.exit(1)
    frozen_hash = config_hash_file.read_text().strip()
    # Use FrozenConfig to compute the same partial hash
    compare_config = {k: v for k, v in config.items() if k != "mock"}
    current_frozen = FrozenConfig(config=compare_config)
    current_hash = current_frozen.config_hash
    if frozen_hash != current_hash:
        print(
            f"  ERROR: Config hash mismatch!\n"
            f"    frozen:  {frozen_hash[:16]}...\n"
            f"    current: {current_hash[:16]}...\n"
            f"  Create a new experiment ID instead of changing config."
        )
        sys.exit(1)
    return frozen_hash


def _load_dataset_split(out: Path, split_name: str) -> list[dict]:
    """Load a single dataset split."""
    fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
    with open(out / "dataset" / fname) as f:
        return json.load(f)


def _load_counterfactuals(out: Path, split_name: str) -> dict:
    """Load counterfactuals for a split."""
    fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"
    cf_file = out / "counterfactuals" / fname
    if not cf_file.exists():
        return {}
    with open(cf_file) as f:
        return json.load(f)


def _build_utility_matrix(
    tasks: list[dict], cfs: dict, action_ids: list[str],
) -> np.ndarray:
    """Build utility matrix [n_tasks, n_actions] from counterfactuals."""
    n = len(tasks)
    U = np.zeros((n, len(action_ids)), dtype=np.float32)
    for i, task in enumerate(tasks):
        tid = task["task_id"]
        for j, aid in enumerate(action_ids):
            U[i, j] = cfs.get(tid, {}).get(aid, {}).get("utility", 0.0)
    return U


# ──────────────────────────────────────────────────────────────────────
# Mock execution
# ──────────────────────────────────────────────────────────────────────

def _mock_execute(task: dict, action_id: str, config: dict) -> tuple[ExecutionStatus, ObservedCost]:
    """Mock execution that produces realistic-looking outcomes.

    The mock signal is: the oracle action hint gets higher correctness probability.
    This is the SAME orchestration path as real execution — only the backend is mocked.
    """
    hint = task.get("oracle_action_hint", "")
    difficulty = task.get("difficulty", "medium")
    base_prob = {"easy": 0.6, "medium": 0.3, "hard": 0.1}[difficulty]
    if action_id == hint:
        prob = {"easy": 0.95, "medium": 0.85, "hard": 0.75}[difficulty]
    else:
        prob = base_prob
    rng = np.random.RandomState(int(hashlib.sha256((task["task_id"] + action_id).encode()).hexdigest(), 16) % (2**32))
    correct = rng.random() < prob
    status = ExecutionStatus.CORRECT if correct else ExecutionStatus.INCORRECT

    # Mock cost varies by action
    if "direct_fast" in action_id:
        cost = ObservedCost(
            prompt_tokens=100, completion_tokens=int(rng.randint(50, 200)),
            llm_call_count=1, wall_latency_ms=float(rng.uniform(200, 1000)),
        )
    elif "direct_think" in action_id:
        cost = ObservedCost(
            prompt_tokens=100, completion_tokens=int(rng.randint(200, 800)),
            reasoning_tokens=int(rng.randint(100, 500)),
            llm_call_count=1, wall_latency_ms=float(rng.uniform(1000, 5000)),
        )
    elif "retrieval" in action_id:
        cost = ObservedCost(
            prompt_tokens=300, completion_tokens=int(rng.randint(50, 300)),
            llm_call_count=1, retrieval_calls=1,
            wall_latency_ms=float(rng.uniform(500, 2000)),
        )
    else:  # decompose
        cost = ObservedCost(
            prompt_tokens=200, completion_tokens=int(rng.randint(200, 600)),
            llm_call_count=int(rng.randint(2, 5)),
            wall_latency_ms=float(rng.uniform(2000, 8000)),
        )
    return status, cost


def _real_execute(task: dict, action_id: str, config: dict,
                  registry: B5ExecutorRegistry) -> tuple[ExecutionStatus, ObservedCost]:
    """Real execution using the B5 executor registry."""
    executor = registry.get(action_id)
    if executor is None:
        return ExecutionStatus.EXECUTION_ERROR, ObservedCost(
            llm_call_count=0, wall_latency_ms=0.0,
        )
    t0 = time.time()
    result = executor.execute(task)
    latency_ms = (time.time() - t0) * 1000.0

    if result.failure_type == "execution_error":
        status = ExecutionStatus.EXECUTION_ERROR
    elif result.failure_type == "parse_error":
        status = ExecutionStatus.INVALID_OUTPUT
    elif result.verified_correct:
        status = ExecutionStatus.CORRECT
    else:
        status = ExecutionStatus.INCORRECT

    cost = ObservedCost(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        llm_call_count=1,
        wall_latency_ms=latency_ms,
    )
    return status, cost


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

    # Freeze config (exclude runtime-only 'mock' flag from frozen hash)
    frozen_config = {k: v for k, v in config.items() if k != "mock"}
    frozen = FrozenConfig(config=frozen_config)
    config_dir = out / "config"
    config_dir.mkdir(exist_ok=True)
    atomic_write_json(config_dir / "experiment_config.json", config)
    atomic_write_json(config_dir / "frozen_config.json", frozen.to_dict())
    atomic_write_text(config_dir / "config_hash.txt", frozen.config_hash)

    # Experiment state — freeze from DEVELOPMENT
    state = ExperimentState(experiment_id=experiment_id)
    state.freeze(frozen)
    _save_state(config, state)

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
        atomic_write_json(ds_dir / fname, split.tasks)

    all_groups = {}
    for name, split in dataset.items():
        for g in split.groups:
            all_groups[g] = name
    atomic_write_json(ds_dir / "groups.json", all_groups)

    atomic_write_json(ds_dir / "dataset_manifest.json", {
        "n_train": len(dataset["train"].tasks),
        "n_dev": len(dataset["dev"].tasks),
        "n_final": len(dataset["final"].tasks),
        "n_ood": len(dataset["final_ood"].tasks),
        "split_mode": ds_config.get("split_mode", "standard"),
    })

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
    atomic_write_json(checks_dir / "leakage.json", {
        "passed": all_pass,
        "checks": [c.to_dict() if hasattr(c, "to_dict") else c.__dict__ for c in checks],
    })

    if not all_pass:
        print("  ERROR: Leakage checks failed!")
        state = _load_state(config)
        state.mark_failed("leakage")
        _save_state(config, state)
        sys.exit(1)

    # Provenance
    prov_dir = out / "provenance"
    prov_dir.mkdir(exist_ok=True)
    atomic_write_json(prov_dir / "source.json", {
        "package_version": "0.4.0a3",
        "config_hash": frozen.config_hash,
    })
    atomic_write_json(prov_dir / "model.json", config.get("model", {}))
    atomic_write_json(prov_dir / "environment.json", {
        "python_version": sys.version,
        "platform": sys.platform,
    })

    print(f"  Dataset: {len(dataset['train'].tasks)} train, "
          f"{len(dataset['dev'].tasks)} dev, "
          f"{len(dataset['final'].tasks)} final, "
          f"{len(dataset['final_ood'].tasks)} OOD")
    print(f"  Leakage checks: PASS")
    print(f"  Config hash: {frozen.config_hash[:16]}...")
    print(f"  Status: FROZEN")


# ──────────────────────────────────────────────────────────────────────
# Stage: development-counterfactuals
# ──────────────────────────────────────────────────────────────────────

def stage_development_counterfactuals(config: dict, resume: bool = False) -> None:
    """Execute all actions on TRAIN + DEV only."""
    _run_counterfactuals(config, splits=["train", "dev"],
                         stage_name="development-counterfactuals", resume=resume)


def stage_final_counterfactuals(config: dict, resume: bool = False) -> None:
    """Execute all actions on FINAL + FINAL_OOD.

    Requires TRAIN_COMPLETE state (policy must be frozen).
    """
    state = _load_state(config)
    if state.status not in (ExperimentStatus.TRAIN_COMPLETE, ExperimentStatus.FINAL_RUNNING):
        print(f"  ERROR: final-counterfactuals requires TRAIN_COMPLETE, "
              f"current={state.status.value}")
        sys.exit(1)
    # Transition to FINAL_RUNNING
    if state.status == ExperimentStatus.TRAIN_COMPLETE:
        state.start_final(config)
        _save_state(config, state)

    _run_counterfactuals(config, splits=["final", "final_ood"],
                         stage_name="final-counterfactuals", resume=resume)


def _run_counterfactuals(config: dict, splits: list[str],
                         stage_name: str, resume: bool = False) -> None:
    """Execute counterfactuals for the given splits."""
    out = get_output_dir(config)
    action_ids = list(B5_ACTION_IDS)
    frozen_hash = _verify_config_hash(config)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: {stage_name} (resume={resume})")
    print(f"{'='*60}")

    # Load state for final access guard
    state = _load_state(config)
    guard = FinalAccessGuard(state, out / "checks" / "final_access_ledger.jsonl")

    cf_dir = out / "counterfactuals"
    cf_dir.mkdir(exist_ok=True)

    # Build executor registry for real mode
    mock = config.get("mock", False)
    registry = None
    if not mock:
        from daph_learning.executive.executors import LLMGenerationConfig
        model_cfg = config.get("model", {})
        llm_config = LLMGenerationConfig(
            model_id=model_cfg.get("model_id", ""),
            vllm_port=model_cfg.get("vllm_port", 8000),
            vllm_api_key_env=model_cfg.get("vllm_api_key_env", ""),
            vllm_base_url=model_cfg.get("vllm_base_url", ""),
            vllm_max_concurrent=model_cfg.get("vllm_max_concurrent", 24),
        )
        # Build retrieval store from TRAIN only
        train_tasks = _load_dataset_split(out, "train")
        retrieval_examples = build_b5_retrieval_store(train_tasks)
        registry = build_b5_executors(llm_config, retrieval_examples=retrieval_examples)

    for split_name in splits:
        # Check final access guard
        if split_name in ("final", "final_ood"):
            guard.assert_can_read(
                split_name,
                artifact=f"counterfactuals/{split_name}.json",
                purpose=f"final counterfactual execution",
                stage=stage_name,
            )

        tasks = _load_dataset_split(out, split_name)
        fname = "final_ood.json" if split_name == "final_ood" else f"{split_name}.json"

        # Resume: load existing records
        existing_records = {}
        if resume:
            cf_file = cf_dir / fname
            if cf_file.exists():
                try:
                    records = json.loads(cf_file.read_text())
                    task_map = {t["task_id"]: t for t in tasks}
                    for tid, actions in records.items():
                        for aid, result in actions.items():
                            if (result.get("config_hash") == frozen_hash
                                    and tid in task_map
                                    and result.get("task_hash") == _task_hash(task_map[tid])):
                                existing_records[(tid, aid)] = result
                except (json.JSONDecodeError, KeyError):
                    pass

        results = {}
        n_new = 0
        n_reused = 0

        for task in tasks:
            tid = task["task_id"]
            task_h = _task_hash(task)
            results[tid] = {}

            for action_id in action_ids:
                key = (tid, action_id)
                if resume and key in existing_records:
                    results[tid][action_id] = existing_records[key]
                    n_reused += 1
                    continue

                action_cfg_hash = _action_config_hash(action_id, config)

                if mock:
                    status, cost = _mock_execute(task, action_id, config)
                else:
                    status, cost = _real_execute(task, action_id, config, registry)

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

        atomic_write_json(cf_dir / fname, results)
        print(f"  {split_name}: {n_new} new, {n_reused} reused, {len(results)} total")

    # Save summary
    atomic_write_json(cf_dir / "summary.json", {
        "splits": splits,
        "action_ids": action_ids,
        "config_hash": frozen_hash,
        "stage": stage_name,
    })
    print(f"  Counterfactuals complete.")


# ──────────────────────────────────────────────────────────────────────
# Stage: development-representations
# ──────────────────────────────────────────────────────────────────────

def stage_development_representations(config: dict, resume: bool = False) -> None:
    """Capture hidden states for TRAIN + DEV only."""
    _run_representations(config, splits=["train", "dev"],
                         stage_name="development-representations", resume=resume)


def stage_final_representations(config: dict, resume: bool = False) -> None:
    """Capture hidden states for FINAL + FINAL_OOD."""
    state = _load_state(config)
    if state.status != ExperimentStatus.FINAL_RUNNING:
        print(f"  ERROR: final-representations requires FINAL_RUNNING, "
              f"current={state.status.value}")
        sys.exit(1)
    _run_representations(config, splits=["final", "final_ood"],
                         stage_name="final-representations", resume=resume)


def _run_representations(config: dict, splits: list[str],
                         stage_name: str, resume: bool = False) -> None:
    """Capture hidden states for the given splits."""
    out = get_output_dir(config)
    _verify_config_hash(config)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: {stage_name} (resume={resume})")
    print(f"{'='*60}")

    state = _load_state(config)
    guard = FinalAccessGuard(state, out / "checks" / "final_access_ledger.jsonl")

    rep_dir = out / "representations"
    rep_dir.mkdir(exist_ok=True)

    mock = config.get("mock", False)
    rep_config = config.get("representation", {})
    layers = rep_config.get("layers", ["0.25", "0.50", "0.75", "final"])
    poolings = rep_config.get("poolings", ["last_content_token", "mean_content"])

    if mock:
        # Generate synthetic hidden states
        hint_to_idx = {aid: i for i, aid in enumerate(B5_ACTION_IDS)}
        raw_dim = 128

        for split_name in splits:
            if split_name in ("final", "final_ood"):
                guard.assert_can_read(
                    split_name,
                    artifact=f"representations/{split_name}.npz",
                    purpose=f"final representation capture",
                    stage=stage_name,
                )

            fname = "final_ood.npz" if split_name == "final_ood" else f"{split_name}.npz"
            rep_file = rep_dir / fname
            if resume and rep_file.exists():
                print(f"  {split_name}: cached (resume)")
                continue

            tasks = _load_dataset_split(out, split_name)
            n = len(tasks)
            features = np.zeros((n, raw_dim), dtype=np.float32)
            for i, task in enumerate(tasks):
                hint = task.get("oracle_action_hint", B5_ACTION_IDS[0])
                hint_idx = hint_to_idx.get(hint, 0)
                # Deterministic seed from task_id for reproducibility
                task_seed = int(hashlib.sha256(task["task_id"].encode()).hexdigest(), 16) % (2**32)
                task_rng = np.random.RandomState(task_seed)
                features[i, hint_idx * 32:(hint_idx + 1) * 32] = 1.0 + task_rng.randn(32) * 0.1
                features[i, 128 - 32:] = task_rng.randn(32) * 0.5

            task_ids = np.array([t["task_id"] for t in tasks], dtype=object)
            atomic_write_npz(
                rep_file,
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
        # Real hidden state capture
        from daph_learning.executive.hidden_state import (
            HiddenStateConfig, load_model_for_capture, capture_hidden_states,
        )
        model_cfg = config.get("model", {})
        model, tokenizer = load_model_for_capture(
            model_id=model_cfg.get("model_id", ""),
            revision=model_cfg.get("revision", ""),
            device=model_cfg.get("device", "cuda"),
        )
        hs_config = HiddenStateConfig(
            layers=[float(l) for l in layers],
            pooling=poolings[0] if len(poolings) == 1 else "all",
        )
        for split_name in splits:
            if split_name in ("final", "final_ood"):
                guard.assert_can_read(
                    split_name,
                    artifact=f"representations/{split_name}.npz",
                    purpose=f"final representation capture",
                    stage=stage_name,
                )
            fname = "final_ood.npz" if split_name == "final_ood" else f"{split_name}.npz"
            rep_file = rep_dir / fname
            if resume and rep_file.exists():
                print(f"  {split_name}: cached (resume)")
                continue
            tasks = _load_dataset_split(out, split_name)
            results = capture_hidden_states(
                tasks=tasks, model=model, tokenizer=tokenizer,
                config=hs_config, device=model_cfg.get("device", "cuda"),
            )
            # Flatten to single feature matrix (use first layer × first pooling)
            key = list(results.keys())[0]
            features = results[key]
            task_ids = np.array([t["task_id"] for t in tasks], dtype=object)
            atomic_write_npz(
                rep_file,
                features=features,
                task_ids=task_ids,
                layer=layers[0],
                pooling=poolings[0],
                original_dim=features.shape[1],
                transformed_dim=features.shape[1],
                model_revision=model_cfg.get("revision", ""),
                tokenizer_revision=getattr(tokenizer, "name_or_path", ""),
                chat_template_hash="",
            )
            print(f"  {split_name}: {len(tasks)} tasks, {features.shape[1]} dims")

    # Representation manifest
    atomic_write_json(rep_dir / "representation_manifest.json", {
        "layers": layers,
        "poolings": poolings,
        "model_id": config.get("model", {}).get("model_id", ""),
        "model_revision": config.get("model", {}).get("revision", ""),
    })
    print(f"  Representations complete.")


# ──────────────────────────────────────────────────────────────────────
# Stage: train
# ──────────────────────────────────────────────────────────────────────

def stage_train(config: dict) -> None:
    """Train policies, surface baselines, shams on TRAIN + DEV only.

    FINAL data is NEVER accessed during training.
    """
    out = get_output_dir(config)
    action_ids = list(B5_ACTION_IDS)
    frozen_hash = _verify_config_hash(config)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: train")
    print(f"{'='*60}")

    # Load state and transition to TRAIN_RUNNING
    state = _load_state(config)
    state.start_training()
    _save_state(config, state)

    # During training, only TRAIN and DEV data are loaded.
    # The state is TRAIN_RUNNING which is NOT final-accessible,
    # so any accidental FINAL access would be caught by the guard.
    # Load ONLY train and dev data — NO FINAL
    train_tasks = _load_dataset_split(out, "train")
    dev_tasks = _load_dataset_split(out, "dev")

    # Load counterfactuals for train and dev only
    train_cfs = _load_counterfactuals(out, "train")
    dev_cfs = _load_counterfactuals(out, "dev")

    # Load representations for train and dev only
    train_rep = np.load(out / "representations" / "train.npz", allow_pickle=True)
    dev_rep = np.load(out / "representations" / "dev.npz", allow_pickle=True)

    # Build utility matrices (TRAIN + DEV only)
    U_train = _build_utility_matrix(train_tasks, train_cfs, action_ids)
    U_dev = _build_utility_matrix(dev_tasks, dev_cfs, action_ids)

    # Verify representation task IDs match dataset
    train_task_ids = set(t["task_id"] for t in train_tasks)
    rep_task_ids = set(str(t) for t in train_rep["task_ids"])
    assert train_task_ids == rep_task_ids, "representation task IDs don't match dataset"
    dev_task_ids = set(t["task_id"] for t in dev_tasks)
    dev_rep_ids = set(str(t) for t in dev_rep["task_ids"])
    assert dev_task_ids == dev_rep_ids, "dev representation task IDs don't match dataset"

    train_features = train_rep["features"]
    dev_features = dev_rep["features"]

    # PCA — fit on TRAIN only
    from daph_learning.executive.dim_reduction import PCAPipeline
    pca = PCAPipeline()
    pca_dim = config.get("representation", {}).get("pca_dim", 64)
    train_reduced = pca.fit_transform(train_features, pca_dim)
    dev_reduced = pca.transform(dev_features)

    # Save PCA (transforms directory)
    pca_dir = out / "transforms"
    pca_dir.mkdir(exist_ok=True)
    atomic_write_npz(
        pca_dir / "pca_artifact.npz",
        mean=pca.mean, scale=pca.std, components=pca.components,
        explained_variance=pca.explained_variance_ratio,
        input_dim=train_features.shape[1], output_dim=pca_dim,
    )
    atomic_write_json(pca_dir / "pca_manifest.json", {
        "fit_split": "train",
        "input_dim": int(train_features.shape[1]),
        "output_dim": int(pca_dim),
        "fit_task_count": len(train_tasks),
    })

    pol_dir = out / "policies"
    pol_dir.mkdir(exist_ok=True)

    # Train hidden policies on TRAIN only
    linear = LinearQPolicy(action_ids=action_ids, n_iter=1000, l2=0.001)
    linear.fit(train_reduced, U_train)
    linear.save(str(pol_dir / "hidden_policy.json"))

    ridge = RidgeQPolicy(action_ids=action_ids, alpha=1.0)
    ridge.fit(train_reduced, U_train)
    ridge.save(str(pol_dir / "ridge_policy.json"))

    mlp = MLPQPolicy(action_ids=action_ids, hidden1=128, hidden2=64, dropout=0.1, n_iter=200, learning_rate=0.001)
    mlp.fit(train_reduced, U_train)
    mlp.save(str(pol_dir / "mlp_policy.json"))

    # Surface baseline — fit on TRAIN, select on DEV
    surface_extractor = SurfaceFeatureExtractor(
        feature_types=("subtype", "prompt_length", "tfidf")
    )
    train_surface = surface_extractor.fit_transform(train_tasks)
    dev_surface = surface_extractor.transform(dev_tasks)

    surface_ensemble = SurfaceEnsemblePolicy(action_ids=action_ids, alpha=1.0)
    surface_ensemble.fit(train_surface, U_train)
    surface_ensemble.save(str(pol_dir / "surface_baselines.json"))

    # Fixed baselines — from TRAIN only
    fixed_utils = U_train.mean(axis=0)
    best_fixed_idx = int(np.argmax(fixed_utils))
    atomic_write_json(pol_dir / "fixed_baselines.json", {
        "action_ids": action_ids,
        "mean_utilities": fixed_utils.tolist(),
        "best_fixed_action": action_ids[best_fixed_idx],
        "best_fixed_idx": best_fixed_idx,
    })

    # Shams — train on TRAIN, evaluate on DEV (NOT FINAL)
    sham_dir = out / "policies" / "shams"
    sham_dir.mkdir(parents=True, exist_ok=True)
    n_shams = config.get("qualification", {}).get("n_shams", 50)

    train_subtypes = [t.get("subtype", "unknown") for t in train_tasks]
    dev_subtypes = [t.get("subtype", "unknown") for t in dev_tasks]
    dev_groups = [t.get("group_id", "g0") for t in dev_tasks]
    train_split_ids = np.zeros(len(train_tasks), dtype=int)
    dev_split_ids = np.ones(len(dev_tasks), dtype=int)

    hidden_preds_dev = linear.predict(dev_reduced)
    sham_result = run_matched_sham_evaluation(
        train_features=train_reduced,
        train_utilities=U_train,
        test_features=dev_reduced,
        test_utilities=U_dev,
        test_group_ids=dev_groups,
        train_subtypes=train_subtypes,
        test_subtypes=dev_subtypes,
        train_split_ids=train_split_ids,
        test_split_ids=dev_split_ids,
        real_hidden_predictions=hidden_preds_dev,
        policy_cls=LinearQPolicy,
        policy_kwargs={"action_ids": action_ids, "n_iter": 500, "l2": 0.01},
        n_shams=n_shams,
    )

    atomic_write_json(sham_dir / "sham_runs.json", sham_result.to_dict())

    # Policy manifest
    atomic_write_json(pol_dir / "policy_manifest.json", {
        "policies": ["hidden_policy", "ridge_policy", "mlp_policy",
                     "fixed_baselines", "surface_baselines"],
        "trained_on": "train",
        "selected_on": "dev",
    })

    print(f"  Policies trained: linear, ridge, MLP, surface_ensemble")
    print(f"  Shams: {n_shams} (trained on TRAIN, evaluated on DEV)")
    print(f"  Best fixed: {action_ids[best_fixed_idx]}")
    print(f"  FINAL data NOT accessed during training")


# ──────────────────────────────────────────────────────────────────────
# Stage: freeze-policy
# ──────────────────────────────────────────────────────────────────────

def stage_freeze_policy(config: dict) -> None:
    """Select best policy on DEV, freeze policy manifest."""
    out = get_output_dir(config)
    action_ids = list(B5_ACTION_IDS)
    frozen_hash = _verify_config_hash(config)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: freeze-policy")
    print(f"{'='*60}")

    # Load state — must be TRAIN_RUNNING
    state = _load_state(config)
    if state.status != ExperimentStatus.TRAIN_RUNNING:
        print(f"  ERROR: freeze-policy requires TRAIN_RUNNING, "
              f"current={state.status.value}")
        sys.exit(1)

    # Load DEV data only
    dev_tasks = _load_dataset_split(out, "dev")
    dev_cfs = _load_counterfactuals(out, "dev")
    U_dev = _build_utility_matrix(dev_tasks, dev_cfs, action_ids)

    # Load representations for DEV
    dev_rep = np.load(out / "representations" / "dev.npz", allow_pickle=True)
    dev_task_ids = set(t["task_id"] for t in dev_tasks)
    dev_rep_ids = set(str(t) for t in dev_rep["task_ids"])
    assert dev_task_ids == dev_rep_ids, "dev representation task IDs don't match"

    # Load PCA transform
    from daph_learning.executive.dim_reduction import PCAPipeline
    pca = PCAPipeline()
    pca_data = np.load(out / "transforms" / "pca_artifact.npz", allow_pickle=True)
    pca.mean = pca_data["mean"]
    pca.std = pca_data["scale"]
    pca.components = pca_data["components"]
    dev_reduced = pca.transform(dev_rep["features"])

    # Oracle on DEV
    oracle_dev = U_dev.max(axis=1)

    # Evaluate candidates on DEV using regret: R = U_oracle - U_policy
    candidates = {}
    for name, policy_file in [
        ("linear", "hidden_policy.json"),
        ("ridge", "ridge_policy.json"),
        ("mlp", "mlp_policy.json"),
    ]:
        policy = LinearQPolicy.load(str(out / "policies" / policy_file)) if name == "linear" \
            else RidgeQPolicy.load(str(out / "policies" / policy_file)) if name == "ridge" \
            else MLPQPolicy.load(str(out / "policies" / policy_file))
        preds = policy.predict(dev_reduced)
        utils = U_dev[np.arange(len(U_dev)), preds]
        regret = float((oracle_dev - utils).mean())
        candidates[name] = {
            "regret": regret,
            "utility": float(utils.mean()),
        }

    # Select: min DEV regret, tie-break by simplicity (linear < ridge < mlp)
    simplicity_order = {"linear": 0, "ridge": 1, "mlp": 2}
    selected = min(candidates.keys(),
                   key=lambda k: (candidates[k]["regret"], simplicity_order[k]))

    # Write selection.json
    pol_dir = out / "policies"
    atomic_write_json(pol_dir / "selection.json", {
        "candidates": candidates,
        "selection_metric": "dev_regret",
        "selected_policy": selected,
        "selected_before_final": True,
    })

    # Freeze policy manifest with hashes
    selected_file = {
        "linear": "hidden_policy.json",
        "ridge": "ridge_policy.json",
        "mlp": "mlp_policy.json",
    }[selected]
    selected_path = out / "policies" / selected_file
    selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()

    surface_path = out / "policies" / "surface_baselines.json"
    surface_hash = hashlib.sha256(surface_path.read_bytes()).hexdigest()

    pca_path = out / "transforms" / "pca_artifact.npz"
    pca_hash = hashlib.sha256(pca_path.read_bytes()).hexdigest()

    atomic_write_json(pol_dir / "frozen_policy_manifest.json", {
        "selected_policy": selected,
        "selected_policy_file": selected_file,
        "selected_policy_hash": selected_hash,
        "surface_ensemble_hash": surface_hash,
        "pca_hash": pca_hash,
        "utility_weights": config.get("utility_weights", {}),
        "action_definitions": list(action_ids),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    # Transition to TRAIN_COMPLETE
    state.complete_training()
    _save_state(config, state)

    print(f"  Candidates: {candidates}")
    print(f"  Selected: {selected} (DEV regret={candidates[selected]['regret']:.4f})")
    print(f"  Policy frozen. Status: TRAIN_COMPLETE")


# ──────────────────────────────────────────────────────────────────────
# Stage: qualify
# ──────────────────────────────────────────────────────────────────────

def stage_qualify(config: dict) -> None:
    """Evaluate FINAL once, run gates, generate report.

    Ordering: compute → persist → manifest → validate → reproduce → report.
    """
    out = get_output_dir(config)
    experiment_id = config.get("experiment_id", "executive_b5_adaptive_compute")
    action_ids = list(B5_ACTION_IDS)
    frozen_hash = _verify_config_hash(config)

    print(f"\n{'='*60}")
    print(f"  B5 Stage: qualify")
    print(f"{'='*60}")

    # Load state — must be FINAL_RUNNING
    state = _load_state(config)
    if state.status != ExperimentStatus.FINAL_RUNNING:
        print(f"  ERROR: qualify requires FINAL_RUNNING, current={state.status.value}")
        sys.exit(1)

    # Final access guard
    guard = FinalAccessGuard(state, out / "checks" / "final_access_ledger.jsonl")

    # Load FINAL data through guard
    guard.assert_can_read_final(
        artifact="dataset/final.json", purpose="primary final evaluation", stage="qualify")
    final_tasks = _load_dataset_split(out, "final")
    guard.assert_can_read_ood(
        artifact="dataset/final_ood.json", purpose="OOD evaluation", stage="qualify")
    ood_tasks = _load_dataset_split(out, "final_ood")

    guard.assert_can_read_final(
        artifact="counterfactuals/final.json", purpose="final utilities", stage="qualify")
    final_cfs = _load_counterfactuals(out, "final")
    guard.assert_can_read_ood(
        artifact="counterfactuals/final_ood.json", purpose="OOD utilities", stage="qualify")
    ood_cfs = _load_counterfactuals(out, "final_ood")

    # Load TRAIN data for fixed baselines (no guard needed)
    train_tasks = _load_dataset_split(out, "train")
    train_cfs = _load_counterfactuals(out, "train")

    U_train = _build_utility_matrix(train_tasks, train_cfs, action_ids)
    U_final = _build_utility_matrix(final_tasks, final_cfs, action_ids)
    U_ood = _build_utility_matrix(ood_tasks, ood_cfs, action_ids)

    final_groups = [t.get("group_id", "g0") for t in final_tasks]
    final_subtypes = [t.get("subtype", "unknown") for t in final_tasks]
    final_families = [t.get("family", "unknown") for t in final_tasks]
    final_difficulties = [t.get("difficulty", "medium") for t in final_tasks]
    final_prompt_lengths = [len(t.get("prompt", "")) for t in final_tasks]

    # Load frozen policy selection
    selection = json.loads((out / "policies" / "selection.json").read_text())
    selected_name = selection["selected_policy"]
    selected_file = {
        "linear": "hidden_policy.json",
        "ridge": "ridge_policy.json",
        "mlp": "mlp_policy.json",
    }[selected_name]

    # Verify frozen policy manifest hashes match the actual files
    frozen_manifest_path = out / "policies" / "frozen_policy_manifest.json"
    if frozen_manifest_path.exists():
        frozen_manifest = json.loads(frozen_manifest_path.read_text())
        selected_path = out / "policies" / selected_file
        actual_selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        if actual_selected_hash != frozen_manifest.get("selected_policy_hash"):
            print(f"  ERROR: Selected policy file hash mismatch — "
                  f"file may have been modified after freeze")
            sys.exit(1)
        # Verify PCA hash
        pca_path = out / "transforms" / "pca_artifact.npz"
        actual_pca_hash = hashlib.sha256(pca_path.read_bytes()).hexdigest()
        if actual_pca_hash != frozen_manifest.get("pca_hash"):
            print(f"  ERROR: PCA transform file hash mismatch — "
                  f"file may have been modified after freeze")
            sys.exit(1)
        # Verify surface ensemble hash
        surface_path = out / "policies" / "surface_baselines.json"
        actual_surface_hash = hashlib.sha256(surface_path.read_bytes()).hexdigest()
        if actual_surface_hash != frozen_manifest.get("surface_ensemble_hash"):
            print(f"  ERROR: Surface ensemble file hash mismatch — "
                  f"file may have been modified after freeze")
            sys.exit(1)

    # Load selected policy (NOT retrain)
    if selected_name == "linear":
        selected_policy = LinearQPolicy.load(str(out / "policies" / selected_file))
    elif selected_name == "ridge":
        selected_policy = RidgeQPolicy.load(str(out / "policies" / selected_file))
    else:
        selected_policy = MLPQPolicy.load(str(out / "policies" / selected_file))

    # Load PCA transform (NOT refit)
    from daph_learning.executive.dim_reduction import PCAPipeline
    pca = PCAPipeline()
    pca_data = np.load(out / "transforms" / "pca_artifact.npz", allow_pickle=True)
    pca.mean = pca_data["mean"]
    pca.std = pca_data["scale"]
    pca.components = pca_data["components"]

    # Load FINAL representations through guard
    guard.assert_can_read_final(
        artifact="representations/final.npz", purpose="final hidden states", stage="qualify")
    final_rep = np.load(out / "representations" / "final.npz", allow_pickle=True)
    final_reduced = pca.transform(final_rep["features"])

    # Verify task IDs match
    final_task_ids = set(t["task_id"] for t in final_tasks)
    final_rep_ids = set(str(t) for t in final_rep["task_ids"])
    assert final_task_ids == final_rep_ids, "final representation task IDs don't match"

    # Hidden policy predictions
    hidden_preds = selected_policy.predict(final_reduced)
    hidden_utils = U_final[np.arange(len(U_final)), hidden_preds]

    # Fixed baselines (loaded, not recomputed)
    fixed_data = json.loads((out / "policies" / "fixed_baselines.json").read_text())
    best_fixed_idx = fixed_data["best_fixed_idx"]
    best_fixed_action = fixed_data["best_fixed_action"]
    best_fixed_utils = U_final[:, best_fixed_idx]

    # Surface ensemble (loaded, not retrained)
    # The surface feature extractor must be fitted on TRAIN to produce
    # the same feature space. The ridge weights are loaded from the saved policy.
    surface_ensemble = SurfaceEnsemblePolicy.load(str(out / "policies" / "surface_baselines.json"))
    # Refit the feature extractor on TRAIN (this is deterministic and doesn't
    # change the frozen ridge weights)
    surface_extractor = SurfaceFeatureExtractor(
        feature_types=("subtype", "prompt_length", "tfidf")
    )
    train_surface = surface_extractor.fit_transform(train_tasks)
    final_surface = surface_extractor.transform(final_tasks)
    surface_preds = surface_ensemble.predict(final_surface)
    surface_utils = U_final[np.arange(len(U_final)), surface_preds]

    # Oracle
    oracle_utils = U_final.max(axis=1)
    oracle_actions = U_final.argmax(axis=1)

    # Canonical comparisons
    n_boot = config.get("qualification", {}).get("bootstrap_iterations", 10000)
    comp_hf = make_paired_comparison(
        "hidden", "best_fixed", hidden_utils, best_fixed_utils, final_groups,
        subtypes=final_subtypes, n_replicates=n_boot,
    )
    comp_hs = make_paired_comparison(
        "hidden", "surface_ensemble", hidden_utils, surface_utils, final_groups,
        subtypes=final_subtypes, n_replicates=n_boot,
    )

    # Bootstrap results for audit
    boot_hf = paired_group_bootstrap(
        hidden_utils, best_fixed_utils, final_groups,
        comparison="hidden_vs_bestfixed", n_replicates=n_boot,
    )
    boot_hs = paired_group_bootstrap(
        hidden_utils, surface_utils, final_groups,
        comparison="hidden_vs_surface", n_replicates=n_boot,
    )

    # Sham — load persisted (trained during development)
    sham_data = json.loads(
        (out / "policies" / "shams" / "sham_runs.json").read_text()
    )

    # Wrap sham dict as object with attributes for evaluate_gates
    class _ShamWrapper:
        def __init__(self, data: dict):
            self.hidden_vs_sham_lcb95 = data.get("hidden_vs_sham_lcb95", 0.0)
            self.prob_hidden_gt_sham = data.get("prob_hidden_gt_sham", 0.0)
            self.sham_regrets = data.get("sham_regrets", [])
            self.n_shams = data.get("n_shams", 0)
        def to_dict(self) -> dict:
            return sham_data
    sham_result_obj = _ShamWrapper(sham_data)

    # Gap capture
    gc = gap_capture(hidden_utils.mean(), best_fixed_utils.mean(), oracle_utils.mean())

    # Selection accuracy
    sel_acc = selection_accuracy(hidden_preds, oracle_actions)

    # B5 diagnostics
    crossover = empirical_crossover_analysis(U_final, final_families, action_ids)
    think_fast = think_fast_delta_analysis(
        U_final, final_families, final_difficulties, final_prompt_lengths,
        fast_idx=action_ids.index("action.reasoning.direct_fast"),
        think_idx=action_ids.index("action.reasoning.direct_think"),
        decompose_idx=action_ids.index("action.reasoning.decompose"),
        retrieve_idx=action_ids.index("action.retrieval.examples"),
    )
    margin = action_advantage_margin(U_final)

    # OOD evaluation
    guard.assert_can_read_ood(
        artifact="representations/final_ood.npz", purpose="OOD hidden states", stage="qualify")
    ood_rep = np.load(out / "representations" / "final_ood.npz", allow_pickle=True)
    ood_reduced = pca.transform(ood_rep["features"])
    ood_hidden_preds = selected_policy.predict(ood_reduced)
    ood_hidden_utils = U_ood[np.arange(len(U_ood)), ood_hidden_preds]
    ood_surface_preds = surface_ensemble.predict(surface_extractor.transform(ood_tasks))
    ood_surface_utils = U_ood[np.arange(len(U_ood)), ood_surface_preds]
    ood_fixed_utils = U_ood[:, best_fixed_idx]
    ood_oracle_utils = U_ood.max(axis=1)
    ood_gc = gap_capture(ood_hidden_utils.mean(), ood_fixed_utils.mean(),
                         ood_oracle_utils.mean())

    # Per-task results
    per_task = []
    for i, task in enumerate(final_tasks):
        per_task.append({
            "task_id": task["task_id"],
            "group_id": task.get("group_id", "g0"),
            "family": task.get("family", "unknown"),
            "oracle_action": action_ids[int(oracle_actions[i])],
            "oracle_utility": float(oracle_utils[i]),
            "hidden_action": action_ids[int(hidden_preds[i])],
            "hidden_utility": float(hidden_utils[i]),
            "surface_action": action_ids[int(surface_preds[i])],
            "surface_utility": float(surface_utils[i]),
            "fixed_utilities": {action_ids[j]: float(U_final[i, j]) for j in range(len(action_ids))},
            "action_utilities": {action_ids[j]: float(U_final[i, j]) for j in range(len(action_ids))},
            "action_advantage_margin": float(
                sorted(U_final[i], reverse=True)[0] - sorted(U_final[i], reverse=True)[1]
            ),
        })

    # Per-group results
    per_group = []
    unique_groups = sorted(set(final_groups))
    for g in unique_groups:
        mask = np.array([g == gi for gi in final_groups])
        # Compute per-group advantage margin from the group's own utilities
        group_U = U_final[mask]
        group_sorted = np.sort(group_U, axis=1)
        if group_U.shape[1] >= 2:
            group_margins = group_sorted[:, -1] - group_sorted[:, -2]
        else:
            group_margins = np.zeros(group_U.shape[0])
        per_group.append({
            "group_id": g,
            "task_count": int(mask.sum()),
            "hidden_utility": float(hidden_utils[mask].mean()),
            "surface_utility": float(surface_utils[mask].mean()),
            "best_fixed_utility": float(best_fixed_utils[mask].mean()),
            "oracle_utility": float(oracle_utils[mask].mean()),
            "hidden_vs_fixed_delta": float(hidden_utils[mask].mean() - best_fixed_utils[mask].mean()),
            "hidden_vs_surface_delta": float(hidden_utils[mask].mean() - surface_utils[mask].mean()),
            "selection_accuracy": float(
                (hidden_preds[mask] == oracle_actions[mask]).mean()
            ),
            "mean_advantage_margin": float(group_margins.mean()) if len(group_margins) > 0 else 0.0,
        })

    # OOD results
    ood_results = {
        "ood_hidden_utility": float(ood_hidden_utils.mean()),
        "ood_surface_utility": float(ood_surface_utils.mean()),
        "ood_fixed_utility": float(ood_fixed_utils.mean()),
        "ood_oracle_utility": float(ood_oracle_utils.mean()),
        "ood_gap_capture": float(ood_gc),
        "ood_n_tasks": len(ood_tasks),
    }

    # Compute budget frontier
    # Build per-policy metrics from counterfactuals
    policy_results = {}
    for j, aid in enumerate(action_ids):
        policy_results[aid] = {
            "accuracy": float((U_final[:, j] > 0).mean()),
            "utility": float(U_final[:, j].mean()),
            "latency_ms": 0.0,
            "tokens": 0.0,
            "llm_calls": 1,
            "oracle_utility": float(oracle_utils.mean()),
        }
    policy_results["hidden"] = {
        "accuracy": float((hidden_utils > 0).mean()),
        "utility": float(hidden_utils.mean()),
        "llm_calls": 1,
        "oracle_utility": float(oracle_utils.mean()),
    }
    policy_results["surface"] = {
        "accuracy": float((surface_utils > 0).mean()),
        "utility": float(surface_utils.mean()),
        "llm_calls": 1,
        "oracle_utility": float(oracle_utils.mean()),
    }
    policy_results["oracle"] = {
        "accuracy": 1.0,
        "utility": float(oracle_utils.mean()),
        "llm_calls": 0,
        "oracle_utility": float(oracle_utils.mean()),
    }
    budget_frontier = compute_budget_frontier(policy_results)

    # Crossover report
    atomic_write_json(out / "dataset" / "crossover_report.json", crossover)

    # Leakage and integrity checks
    leakage_report = run_leakage_checks_from_artifacts(out, experiment_id=experiment_id)

    # Write qualification artifacts FIRST, then validate
    qual_dir = out / "qualification"
    qual_dir.mkdir(exist_ok=True)

    # Bootstrap NPZ
    atomic_write_npz(
        qual_dir / "bootstrap_results.npz",
        hidden_minus_fixed=boot_hf.deltas if hasattr(boot_hf, 'deltas') else np.array([]),
        hidden_minus_surface=boot_hs.deltas if hasattr(boot_hs, 'deltas') else np.array([]),
        hidden_minus_sham=np.array(sham_data.get("sham_regrets", [])),
    )

    # Per-task and per-group results
    atomic_write_json(qual_dir / "per_task_results.json", per_task)
    atomic_write_json(qual_dir / "per_group_results.json", per_group)
    atomic_write_json(qual_dir / "ood_results.json", ood_results)

    # Evaluate gates
    qual_result = evaluate_gates(
        experiment_id=experiment_id,
        boot_hidden_vs_bestfixed=boot_hf,
        boot_hidden_vs_surface=boot_hs,
        sham_result=sham_result_obj,
        group_results=compute_group_local_results(
            task_ids=[t["task_id"] for t in final_tasks],
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
        integrity_passed=True,  # will validate after writing
    )

    # Authoritative qualification.json
    qualification = {
        "experiment_id": experiment_id,
        "status": "QUALIFIED" if qual_result.overall_passed else "FAILED_QUALIFICATION",
        "primary_policy": selected_name,
        "best_fixed_policy": best_fixed_action,
        "best_surface_policy": "surface_ensemble",
        "oracle": {"utility": float(oracle_utils.mean())},
        "policies": {
            "hidden": {"name": selected_name, "utility": float(hidden_utils.mean())},
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
        "integrity": {"passed": True},  # will update after validation
        "leakage": {"passed": leakage_report.passed},
        "diagnostics": {
            "crossover": crossover,
            "think_fast_delta": think_fast,
            "action_advantage_margin": margin,
            "compute_budget_frontier": budget_frontier,
        },
        "ood": ood_results,
        "n_train": len(train_tasks),
        "n_final": len(final_tasks),
        "n_ood": len(ood_tasks),
        "action_ids": action_ids,
        "selected_policy": selected_name,
        "config_hash": frozen_hash,
    }
    # Write qualification.json with integrity=False FIRST (safe default)
    qualification["integrity"]["passed"] = False
    atomic_write_json(qual_dir / "qualification.json", qualification)

    # Write checks (leakage, final_isolation, report_consistency) BEFORE integrity validation
    checks_dir = out / "checks"
    checks_dir.mkdir(exist_ok=True)
    atomic_write_json(checks_dir / "leakage.json", {
        "passed": leakage_report.passed,
        "checks": [c.to_dict() if hasattr(c, "to_dict") else c.__dict__
                    for c in leakage_report.checks],
    })

    # Final isolation check
    final_iso = check_final_isolation(out / "checks" / "final_access_ledger.jsonl")
    atomic_write_json(checks_dir / "final_isolation.json", final_iso)

    # Generate report from qualification.json (not recalculating)
    report = _generate_report(qualification)
    reports_dir = out / "reports"
    reports_dir.mkdir(exist_ok=True)
    atomic_write_text(reports_dir / "final_report.md", report)

    # Report consistency check — verify key metrics from qualification.json
    # appear in the report text
    report_consistency_passed = True
    try:
        report_text = report
        # Check that key utility values appear in the report
        key_values = [
            qualification["policies"]["hidden"]["utility"],
            qualification["policies"]["best_fixed"]["utility"],
            qualification["policies"]["surface_ensemble"]["utility"],
            qualification["oracle"]["utility"],
        ]
        for val in key_values:
            if f"{val:.4f}" not in report_text:
                report_consistency_passed = False
                break
    except (KeyError, TypeError):
        report_consistency_passed = False
    atomic_write_json(checks_dir / "report_consistency.json", {
        "passed": report_consistency_passed,
    })

    # Write placeholder reproduction.json so integrity tree validation passes
    atomic_write_json(checks_dir / "reproduction.json", {"passed": False, "pending": True})

    # Write placeholder integrity.json so integrity tree validation passes
    atomic_write_json(checks_dir / "integrity.json", {"passed": False, "pending": True})

    # Build manifest BEFORE integrity validation (it's a required artifact)
    manifest_builder = ManifestBuilder(
        experiment_id=experiment_id,
        experiment_family="executive_b5_adaptive_compute",
        artifact_root=out,
        config_hash=frozen_hash,
    )
    # Add key artifacts
    for rel_path in [
        "status.json", "config/experiment_config.json", "config/frozen_config.json",
        "dataset/train.json", "dataset/dev.json", "dataset/final.json", "dataset/final_ood.json",
        "counterfactuals/train.json", "counterfactuals/dev.json",
        "counterfactuals/final.json", "counterfactuals/final_ood.json",
        "representations/train.npz", "representations/dev.npz",
        "representations/final.npz", "representations/final_ood.npz",
        "transforms/pca_artifact.npz",
        "policies/hidden_policy.json", "policies/ridge_policy.json", "policies/mlp_policy.json",
        "policies/fixed_baselines.json", "policies/surface_baselines.json",
        "policies/selection.json", "policies/frozen_policy_manifest.json",
        "qualification/qualification.json", "qualification/per_task_results.json",
        "qualification/per_group_results.json", "qualification/ood_results.json",
        "qualification/bootstrap_results.npz",
        "checks/leakage.json", "checks/integrity.json",
        "checks/final_isolation.json", "checks/reproduction.json",
        "reports/final_report.md",
    ]:
        if (out / rel_path).exists():
            try:
                manifest_builder.add_file(rel_path)
            except (ValueError, FileNotFoundError):
                pass
    manifest = manifest_builder.build()
    atomic_write_json(out / "manifest.json", manifest)

    # NOW validate integrity (after ALL artifacts are written)
    integrity_report = validate_required_tree(out, B5_REQUIRED_ARTIFACTS, experiment_id=experiment_id)

    # Overwrite placeholder integrity.json with real result
    atomic_write_json(checks_dir / "integrity.json", integrity_report.to_dict()
                      if hasattr(integrity_report, "to_dict") else {"passed": integrity_report.passed})

    # Update qualification.json with real integrity result
    qualification["integrity"]["passed"] = integrity_report.passed
    atomic_write_json(qual_dir / "qualification.json", qualification)

    # Update state — use specific failure reason
    if qual_result.overall_passed and integrity_report.passed and leakage_report.passed:
        state.mark_qualified()
    elif not leakage_report.passed:
        state.mark_failed("leakage")
    elif not integrity_report.passed:
        state.mark_failed("integrity")
    else:
        state.mark_failed("qualification")
    _save_state(config, state)

    print(f"  Status: {qualification['status']}")
    print(f"  Selected policy: {selected_name}")
    print(f"  Hidden utility: {hidden_utils.mean():.4f}")
    print(f"  Best fixed utility: {best_fixed_utils.mean():.4f}")
    print(f"  Surface ensemble utility: {surface_utils.mean():.4f}")
    print(f"  Oracle utility: {oracle_utils.mean():.4f}")
    print(f"  Gap capture: {gc:.1%}")
    print(f"  OOD gap capture: {ood_gc:.1%}")
    print(f"  Integrity: {'PASS' if integrity_report.passed else 'FAIL'}")
    print(f"  Final isolation: {'PASS' if final_iso['passed'] else 'FAIL'}")
    print(f"  Failed gates: {qual_result.failed_gates}")


def _generate_report(qualification: dict) -> str:
    """Generate markdown report from qualification.json (no recalculation)."""
    lines = [
        f"# B5 Final Report — {qualification['experiment_id']}",
        "",
        f"**Status**: {qualification['status']}",
        f"**Selected policy**: {qualification.get('selected_policy', 'unknown')}",
        "",
        "## Utilities",
        "",
        f"| Policy | Utility |",
        f"|---|---|",
        f"| Hidden ({qualification.get('selected_policy', '')}) | {qualification['policies']['hidden']['utility']:.4f} |",
        f"| Best fixed ({qualification['policies']['best_fixed']['name']}) | {qualification['policies']['best_fixed']['utility']:.4f} |",
        f"| Surface ensemble | {qualification['policies']['surface_ensemble']['utility']:.4f} |",
        f"| Oracle | {qualification['oracle']['utility']:.4f} |",
        "",
        "## Gates",
        "",
        "| Gate | Value | Threshold | Result |",
        "|---|---|---|---|",
    ]
    for gname, gval in qualification.get("gates", {}).items():
        lines.append(f"| {gname} | {gval.get('value', 0):.4f} | | {'PASS' if gval['passed'] else 'FAIL'} |")
    lines.extend([
        "",
        "## Comparisons",
        "",
    ])
    for cname, cval in qualification.get("comparisons", {}).items():
        lines.append(f"### {cname}")
        lines.append(f"- Point delta: {cval.get('point_delta', 0):.4f}")
        lines.append(f"- LCB95: {cval.get('lcb95', 0):.4f}")
        lines.append(f"- UCB95: {cval.get('ucb95', 0):.4f}")
        lines.append(f"- P(delta > 0): {cval.get('p_positive', 0):.1%}")
        lines.append("")
    lines.extend([
        "## OOD Results",
        "",
        f"- OOD hidden utility: {qualification.get('ood', {}).get('ood_hidden_utility', 0):.4f}",
        f"- OOD gap capture: {qualification.get('ood', {}).get('ood_gap_capture', 0):.1%}",
        "",
        "## Integrity",
        "",
        f"- Integrity: {'PASS' if qualification.get('integrity', {}).get('passed') else 'FAIL'}",
        f"- Leakage: {'PASS' if qualification.get('leakage', {}).get('passed') else 'FAIL'}",
    ])
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Stage: reproduce
# ──────────────────────────────────────────────────────────────────────

def stage_reproduce(config: dict) -> None:
    """Verify offline reproduction from persisted evidence."""
    out = get_output_dir(config)
    print(f"\n{'='*60}")
    print(f"  B5 Stage: reproduce")
    print(f"{'='*60}")

    # Write a placeholder reproduction.json so the artifact tree validation
    # can find it. We'll overwrite with the real result after.
    checks_dir = out / "checks"
    checks_dir.mkdir(exist_ok=True)
    atomic_write_json(checks_dir / "reproduction.json", {"passed": False, "pending": True})

    from daph_learning.executive.reproduce import reproduce
    result = reproduce(out, experiment_family="b5")

    # Write the real result
    atomic_write_json(checks_dir / "reproduction.json", result)

    if result.get("passed"):
        print("  Reproduction: PASS")
    else:
        print(f"  Reproduction: FAIL — {result.get('errors', [])}")
        # Mark as FAILED_REPRODUCTION
        state = _load_state(config)
        if state.status == ExperimentStatus.QUALIFIED:
            state.mark_failed("reproduction")
            _save_state(config, state)
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

STAGE_ORDER = [
    "prepare",
    "development-counterfactuals",
    "development-representations",
    "train",
    "freeze-policy",
    "final-counterfactuals",
    "final-representations",
    "qualify",
    "reproduce",
]


def main():
    parser = argparse.ArgumentParser(description="B5 Staged Runner")
    parser.add_argument("stage", choices=STAGE_ORDER + ["all", "counterfactuals", "representations"])
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing records")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock execution (for testing)")
    parser.add_argument("--phase", choices=["development", "final"], default=None,
                        help="Phase for counterfactuals/representations")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mock:
        config["mock"] = True

    # Backward compat: counterfactuals/representations with --phase
    if args.stage == "counterfactuals":
        if args.phase == "final":
            stage_final_counterfactuals(config, resume=args.resume)
        else:
            stage_development_counterfactuals(config, resume=args.resume)
        return
    if args.stage == "representations":
        if args.phase == "final":
            stage_final_representations(config, resume=args.resume)
        else:
            stage_development_representations(config, resume=args.resume)
        return

    if args.stage == "all":
        stage_prepare(config)
        stage_development_counterfactuals(config, resume=args.resume)
        stage_development_representations(config, resume=args.resume)
        stage_train(config)
        stage_freeze_policy(config)
        stage_final_counterfactuals(config, resume=args.resume)
        stage_final_representations(config, resume=args.resume)
        stage_qualify(config)
        stage_reproduce(config)
    else:
        func = globals().get(f"stage_{args.stage.replace('-', '_')}")
        if func is None:
            print(f"Unknown stage: {args.stage}")
            sys.exit(1)
        if args.stage in ("development-counterfactuals", "final-counterfactuals",
                          "development-representations", "final-representations"):
            func(config, resume=args.resume)
        else:
            func(config)


if __name__ == "__main__":
    main()
