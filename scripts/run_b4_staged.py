#!/usr/bin/env python3
"""DAPH v0.4 — B4 Staged Runner: Qwen Hidden-State Executive Qualification.

Implements the staged architecture:
  Stage A: vLLM → execute all actions → save counterfactual outcomes
  Stage B: stop vLLM → load Qwen3-8B HF → capture hidden states → save NPZ
  Stage C: offline → representation sweep, PCA, ablation, sham, qualification

Usage:
  # Stage A (vLLM running)
  python scripts/run_b4_staged.py --stage A --config configs/executive_b4.yaml

  # Stage B (vLLM stopped, GPU free)
  python scripts/run_b4_staged.py --stage B --config configs/executive_b4.yaml

  # Stage C (no GPU needed)
  python scripts/run_b4_staged.py --stage C --config configs/executive_b4.yaml

  # All stages (for local testing without vLLM)
  python scripts/run_b4_staged.py --stage all --config configs/executive_b4.yaml --mock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_output_dir(config: dict) -> Path:
    """Get the output directory."""
    return Path(config.get("output_dir", "artifacts/executive_b4"))


# ──────────────────────────────────────────────────────────────────────
# Stage A: Execute all actions via vLLM, save counterfactual outcomes
# ──────────────────────────────────────────────────────────────────────

def run_stage_a(config: dict, mock: bool = False):
    """Stage A: Execute all actions on all tasks via vLLM.

    Saves:
      dataset/train_tasks.json, dev_tasks.json, final_tasks.json
      counterfactuals/train_cf.json, dev_cf.json, final_cf.json
      representations/train_logprob.npy, dev_logprob.npy, final_logprob.npy
    """
    from daph_learning.executive import (
        generate_b4_dataset,
        build_b4_retrieval_store,
        ExecutorRegistry,
        LLMGenerationConfig,
        DirectReasoningExecutor,
        RetrievalVectorExecutor,
        ReasoningDecomposeExecutor,
        ActionSpace,
        ActionDescriptor,
        capture_logprob_features,
    )

    out = get_output_dir(config)
    ds_dir = out / "dataset"
    cf_dir = out / "counterfactuals"
    rep_dir = out / "representations"
    for d in [ds_dir, cf_dir, rep_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generate dataset
    print("\n" + "="*60)
    print("  Stage A: Execute all actions via vLLM")
    print("="*60)

    data_cfg = config.get("data", {})
    dataset = generate_b4_dataset(
        n_groups=data_cfg.get("n_groups", 160),
        tasks_per_group=data_cfg.get("tasks_per_group", 8),
        seed=data_cfg.get("seed", 20260801),
        train_frac=data_cfg.get("train_frac", 0.50),
        dev_frac=data_cfg.get("dev_frac", 0.20),
    )

    print(f"\nDataset:")
    for name, split in dataset.items():
        print(f"  {name}: {split.n_groups} groups, {split.n_tasks} tasks")

    # Save tasks
    for name, split in dataset.items():
        with open(ds_dir / f"{name}_tasks.json", "w") as f:
            json.dump(split.tasks, f, indent=2)
        print(f"  Saved {name}_tasks.json")

    # 2. Build retrieval store from train tasks
    store = build_b4_retrieval_store(dataset["train"].tasks,
                                     n_per_subtype=data_cfg.get("n_store_per_subtype", 5))
    print(f"\nRetrieval store: {len(store)} examples")

    # 3. Build executors
    mc = config.get("model", {})
    llm_cfg = LLMGenerationConfig(
        model_id=mc.get("name", "Qwen/Qwen3-8B"),
        max_tokens=mc.get("max_tokens", 4096),
        temperature=mc.get("temperature", 0.0),
        vllm_port=mc.get("vllm_port", 8000),
        vllm_api_key=mc.get("vllm_api_key", os.environ.get("VLLM_API_KEY", "")),
        vllm_base_url=mc.get("vllm_base_url", ""),
        vllm_max_concurrent=mc.get("vllm_max_concurrent", 64),
    )

    # Build action space
    actions_cfg = config.get("action_space", {}).get("actions", [])
    action_descriptors = []
    for a in actions_cfg:
        action_descriptors.append(ActionDescriptor(
            action_id=a["action_id"],
            display_name=a.get("display_name", a["action_id"]),
            description=a.get("description", ""),
            cost_estimate=a.get("cost_estimate", 0.15),
            tags=a.get("tags", []),
        ))
    action_space = ActionSpace(actions=action_descriptors)
    action_ids = [a.action_id for a in action_descriptors]

    # Build executors
    direct = DirectReasoningExecutor(config=llm_cfg)
    retrieval = RetrievalVectorExecutor(config=llm_cfg, examples=store, n_retrieved=3)
    decompose = ReasoningDecomposeExecutor(config=llm_cfg)

    registry = ExecutorRegistry()
    registry.register(direct)
    registry.register(retrieval)
    registry.register(decompose)

    # 4. Execute all actions on all tasks
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def execute_task(task, split_name):
        """Execute all actions on a single task."""
        cf = registry.execute_all(task, action_space)
        return split_name, task, cf

    for split_name in ["train", "dev", "final"]:
        split = dataset[split_name]
        tasks = split.tasks
        print(f"\nExecuting {len(tasks)} {split_name} tasks × {len(action_ids)} actions...")

        cf_results = {}
        t0 = time.time()

        if mock:
            # Mock execution: random outcomes
            rng = np.random.RandomState(42)
            for task in tasks:
                utilities = {}
                for aid in action_ids:
                    correct = rng.random() > 0.5
                    latency = rng.uniform(1000, 30000)
                    utilities[aid] = {
                        "executed": True,
                        "verified_correct": bool(correct),
                        "latency_ms": float(latency),
                        "compute_cost": 0.15,
                        "failure_type": None,
                        "output_preview": f"MOCK: {'correct' if correct else 'incorrect'}",
                    }
                cf_results[task["task_id"]] = utilities
        else:
            # Real execution with concurrency
            max_workers = mc.get("vllm_max_concurrent", 16)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(execute_task, t, split_name): t for t in tasks}
                completed = 0
                for future in as_completed(futures):
                    sn, task, cf = future.result()
                    cf_results[task["task_id"]] = {
                        aid: {
                            "executed": e.executed,
                            "verified_correct": e.verified_correct,
                            "latency_ms": e.latency_ms,
                            "compute_cost": e.compute_cost,
                            "failure_type": e.failure_type,
                            "output_preview": (e.output or "")[:500] if e.output else "",
                        }
                        for aid, e in cf.executions.items()
                    }
                    completed += 1
                    if completed % 50 == 0 or completed == len(tasks):
                        elapsed = time.time() - t0
                        print(f"  [{completed}/{len(tasks)}] {elapsed:.1f}s "
                              f"({completed/elapsed:.1f} tasks/s)")

        elapsed = time.time() - t0
        print(f"  Completed {len(tasks)} in {elapsed:.1f}s")

        # Save counterfactuals
        with open(cf_dir / f"{split_name}_cf.json", "w") as f:
            json.dump(cf_results, f, indent=2)
        print(f"  Saved {split_name}_cf.json")

        # Capture logprob features (for ablation comparison)
        if not mock:
            print(f"  Capturing logprob features for {split_name}...")
            logprob_features = capture_logprob_features(
                tasks,
                vllm_base_url=f"http://localhost:{mc.get('vllm_port', 8000)}",
                vllm_api_key=mc.get("vllm_api_key", os.environ.get("VLLM_API_KEY", "")),
                model_name=mc.get("name", ""),
                batch_size=32,
            )
        else:
            logprob_features = np.random.randn(len(tasks), 10).astype(np.float32)

        np.save(rep_dir / f"{split_name}_logprob.npy", logprob_features)
        print(f"  Saved {split_name}_logprob.npy ({logprob_features.shape})")

    # 5. Save provenance
    prov_dir = out / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    model_manifest = {
        "model_name": mc.get("name", "Qwen/Qwen3-8B"),
        "revision": mc.get("revision", "unknown"),
        "vllm_port": mc.get("vllm_port", 8000),
        "max_tokens": mc.get("max_tokens", 4096),
        "temperature": mc.get("temperature", 0.0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(prov_dir / "model_manifest.json", "w") as f:
        json.dump(model_manifest, f, indent=2)

    print(f"\nStage A complete. Next: stop vLLM and run Stage B.")
    print(f"  kill the vLLM process, then:")
    print(f"  python scripts/run_b4_staged.py --stage B --config {config.get('_config_path', 'configs/executive_b4.yaml')}")


# ──────────────────────────────────────────────────────────────────────
# Stage B: Load HF model, capture hidden states
# ──────────────────────────────────────────────────────────────────────

def run_stage_b(config: dict, mock: bool = False):
    """Stage B: Load frozen Qwen3-8B, capture hidden states for all tasks.

    Saves:
      representations/train_hidden.npz, dev_hidden.npz, final_hidden.npz
      representations/representation_manifest.json
    """
    from daph_learning.executive import (
        HiddenStateConfig,
        load_model_for_capture,
        capture_hidden_states,
    )

    out = get_output_dir(config)
    ds_dir = out / "dataset"
    rep_dir = out / "representations"

    print("\n" + "="*60)
    print("  Stage B: Capture hidden states from frozen Qwen3-8B")
    print("="*60)

    # Load tasks
    all_tasks = {}
    for split_name in ["train", "dev", "final"]:
        with open(ds_dir / f"{split_name}_tasks.json") as f:
            all_tasks[split_name] = json.load(f)
        print(f"  {split_name}: {len(all_tasks[split_name])} tasks")

    mc = config.get("model", {})
    rep_cfg = config.get("representation", {})

    if mock:
        # Mock hidden states with proper key format
        hidden_dim = 128  # small for mock
        layer_indices = [9, 18, 27, 36]
        pooling_strategies = ["last_token", "mean_prompt", "mean_content"]
        for split_name, tasks in all_tasks.items():
            n = len(tasks)
            pooled_dict = {}
            for pooling in pooling_strategies:
                for li in layer_indices:
                    key = f"{pooling}/layer_{li:02d}"
                    pooled_dict[key] = np.random.randn(n, hidden_dim).astype(np.float16)
            np.savez_compressed(rep_dir / f"{split_name}_hidden.npz", **pooled_dict)
            print(f"  Saved mock {split_name}_hidden.npz ({len(pooled_dict)} representations)")

        # Save manifest
        manifest = {
            "model": {"num_hidden_layers": 36, "hidden_size": hidden_dim},
            "layers": [f"layer_{li:02d}" for li in layer_indices],
            "pooling": pooling_strategies,
        }
        with open(rep_dir / "representation_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        # Load frozen model
        hs_cfg = HiddenStateConfig(
            model_name=mc.get("name", "Qwen/Qwen3-8B"),
            revision=mc.get("revision", "b968826d9c46dd6066d109eabc6255188de91218"),
            layers=rep_cfg.get("capture_layers", [0.25, 0.5, 0.75, 1.0]),
            pooling="all",  # capture all pooling strategies
            max_length=rep_cfg.get("max_length", 512),
            batch_size=rep_cfg.get("batch_size", 4),
            dtype=mc.get("dtype", "bfloat16"),
        )

        print(f"\n  Loading {hs_cfg.model_name} (rev={hs_cfg.revision[:12]}...)...")
        model, tokenizer, config_info = load_model_for_capture(
            hs_cfg.model_name,
            revision=hs_cfg.revision,
            device=mc.get("device", "cuda"),
            dtype=mc.get("dtype", "bfloat16"),
        )
        print(f"  Model loaded: {config_info['num_hidden_layers']} layers, "
              f"hidden_size={config_info['hidden_size']}")

        # Capture hidden states for each split
        manifest = {"model": config_info, "layers": [], "pooling": []}

        for split_name, tasks in all_tasks.items():
            print(f"\n  Capturing {split_name} hidden states ({len(tasks)} tasks)...")
            raw_path = str(rep_dir / f"{split_name}_hidden_raw.npz")

            results = capture_hidden_states(
                tasks, model, tokenizer, hs_cfg,
                device=mc.get("device", "cuda"),
                save_raw_path=raw_path,
            )

            # Save pooled representations
            pooled_dict = {}
            for key, arr in results.items():
                pooled_dict[key] = arr.astype(np.float16)
                pooling, layer = key.split("/")
                if layer not in manifest["layers"]:
                    manifest["layers"].append(layer)
                if pooling not in manifest["pooling"]:
                    manifest["pooling"].append(pooling)

            np.savez_compressed(rep_dir / f"{split_name}_hidden.npz", **pooled_dict)
            print(f"  Saved {split_name}_hidden.npz ({len(pooled_dict)} representations)")

        # Save manifest
        with open(rep_dir / "representation_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  Saved representation_manifest.json")

        # Free model memory
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Model unloaded, GPU memory freed")

    print(f"\nStage B complete. Next: run Stage C (offline training).")
    print(f"  python scripts/run_b4_staged.py --stage C --config {config.get('_config_path', 'configs/executive_b4.yaml')}")


# ──────────────────────────────────────────────────────────────────────
# Stage C: Offline training, representation sweep, ablation, qualification
# ──────────────────────────────────────────────────────────────────────

def run_stage_c(config: dict, mock: bool = False):
    """Stage C: Offline policy training and evaluation.

    Runs:
      1. Representation sweep (12 arms: 4 layers × 3 pooling)
      2. PCA dimension selection
      3. Full ablation (fixed, subtype, surface, logprob, hidden, hidden+logprob, sham)
      4. Qualification with bootstrap
    """
    from daph_learning.executive import (
        QRegressionPolicy,
        PCAPipeline,
        compute_regret,
        mean_regret,
    )
    from daph_learning.executive.sham import run_sham_experiment

    out = get_output_dir(config)
    ds_dir = out / "dataset"
    cf_dir = out / "counterfactuals"
    rep_dir = out / "representations"

    print("\n" + "="*60)
    print("  Stage C: Offline training and qualification")
    print("="*60)

    # 1. Load all data
    print("\n  Loading data...")
    tasks = {}
    cfs = {}
    logprobs = {}
    hidden = {}

    for split_name in ["train", "dev", "final"]:
        with open(ds_dir / f"{split_name}_tasks.json") as f:
            tasks[split_name] = json.load(f)
        with open(cf_dir / f"{split_name}_cf.json") as f:
            cfs[split_name] = json.load(f)
        logprobs[split_name] = np.load(rep_dir / f"{split_name}_logprob.npy")
        hidden[split_name] = np.load(rep_dir / f"{split_name}_hidden.npz")

    # 2. Build utility arrays from counterfactuals
    print("  Building utility arrays...")

    # Get action IDs from config
    actions_cfg = config.get("action_space", {}).get("actions", [])
    action_ids = [a["action_id"] for a in actions_cfg]
    n_actions = len(action_ids)

    # Utility model parameters
    um = config.get("utility_model", {})
    quality_weight = um.get("quality_weight", 1.0)
    lambda_time = um.get("lambda_time", 0.005)
    lambda_compute = um.get("lambda_compute", 0.05)
    lambda_risk = um.get("lambda_risk", 1.0)
    time_ref = um.get("time_reference_ms", 15000.0)

    def compute_utilities(split_name):
        """Compute utility matrix from counterfactual outcomes."""
        split_tasks = tasks[split_name]
        split_cfs = cfs[split_name]
        n = len(split_tasks)
        U = np.zeros((n, n_actions), dtype=np.float32)
        subtypes = []
        group_ids = []

        for i, task in enumerate(split_tasks):
            tid = task["task_id"]
            subtypes.append(task.get("subtype", "unknown"))
            group_ids.append(task.get("group_id", "g0"))
            task_cfs = split_cfs[tid]
            for j, aid in enumerate(action_ids):
                e = task_cfs[aid]
                quality = 1.0 if e["verified_correct"] else 0.0
                latency = e["latency_ms"]
                cost = e["compute_cost"]
                risk = 1.0 if e.get("failure_type") else 0.0
                u = quality_weight * quality \
                    - lambda_time * (latency / time_ref) \
                    - lambda_compute * cost \
                    - lambda_risk * risk
                U[i, j] = u
        return U, subtypes, group_ids

    U_train, train_subtypes, train_groups = compute_utilities("train")
    U_dev, dev_subtypes, dev_groups = compute_utilities("dev")
    U_final, final_subtypes, final_groups = compute_utilities("final")

    print(f"  Train utilities: {U_train.shape}, mean={U_train.mean():.4f}")
    print(f"  Dev utilities:   {U_dev.shape}, mean={U_dev.mean():.4f}")
    print(f"  Final utilities: {U_final.shape}, mean={U_final.mean():.4f}")

    # Best fixed action
    fixed_utilities = U_train.mean(axis=0)
    best_fixed_idx = int(np.argmax(fixed_utilities))
    best_fixed_action = action_ids[best_fixed_idx]
    print(f"  Best fixed action: {best_fixed_action} (utility={fixed_utilities[best_fixed_idx]:.4f})")

    # 3. Representation sweep (4 layers × 3 pooling = 12 arms)
    print("\n  Representation sweep (12 arms)...")
    rep_cfg = config.get("representation", {})
    pca_dims = rep_cfg.get("pca_dimensions", [32, 64, 128, 256])

    sweep_results = {}
    best_rep = None
    best_dev_regret = np.inf

    # Get available representations from hidden states
    available_reps = [k for k in hidden["train"].files if "/" in k]
    print(f"  Available representations: {len(available_reps)}")

    for rep_key in sorted(available_reps):
        # Extract features
        train_h = hidden["train"][rep_key].astype(np.float32)
        dev_h = hidden["dev"][rep_key].astype(np.float32)
        final_h = hidden["final"][rep_key].astype(np.float32)

        # Try each PCA dimension
        for pca_dim in pca_dims:
            arm_name = f"{rep_key}/pca_{pca_dim}"

            # Fit PCA on train only
            pipeline = PCAPipeline()
            try:
                train_reduced = pipeline.fit_transform(train_h, pca_dim)
            except Exception as e:
                print(f"    {arm_name}: PCA failed ({e})")
                continue
            dev_reduced = pipeline.transform(dev_h)

            # Train Q-regression policy
            policy = QRegressionPolicy(
                action_ids=action_ids,
                learning_rate=0.01,
                n_iter=1000,
                l2=0.001,
                loss="huber",
            )
            policy.fit(train_reduced, U_train)

            # Evaluate on dev
            dev_preds = policy.predict(dev_reduced)
            dev_regret = mean_regret(dev_preds, U_dev)
            dev_oracle = U_dev.max(axis=1).mean()
            dev_policy_utility = U_dev[np.arange(len(dev_preds)), dev_preds].mean()

            sweep_results[arm_name] = {
                "dev_regret": float(dev_regret),
                "dev_policy_utility": float(dev_policy_utility),
                "dev_oracle_utility": float(dev_oracle),
                "pca_dim": pca_dim,
                "explained_var": float(pipeline.explained_variance_ratio.sum()),
            }

            if dev_regret < best_dev_regret:
                best_dev_regret = dev_regret
                best_rep = arm_name
                best_pipeline = pipeline
                best_policy = policy

            print(f"    {arm_name}: dev_regret={dev_regret:.4f}, "
                  f"utility={dev_policy_utility:.4f}, "
                  f"explained_var={pipeline.explained_variance_ratio.sum():.4f}")

    print(f"\n  Best representation: {best_rep} (dev regret={best_dev_regret:.4f})")

    # Save sweep results
    sel_dir = out / "selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    with open(sel_dir / "representation_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    if best_rep is None:
        print("  WARNING: No valid representation found. Using first available.")
        # Fallback: use first available representation
        available_reps = [k for k in hidden["train"].files if "/" in k]
        if available_reps:
            best_rep = f"{available_reps[0]}/pca_32"
            # Refit with fallback
            rep_key = available_reps[0]
            train_h = hidden["train"][rep_key].astype(np.float32)
            pipeline = PCAPipeline()
            train_hidden_reduced = pipeline.fit_transform(train_h, 32)
            best_pipeline = pipeline
        else:
            print("  ERROR: No representations available at all.")
            return
    else:
        # Save best PCA pipeline
        best_pipeline.save(str(sel_dir / "pca_artifact.npz"))

    # 4. Full ablation on final set
    print("\n  Running full ablation on final set...")

    # Prepare features for best hidden representation
    rep_key_parts = best_rep.split("/")
    rep_key = "/".join(rep_key_parts[:2])  # pooling/layer_XX
    pca_dim = int(rep_key_parts[-1].split("_")[1])

    train_h_best = hidden["train"][rep_key].astype(np.float32)
    dev_h_best = hidden["dev"][rep_key].astype(np.float32)
    final_h_best = hidden["final"][rep_key].astype(np.float32)

    # Refit PCA on train
    pipeline = PCAPipeline()
    train_hidden_reduced = pipeline.fit_transform(train_h_best, pca_dim)
    dev_hidden_reduced = pipeline.transform(dev_h_best)
    final_hidden_reduced = pipeline.transform(final_h_best)

    # Logprob features
    train_logprob = logprobs["train"]
    dev_logprob = logprobs["dev"]
    final_logprob = logprobs["final"]

    # Hidden + logprob combined
    train_combined = np.concatenate([train_hidden_reduced, train_logprob], axis=1)
    dev_combined = np.concatenate([dev_hidden_reduced, dev_logprob], axis=1)
    final_combined = np.concatenate([final_hidden_reduced, final_logprob], axis=1)

    # Subtype features (one-hot)
    all_subtypes = sorted(set(train_subtypes))
    subtype_to_idx = {st: i for i, st in enumerate(all_subtypes)}
    n_subtypes = len(all_subtypes)

    def subtype_onehot(subtypes_list):
        arr = np.zeros((len(subtypes_list), n_subtypes), dtype=np.float32)
        for i, st in enumerate(subtypes_list):
            arr[i, subtype_to_idx[st]] = 1.0
        return arr

    train_subtype = subtype_onehot(train_subtypes)
    final_subtype = subtype_onehot(final_subtypes)

    # Train and evaluate each policy
    ablation_results = {}

    def evaluate_policy(name, train_X, final_X, policy_cls=QRegressionPolicy, **kwargs):
        """Train and evaluate a policy."""
        policy = policy_cls(action_ids=action_ids, **kwargs)
        policy.fit(train_X, U_train)
        preds = policy.predict(final_X)
        regret = mean_regret(preds, U_final)
        utility = U_final[np.arange(len(preds)), preds].mean()
        oracle = U_final.max(axis=1).mean()
        gap_capture = (utility - U_final[:, best_fixed_idx].mean()) / (oracle - U_final[:, best_fixed_idx].mean()) if oracle != U_final[:, best_fixed_idx].mean() else 0.0

        result = {
            "final_regret": float(regret),
            "final_utility": float(utility),
            "final_oracle": float(oracle),
            "gap_capture": float(gap_capture),
        }
        ablation_results[name] = result
        print(f"    {name}: regret={regret:.4f}, utility={utility:.4f}, "
              f"gap_capture={gap_capture:.1%}")
        return policy, preds

    # Fixed actions
    for j, aid in enumerate(action_ids):
        preds = np.full(len(U_final), j)
        regret = mean_regret(preds, U_final)
        utility = U_final[:, j].mean()
        oracle = U_final.max(axis=1).mean()
        ablation_results[f"fixed_{aid}"] = {
            "final_regret": float(regret),
            "final_utility": float(utility),
            "final_oracle": float(oracle),
            "gap_capture": 0.0,
        }
        print(f"    fixed_{aid}: regret={regret:.4f}, utility={utility:.4f}")

    # Subtype-only policy
    evaluate_policy("subtype_only", train_subtype, final_subtype,
                    learning_rate=0.01, n_iter=1000, l2=0.01, loss="huber")

    # Logprob policy
    evaluate_policy("logprob", train_logprob, final_logprob,
                    learning_rate=0.01, n_iter=1000, l2=0.001, loss="huber")

    # Hidden policy
    hidden_policy, hidden_preds = evaluate_policy(
        "hidden", train_hidden_reduced, final_hidden_reduced,
        learning_rate=0.01, n_iter=1000, l2=0.001, loss="huber")

    # Hidden + logprob policy
    combined_policy, combined_preds = evaluate_policy(
        "hidden_plus_logprob", train_combined, final_combined,
        learning_rate=0.01, n_iter=1000, l2=0.001, loss="huber")

    # Oracle (for reference)
    oracle_preds = np.argmax(U_final, axis=1)
    oracle_utility = U_final.max(axis=1).mean()
    ablation_results["oracle"] = {
        "final_regret": 0.0,
        "final_utility": float(oracle_utility),
        "final_oracle": float(oracle_utility),
        "gap_capture": 1.0,
    }
    print(f"    oracle: regret=0.0000, utility={oracle_utility:.4f}")

    # 5. Sham experiment
    print(f"\n  Running sham experiment (50 shams)...")
    split_ids_train = np.zeros(len(train_subtypes), dtype=int)  # all train
    split_ids_final = np.full(len(final_subtypes), 2, dtype=int)  # all final

    sham_results = run_sham_experiment(
        train_hidden_reduced, U_train,
        final_hidden_reduced, U_final,
        train_subtypes, final_subtypes,
        split_ids_train, split_ids_final,
        QRegressionPolicy,
        policy_kwargs={
            "action_ids": action_ids,
            "learning_rate": 0.01,
            "n_iter": 1000,
            "l2": 0.001,
            "loss": "huber",
        },
        n_shams=config.get("qualification", {}).get("n_shams", 50),
        seed_base=config.get("qualification", {}).get("sham_seed_base", 10000),
    )

    # 6. Bootstrap confidence intervals
    print(f"\n  Running bootstrap ({config.get('qualification', {}).get('bootstrap_iterations', 10000)} iterations)...")
    n_final = len(U_final)
    n_boot = config.get("qualification", {}).get("bootstrap_iterations", 10000)
    boot_seed = config.get("qualification", {}).get("bootstrap_seed", 20260801)
    rng = np.random.RandomState(boot_seed)

    # Group-based bootstrap
    final_groups_arr = np.array(final_groups)
    unique_groups = np.unique(final_groups_arr)

    hidden_utilities = U_final[np.arange(n_final), hidden_preds]
    combined_utilities = U_final[np.arange(n_final), combined_preds]
    best_fixed_utilities = U_final[:, best_fixed_idx]

    boot_hidden_minus_fixed = []
    boot_combined_minus_logprob = []
    boot_hidden_minus_sham = []

    for _ in range(n_boot):
        # Sample groups with replacement
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        # Map to task indices
        indices = []
        for g in sampled_groups:
            indices.extend(np.where(final_groups_arr == g)[0])
        indices = np.array(indices)

        h_util = hidden_utilities[indices].mean()
        c_util = combined_utilities[indices].mean()
        f_util = best_fixed_utilities[indices].mean()
        l_util = U_final[indices, np.argmax(U_train.mean(axis=0))].mean()  # logprob policy is different

        boot_hidden_minus_fixed.append(h_util - f_util)
        boot_combined_minus_logprob.append(c_util - l_util)

    boot_hf = np.array(boot_hidden_minus_fixed)
    boot_cl = np.array(boot_combined_minus_logprob)

    # 7. Qualification gate
    print(f"\n  Qualification gates:")
    gates = {}

    # Primary: hidden > best-fixed
    h_lcb = float(np.percentile(boot_hf, 2.5))
    gates["hidden_gt_fixed_lcb95"] = h_lcb
    gates["hidden_gt_fixed_pass"] = h_lcb > 0
    print(f"    Hidden > Fixed: LCB95={h_lcb:.4f} {'PASS' if h_lcb > 0 else 'FAIL'}")

    # Hidden > sham
    sham_regrets = np.array(sham_results["sham_regrets"])
    hidden_regret = ablation_results["hidden"]["final_regret"]
    # Hidden should have LOWER regret than sham
    h_vs_sham = sham_regrets - hidden_regret  # positive = hidden is better
    sham_lcb = float(np.percentile(h_vs_sham, 2.5))
    gates["hidden_gt_sham_lcb95"] = sham_lcb
    gates["hidden_gt_sham_pass"] = sham_lcb > 0
    print(f"    Hidden > Sham:  LCB95={sham_lcb:.4f} {'PASS' if sham_lcb > 0 else 'FAIL'}")

    # Gap capture
    gap_capture = ablation_results["hidden"]["gap_capture"]
    gates["gap_capture"] = gap_capture
    gates["gap_capture_pass"] = gap_capture > 0.60
    print(f"    Gap capture:    {gap_capture:.1%} {'PASS' if gap_capture > 0.60 else 'FAIL'}")

    # Positive group fraction
    group_utilities = {}
    for i, g in enumerate(final_groups):
        group_utilities.setdefault(g, []).append(hidden_utilities[i])
    pos_frac = sum(1 for u in group_utilities.values() if np.mean(u) > best_fixed_utilities.mean()) / len(group_utilities)
    gates["positive_group_fraction"] = pos_frac
    gates["positive_group_pass"] = pos_frac > 0.80
    print(f"    Pos group frac: {pos_frac:.1%} {'PASS' if pos_frac > 0.80 else 'FAIL'}")

    overall_pass = all([
        gates["hidden_gt_fixed_pass"],
        gates["hidden_gt_sham_pass"],
        gates["gap_capture_pass"],
        gates["positive_group_pass"],
    ])
    gates["overall_pass"] = overall_pass
    print(f"    Overall:        {'PASS' if overall_pass else 'FAIL'}")

    # 8. Save everything
    pol_dir = out / "policies"
    pol_dir.mkdir(parents=True, exist_ok=True)
    hidden_policy.save(str(pol_dir / "hidden_policy.json"))
    combined_policy.save(str(pol_dir / "hidden_plus_logprob_policy.json"))

    sham_dir = out / "sham"
    sham_dir.mkdir(parents=True, exist_ok=True)
    with open(sham_dir / "sham_results.json", "w") as f:
        json.dump(sham_results, f, indent=2)

    qual_dir = out / "qualification"
    qual_dir.mkdir(parents=True, exist_ok=True)

    qualification = {
        "experiment_id": config.get("experiment_id", "daph_executive_b4"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "best_representation": best_rep,
        "best_fixed_action": best_fixed_action,
        "ablation": ablation_results,
        "sham": sham_results,
        "gates": gates,
        "bootstrap": {
            "hidden_minus_fixed": {
                "point": float(boot_hf.mean()),
                "lcb_95": h_lcb,
                "ucb_95": float(np.percentile(boot_hf, 97.5)),
            },
        },
        "n_train": len(U_train),
        "n_dev": len(U_dev),
        "n_final": len(U_final),
        "action_ids": action_ids,
    }

    with open(qual_dir / "qualification.json", "w") as f:
        json.dump(qualification, f, indent=2)

    # Save report
    report = generate_report(qualification)
    with open(qual_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\n  Saved qualification.json and report.md")
    print(f"\n{'='*60}")
    print(f"  B4 Experiment Complete: {config.get('experiment_id', 'daph_executive_b4')}")
    print(f"  Overall gate: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*60}")


def generate_report(q: dict) -> str:
    """Generate markdown report."""
    lines = [
        "# B4 Executive Qualification Report",
        "",
        f"**Experiment:** `{q['experiment_id']}`",
        f"**Date:** {q['timestamp']}",
        f"**Tasks:** {q['n_train']} train / {q['n_dev']} dev / {q['n_final']} final",
        f"**Actions:** {', '.join(q['action_ids'])}",
        f"**Best representation:** `{q['best_representation']}`",
        f"**Best fixed action:** `{q['best_fixed_action']}`",
        "",
        "## Gate Decision: " + ("PASS" if q['gates']['overall_pass'] else "FAIL"),
        "",
        "## Gates",
        "",
        "| Gate | Value | Threshold | Result |",
        "|------|-------|-----------|--------|",
        f"| Hidden > Fixed (LCB95) | {q['gates']['hidden_gt_fixed_lcb95']:.4f} | > 0 | {'PASS' if q['gates']['hidden_gt_fixed_pass'] else 'FAIL'} |",
        f"| Hidden > Sham (LCB95) | {q['gates']['hidden_gt_sham_lcb95']:.4f} | > 0 | {'PASS' if q['gates']['hidden_gt_sham_pass'] else 'FAIL'} |",
        f"| Gap capture | {q['gates']['gap_capture']:.1%} | > 60% | {'PASS' if q['gates']['gap_capture_pass'] else 'FAIL'} |",
        f"| Positive group fraction | {q['gates']['positive_group_fraction']:.1%} | > 80% | {'PASS' if q['gates']['positive_group_pass'] else 'FAIL'} |",
        "",
        "## Ablation Results (Final Set)",
        "",
        "| Policy | Regret | Utility | Gap Capture |",
        "|--------|--------|---------|-------------|",
    ]

    for name, res in q["ablation"].items():
        lines.append(f"| {name} | {res['final_regret']:.4f} | {res['final_utility']:.4f} | {res.get('gap_capture', 0):.1%} |")

    lines.extend([
        "",
        "## Sham Results",
        "",
        f"- Number of shams: {q['sham']['n_shams']}",
        f"- Sham mean regret: {q['sham']['sham_mean_regret']:.4f}",
        f"- Sham std regret: {q['sham']['sham_std_regret']:.4f}",
        f"- Hidden regret: {q['ablation']['hidden']['final_regret']:.4f}",
        f"- Hidden vs Sham (LCB95): {q['gates']['hidden_gt_sham_lcb95']:.4f}",
        "",
    ])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="B4 Staged Runner")
    parser.add_argument("--stage", required=True,
                        choices=["A", "B", "C", "all"],
                        help="Which stage to run")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock execution (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = args.config

    if args.stage in ("A", "all"):
        run_stage_a(config, mock=args.mock)
    if args.stage in ("B", "all"):
        run_stage_b(config, mock=args.mock)
    if args.stage in ("C", "all"):
        run_stage_c(config, mock=args.mock)


if __name__ == "__main__":
    main()
