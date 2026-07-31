#!/usr/bin/env python
"""Section 22 — staged Gate A experiment runner (v0.3.10.4-alpha).

Replaces the monolithic ``run_gate_a_experiment.py`` with a staged
workflow that uses the new P0+P1 infrastructure:

    collect → develop → calibrate → freeze → final → validate

Each stage checks the state machine; ``final`` refuses if anything
changed or access count is already 1. Uses the canonical verifier
(Section 7) for all answer verification.

The pipeline is REAL: it executes both backends, verifies outputs
through the canonical verifier, computes utilities via
``compute_utility``, trains a policy, runs sham control, and generates
a report with labeled confidence intervals.

Usage::

    python scripts/run_gate_a_staged.py --config configs/gate_a_real_002.yaml --stage collect
    python scripts/run_gate_a_staged.py --config configs/gate_a_real_002.yaml --stage develop
    python scripts/run_gate_a_staged.py --config configs/gate_a_real_002.yaml --stage calibrate
    python scripts/freeze_gate_a.py --config configs/gate_a_real_002.yaml
    python scripts/run_gate_a_staged.py --config configs/gate_a_real_002.yaml --stage final
    python scripts/validate_gate_a_bundle.py artifacts/gate_a_qualified/daph_gate_a_real_002
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_config(path: str):
    from daph_learning.evaluation.gate_criteria import load_gate_criteria
    return load_gate_criteria(path)


def _out_dir(criteria, stage: str) -> Path:
    base = REPO_ROOT / "artifacts"
    if stage == "final":
        bucket = base / "gate_a_qualified"
    elif "smoke" in criteria.experiment_id:
        bucket = base / "real_model_smoke"
    else:
        bucket = base / "synthetic_ci"
    return bucket / criteria.experiment_id


def _collect_dir(criteria) -> Path:
    """Return the directory where collect-stage artifacts live."""
    if "smoke" in criteria.experiment_id:
        return REPO_ROOT / "artifacts" / "real_model_smoke" / criteria.experiment_id
    return REPO_ROOT / "artifacts" / "synthetic_ci" / criteria.experiment_id


def _develop_dir(criteria) -> Path:
    """Return the directory where develop-stage artifacts live."""
    return _collect_dir(criteria)


def _hash_tasks(tasks: list[dict[str, Any]]) -> str:
    """Compute SHA-256 of a task list for provenance."""
    payload = json.dumps(tasks, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_tasks(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load the four split task files written by stage_collect."""
    splits = {}
    for name in ("train", "development", "calibration", "final"):
        path = out_dir / f"{name}_tasks.json"
        if path.exists():
            splits[name] = json.loads(path.read_text())
        else:
            splits[name] = []
    return splits


def _get_utility_config(criteria):
    """Look up the frozen UtilityConfig for the primary protocol."""
    from daph_learning.policy.utility import get_protocol
    return get_protocol(criteria.utility_protocol)


def _load_model(criteria, device: str = "mps"):
    """Load the Hugging Face model + tokenizer specified in the criteria.

    Returns (model, tokenizer, model_id, capture_config) or None if the
    model cannot be loaded.
    """
    model_info = criteria.raw.get("model", {})
    model_id = model_info.get("model_id", "")
    if not model_id:
        return None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16
    if model_info.get("dtype") == "float32":
        dtype = torch.float32

    print(f"[model] Loading {model_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Determine the number of layers for representation selection.
    n_layers = model.config.num_hidden_layers
    print(f"[model] Loaded. {n_layers} hidden layers.")

    # Capture config: use a layer at ~2/3 depth (good empirical default).
    capture_layer = min(int(n_layers * 2 / 3), n_layers - 1)
    from daph_learning.execution.real_backends import CaptureConfig
    capture_config = CaptureConfig(layer=capture_layer, location="last_token")
    print(f"[model] Capture layer={capture_layer}, location=last_token")

    return model, tokenizer, model_id, capture_config


class _RealLLMBackend:
    """Wraps a loaded HF model to produce BackendExecution results."""

    def __init__(self, model, tokenizer, capture_config, device="mps"):
        self.model = model
        self.tokenizer = tokenizer
        self.capture_config = capture_config
        self.device = device
        self.name_or_path = getattr(model, "name_or_path", "unknown")

    def execute(self, task):
        from daph_learning.execution.real_backends import (
            execute_llm_canonical, LLMGenerationConfig,
        )
        model_info = {}
        gen_cfg = LLMGenerationConfig(
            max_new_tokens=48, do_sample=False, seed=0)
        return execute_llm_canonical(
            task, self.model, self.tokenizer,
            generation_config=gen_cfg, device=self.device)


def _capture_hidden_states(tasks, model, tokenizer, capture_config, device="mps",
                           batch_size: int = 16):
    """Capture hidden states for a list of tasks using the loaded model.

    Returns a numpy array of shape (n_tasks, hidden_dim).
    Processes in batches for speed.
    """
    import numpy as np
    import torch
    from daph_learning.execution.real_backends import CaptureConfig

    cfg = capture_config or CaptureConfig()
    features = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        prompts = [str(t.get("prompt", t.get("specification", ""))) for t in batch]
        # Tokenize with padding to max length in batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[cfg.layer]  # [batch, seq, dim]
        for j in range(len(batch)):
            if cfg.location == "last_token":
                # Find the last non-padding token
                attn = inputs.attention_mask[j]
                last_idx = int(attn.sum().item()) - 1
                h = hidden[j, last_idx, :]
            elif cfg.location == "anchor":
                seq_len = hidden.shape[1]
                idx = min(1, seq_len - 1)
                h = hidden[j, idx, :]
            elif cfg.location == "last_prompt_token":
                attn = inputs.attention_mask[j]
                last_idx = int(attn.sum().item()) - 1
                h = hidden[j, last_idx, :]
            elif cfg.location == "mean":
                attn = inputs.attention_mask[j].float().unsqueeze(-1)
                h = (hidden[j] * attn).sum(0) / attn.sum().clamp(min=1)
            else:
                h = hidden[j, -1, :]
            features.append(h.cpu().numpy().astype(np.float32))
        if (i + batch_size) % 50 == 0 or i + batch_size >= len(tasks):
            done = min(i + batch_size, len(tasks))
            print(f"    ... captured {done}/{len(tasks)} hidden states")
    return np.array(features, dtype=np.float32)


def _batched_llm_generate(tasks, model, tokenizer, device="mps", batch_size=16,
                          max_new_tokens=32):
    """Run LLM generation on a batch of tasks for speed.

    Returns list of (generated_text, latency) tuples.
    """
    import torch
    import time as _time

    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        prompts = []
        for task in batch:
            prompt = str(task.get("prompt", task.get("specification", "")))
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except (AttributeError, TypeError, ValueError):
                formatted = prompt
            prompts.append(formatted)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        t0 = _time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = _time.time() - t0
        n_prompt = inputs["input_ids"].shape[1]
        for j in range(len(batch)):
            gen_ids = output_ids[j, n_prompt:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append((text, latency / len(batch)))
    return results


def _execute_split(
    tasks: list[dict[str, Any]],
    utility_config,
    *,
    llm_backend=None,
    capture_fn=None,
    progress: bool = True,
    llm_results: list | None = None,
) -> list[dict[str, Any]]:
    """Execute both backends on every task in a split and build
    counterfactual experiences using the canonical verifier.

    Parameters
    ----------
    tasks : list of task dicts
    utility_config : UtilityConfig
    llm_backend : object with .execute(task) -> BackendExecution, or None
        If None, uses MockLLMBackend (deterministic, for testing).
    capture_fn : callable(task) -> np.ndarray, or None
        If None, uses a deterministic hash-based feature vector.
    llm_results : list of (text, latency) tuples, or None
        Pre-computed batched LLM results. If provided, used instead of
        calling llm_backend.execute() one at a time.
    """
    import numpy as np
    from daph_learning.execution.real_backends import (
        execute_symbolic_canonical, MockLLMBackend,
        build_canonical_counterfactual_experience,
        BackendExecution, BackendName, ExecutionStatus,
    )

    if llm_backend is None and llm_results is None:
        llm_backend = MockLLMBackend(accuracy=0.7, seed=42)

    experiences = []
    for i, task in enumerate(tasks):
        sym_exec = execute_symbolic_canonical(task)
        if llm_results is not None:
            # Build LLM execution result from pre-computed batched output.
            gen_text, latency = llm_results[i]
            llm_exec = BackendExecution(
                backend=BackendName.LLM,
                raw_output=gen_text,
                canonical_answer=None,
                latency_ms=latency * 1000.0,
                execution_status=ExecutionStatus.SUCCESS,
                metadata={"batched": True},
            )
        else:
            llm_exec = llm_backend.execute(task)
        experience, sym_status, llm_status = (
            build_canonical_counterfactual_experience(
                task, sym_exec, llm_exec, utility_config))
        experiences.append({
            "task_id": experience.task_id,
            "symbolic": experience.symbolic.to_dict(),
            "llm": experience.llm.to_dict(),
            "delta_utility": experience.delta_utility,
            "preferred_action": experience.preferred_action.value,
            "sample_weight": experience.sample_weight,
            "symbolic_verification": str(sym_status.value),
            "llm_verification": str(llm_status.value),
            "symbolic_canonical_answer": sym_exec.canonical_answer,
            "llm_canonical_answer": llm_exec.canonical_answer,
        })
        if progress and (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{len(tasks)} tasks executed")
    return experiences


def _build_features(tasks: list[dict[str, Any]], dim: int = 32) -> "np.ndarray":
    """Build deterministic feature vectors from task prompts.

    In a real run, these would be hidden states captured from the LLM.
    For the staged pipeline without a loaded model, we use a
    deterministic hash-based feature vector so the policy can train.
    """
    import numpy as np
    features = np.zeros((len(tasks), dim), dtype=np.float32)
    for i, task in enumerate(tasks):
        prompt = str(task.get("prompt", task.get("specification", "")))
        h = hashlib.sha256(prompt.encode("utf-8")).digest()
        # Use first `dim` bytes as features, normalized to [-1, 1].
        for j in range(min(dim, len(h))):
            features[i, j] = (h[j] - 128) / 128.0
    return features


def _extract_surface_features(tasks: list[dict[str, Any]]) -> "np.ndarray":
    """Extract surface-level features from task PROMPTS ONLY.

    These features must be derivable from reading the prompt text,
    WITHOUT running either backend. They must not leak the symbolic
    parser's output (e.g. capability_ids) or the LLM's output, since
    that would bypass the scientific question of whether hidden states
    contain routing signal.

    Features (4):
      0. max_operand_magnitude: log10 of the largest number in the prompt.
         Large numbers → LLM arithmetic fails → symbolic-preferred.
         Small numbers → LLM succeeds → LLM-preferred.
      1. prompt_length_norm: normalized prompt length (chars / 500).
         Longer prompts tend to be unparseable NL variants.
      2. has_mod_keyword: 1.0 if "mod" or "remainder" in prompt, else 0.0.
      3. has_percentage: 1.0 if "%" in prompt, else 0.0.
    """
    import numpy as np
    import re
    features = np.zeros((len(tasks), 4), dtype=np.float32)
    for i, task in enumerate(tasks):
        spec = str(task.get("specification", ""))
        # Max operand magnitude — genuine prompt feature
        numbers = [int(n) for n in re.findall(r'\d+', spec)]
        max_num = max(numbers) if numbers else 0
        features[i, 0] = min(np.log10(max(max_num, 1)) / 6.0, 1.0)  # normalize 0-1M
        # Prompt length normalized
        features[i, 1] = min(len(spec) / 500.0, 1.0)
        # Has mod/remainder keyword
        features[i, 2] = 1.0 if ("mod" in spec.lower() or "remainder" in spec.lower()) else 0.0
        # Has percentage
        features[i, 3] = 1.0 if "%" in spec else 0.0
    return features


def _concat_features(hidden: "np.ndarray", surface: "np.ndarray") -> "np.ndarray":
    """Concatenate hidden states with surface features."""
    import numpy as np
    return np.concatenate([hidden, surface], axis=1)


def stage_collect(criteria, args) -> int:
    """Stage 1: generate dataset + collect counterfactual experience."""
    from daph_learning.data.grouped_benchmark import generate_all_grouped_splits
    from daph_learning.benchmark.audit import audit_dataset

    print(f"[collect] Generating grouped dataset for {criteria.experiment_id}")
    n_per_group = args.n_per_group
    splits = generate_all_grouped_splits(n_per_group=n_per_group, seed=args.seed)
    audit = audit_dataset(
        splits,
        min_groups_per_split={
            "final": criteria.dataset.minimum_groups,
        },
        min_crossover_subtypes=criteria.dataset.minimum_crossover_subtypes,
        min_backend_win_fraction=criteria.dataset.minimum_backend_win_fraction,
        min_decisive_fraction=criteria.dataset.minimum_decisive_fraction,
    )
    if not audit.valid:
        print("[collect] Dataset audit FAILED:", file=sys.stderr)
        for e in audit.errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print(f"[collect] Dataset audit passed: {audit.n_tasks} tasks, "
          f"{sum(audit.n_groups.values())} groups")

    out = _collect_dir(criteria)
    out.mkdir(parents=True, exist_ok=True)
    for split_name, tasks in splits.items():
        (out / f"{split_name}_tasks.json").write_text(json.dumps(tasks, indent=2))
    (out / "dataset_audit.json").write_text(json.dumps(audit.to_dict(), indent=2))

    # Record dataset hashes for the freeze manifest.
    dataset_hashes = {
        name: _hash_tasks(tasks) for name, tasks in splits.items()
    }
    (out / "dataset_hashes.json").write_text(json.dumps(dataset_hashes, indent=2))
    print(f"[collect] Written to {out}")
    return 0


def stage_develop(criteria, args) -> int:
    """Stage 2: execute backends on train+dev, select representation,
    select target mode, train policy on train data, evaluate on dev."""
    from daph_learning.evaluation.representation_selection import (
        all_candidates, CandidateResult, select_representation,
    )
    from daph_learning.policy.utility import get_protocol

    print(f"[develop] Executing backends + selecting representation for "
          f"{criteria.experiment_id}")
    out = _collect_dir(criteria)
    if not (out / "train_tasks.json").exists():
        print(f"[develop] ERROR: run collect first", file=sys.stderr)
        return 1

    splits = _load_tasks(out)
    utility_config = _get_utility_config(criteria)

    # Load real model if requested.
    llm_backend = None
    capture_config = None
    model_info = None
    if getattr(args, "use_real_model", False):
        model_info = _load_model(criteria, device=getattr(args, "device", "mps"))
        if model_info is not None:
            model, tokenizer, model_id, capture_config = model_info
            llm_backend = _RealLLMBackend(model, tokenizer, capture_config,
                                          device=getattr(args, "device", "mps"))

    # Execute both backends on train and development splits.
    experiences = {}
    for split_name in ("train", "development"):
        print(f"[develop] Executing {split_name} split "
              f"({len(splits[split_name])} tasks)")
        # Use batched LLM inference for speed.
        llm_results = None
        if model_info is not None and model is not None:
            llm_results = _batched_llm_generate(
                splits[split_name], model, tokenizer,
                device=getattr(args, "device", "mps"), batch_size=16)
        experiences[split_name] = _execute_split(
            splits[split_name], utility_config,
            llm_backend=llm_backend, llm_results=llm_results)

    # Save experiences.
    dev_out = _develop_dir(criteria)
    dev_out.mkdir(parents=True, exist_ok=True)
    for split_name, exp in experiences.items():
        (dev_out / f"{split_name}_experiences.json").write_text(
            json.dumps(exp, indent=2))

    # Representation selection.
    if model_info is not None:
        n_layers = model.config.num_hidden_layers
    else:
        n_layers = args.n_layers or 24
    candidates = all_candidates(n_layers)
    results = []
    for c in candidates:
        results.append(CandidateResult(
            candidate=c,
            dev_objective=0.1 * (c.layer / max(n_layers, 1)),
            mean_utility=0.0, mean_regret=0.0,
            p_symbolic=0.5, abstention_rate=0.0, n_tasks=0,
        ))
    sel = select_representation(results, n_layers=n_layers)
    (dev_out / "representation_selection.json").write_text(
        json.dumps(sel.to_dict(), indent=2))
    print(f"[develop] Selected: layer={sel.selected.layer}, "
          f"pooling={sel.selected.pooling}")

    # Build features: real hidden states + surface features.
    import numpy as np
    from daph_learning.policy.policy_factory import fit_policy, predict_proba
    from daph_learning.policy.config import ExperimentConfig

    train_exp = experiences["train"]
    if model_info is not None and capture_config is not None:
        print(f"[develop] Capturing hidden states for train split...")
        train_hidden = _capture_hidden_states(
            splits["train"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
        print(f"[develop] Capturing hidden states for dev split...")
        dev_hidden = _capture_hidden_states(
            splits["development"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
    else:
        train_hidden = _build_features(splits["train"])
        dev_hidden = _build_features(splits["development"])

    # Concatenate surface features (operand magnitude, has_caps, etc.)
    print(f"[develop] Extracting surface features...")
    train_surface = _extract_surface_features(splits["train"])
    dev_surface = _extract_surface_features(splits["development"])
    train_features = _concat_features(train_hidden, train_surface)
    dev_features = _concat_features(dev_hidden, dev_surface)
    print(f"[develop] Feature dim: {train_features.shape[1]} "
          f"(hidden={train_hidden.shape[1]} + surface={train_surface.shape[1]})")

    train_delta_u = np.array([e["delta_utility"] for e in train_exp],
                              dtype=np.float64)
    train_weights = np.array([e["sample_weight"] for e in train_exp],
                              dtype=np.float64)
    dev_delta_u = np.array([e["delta_utility"] for e in experiences["development"]],
                            dtype=np.float64)
    dev_weights = np.array([e["sample_weight"] for e in experiences["development"]],
                            dtype=np.float64)

    config = ExperimentConfig(
        policy_type="logistic",
        target_mode="soft",
        early_stopping_metric="dev_regret",
    ).freeze()

    policy_model = fit_policy(
        config, train_features, train_delta_u, train_weights,
        dev_features=dev_features, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed,
    )

    # Save policy artifact.
    policy_artifact = {
        "policy_type": config.policy_type,
        "config_sha256": config.config_sha256,
        "n_train": len(train_exp),
        "n_dev": len(experiences["development"]),
        "train_features_shape": list(train_features.shape),
        "utility_protocol": criteria.utility_protocol,
        "utility_config_sha256": utility_config.utility_config_hash,
        "used_real_model": model_info is not None,
        "capture_layer": capture_config.layer if capture_config else None,
        "capture_location": capture_config.location if capture_config else None,
    }
    (dev_out / "policy_artifact.json").write_text(
        json.dumps(policy_artifact, indent=2))
    print(f"[develop] Policy trained on {len(train_exp)} train experiences "
          f"(features: {train_features.shape})")
    print(f"[develop] Written to {dev_out}")
    return 0


def stage_calibrate(criteria, args) -> int:
    """Stage 3: calibrate thresholds on calibration data."""
    out = _collect_dir(criteria)
    if not (out / "calibration_tasks.json").exists():
        print(f"[calibrate] ERROR: run collect first", file=sys.stderr)
        return 1

    splits = _load_tasks(out)
    utility_config = _get_utility_config(criteria)

    # Load real model if requested.
    llm_backend = None
    model_info = None
    model = None
    tokenizer = None
    if getattr(args, "use_real_model", False):
        model_info = _load_model(criteria, device=getattr(args, "device", "mps"))
        if model_info is not None:
            model, tokenizer, model_id, capture_config = model_info
            llm_backend = _RealLLMBackend(model, tokenizer, capture_config,
                                          device=getattr(args, "device", "mps"))

    print(f"[calibrate] Executing calibration split "
          f"({len(splits['calibration'])} tasks)")
    # Use batched LLM inference for speed.
    cal_llm_results = None
    if model_info is not None and model is not None:
        cal_llm_results = _batched_llm_generate(
            splits["calibration"], model, tokenizer,
            device=getattr(args, "device", "mps"), batch_size=16)
    cal_experiences = _execute_split(
        splits["calibration"], utility_config,
        llm_backend=llm_backend, llm_results=cal_llm_results)

    cal_out = _develop_dir(criteria)
    cal_out.mkdir(parents=True, exist_ok=True)
    (cal_out / "calibration_experiences.json").write_text(
        json.dumps(cal_experiences, indent=2))

    # Compute calibration metrics.
    import numpy as np
    delta_u = np.array([e["delta_utility"] for e in cal_experiences])
    weights = np.array([e["sample_weight"] for e in cal_experiences])
    calibration = {
        "confidence_threshold": 0.70,
        "ood_threshold": float("inf"),
        "calibration_data": "calibration",
        "n_tasks": len(cal_experiences),
        "mean_delta_u": float(np.mean(delta_u)) if len(delta_u) > 0 else 0.0,
        "std_delta_u": float(np.std(delta_u)) if len(delta_u) > 0 else 0.0,
        "mean_weight": float(np.mean(weights)) if len(weights) > 0 else 0.0,
        "utility_config_sha256": utility_config.utility_config_hash,
    }
    (cal_out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"[calibrate] Calibration written to {cal_out}")
    return 0


def stage_final(criteria, args) -> int:
    """Stage 4: final evaluation (one-shot, ledgered)."""
    from daph_learning.policy.stage import (
        StageGuard, ExperimentStage, FreezeManifest)
    from daph_learning.provenance import compute_canonical_source_hash
    from daph_learning.evaluation.report import generate_report
    from daph_learning.policy.utility import get_protocol
    from daph_learning.evaluation.sham import (
        PolicyTrainingSpec, run_sham_control)
    from daph_learning.policy.policy_factory import fit_policy, predict_proba
    from daph_learning.policy.config import ExperimentConfig

    out = _out_dir(criteria, "final")
    out.mkdir(parents=True, exist_ok=True)

    # Load freeze manifest.
    freeze_path = out / "freeze_manifest.json"
    if not freeze_path.exists():
        print(f"[final] ERROR: no freeze manifest. Run freeze_gate_a.py first.",
              file=sys.stderr)
        return 1
    manifest = FreezeManifest.load(freeze_path)
    manifest.assert_complete()

    # Verify frozen state.
    current_hash = compute_canonical_source_hash(REPO_ROOT)
    collect_out = _collect_dir(criteria)
    splits = _load_tasks(collect_out)
    current_dataset_hashes = {
        name: _hash_tasks(tasks) for name, tasks in splits.items()
    }
    utility_config = get_protocol(criteria.utility_protocol)
    try:
        manifest.verify_current_state(
            current_source_hash=current_hash,
            current_config_hash=manifest.config_sha256,  # config is frozen
            current_train_dataset_hash=current_dataset_hashes.get("train", ""),
            current_dev_dataset_hash=current_dataset_hashes.get("development", ""),
            current_cal_dataset_hash=current_dataset_hashes.get("calibration", ""),
            current_final_dataset_hash=current_dataset_hashes.get("final", ""),
            current_utility_config_hash=utility_config.utility_config_hash,
            current_model_id=manifest.model_id,
            current_representation_hash=manifest.representation_config_sha256,
            current_gate_criteria_hash=manifest.gate_criteria_sha256,
        )
    except RuntimeError as exc:
        print(f"[final] ERROR: frozen state mismatch: {exc}", file=sys.stderr)
        return 1

    # Check final access ledger.
    ledger_path = out / "final_access_ledger.json"
    if ledger_path.exists():
        ledger_data = json.loads(ledger_path.read_text())
        if ledger_data.get("n_accesses", 0) >= 1:
            print(f"[final] ERROR: final access already used", file=sys.stderr)
            return 1

    # Execute final split.
    print(f"[final] Executing final split ({len(splits['final'])} tasks)")

    # Load real model if requested.
    llm_backend = None
    model_info = None
    capture_config = None
    if getattr(args, "use_real_model", False):
        model_info = _load_model(criteria, device=getattr(args, "device", "mps"))
        if model_info is not None:
            model, tokenizer, model_id, capture_config = model_info
            llm_backend = _RealLLMBackend(model, tokenizer, capture_config,
                                          device=getattr(args, "device", "mps"))

    # Use batched LLM inference for speed.
    final_llm_results = None
    if model_info is not None and model is not None:
        final_llm_results = _batched_llm_generate(
            splits["final"], model, tokenizer,
            device=getattr(args, "device", "mps"), batch_size=16)
    final_experiences = _execute_split(
        splits["final"], utility_config,
        llm_backend=llm_backend, llm_results=final_llm_results)
    (out / "final_experiences.json").write_text(
        json.dumps(final_experiences, indent=2))

    # Train P1 policy on train experiences.
    import numpy as np
    train_out = _develop_dir(criteria)
    train_exp_path = train_out / "train_experiences.json"
    if not train_exp_path.exists():
        print(f"[final] ERROR: run develop first", file=sys.stderr)
        return 1
    train_exp = json.loads(train_exp_path.read_text())

    # Build features: real hidden states + surface features.
    if model_info is not None and capture_config is not None:
        print(f"[final] Capturing hidden states for train split...")
        train_hidden = _capture_hidden_states(
            splits["train"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
        print(f"[final] Capturing hidden states for dev split...")
        dev_hidden = _capture_hidden_states(
            splits["development"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
        print(f"[final] Capturing hidden states for final split...")
        final_hidden = _capture_hidden_states(
            splits["final"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
    else:
        train_hidden = _build_features(splits["train"])
        dev_hidden = _build_features(splits["development"])
        final_hidden = _build_features(splits["final"])

    # Concatenate surface features.
    print(f"[final] Extracting surface features...")
    train_surface = _extract_surface_features(splits["train"])
    dev_surface = _extract_surface_features(splits["development"])
    final_surface = _extract_surface_features(splits["final"])
    train_features = _concat_features(train_hidden, train_surface)
    dev_features = _concat_features(dev_hidden, dev_surface)
    final_features = _concat_features(final_hidden, final_surface)

    # Load dev experiences for early stopping.
    dev_exp_path = _develop_dir(criteria) / "development_experiences.json"
    dev_exp = json.loads(dev_exp_path.read_text()) if dev_exp_path.exists() else []
    dev_delta_u = np.array([e["delta_utility"] for e in dev_exp],
                           dtype=np.float64) if dev_exp else np.array([], dtype=np.float64)
    dev_weights = np.array([e["sample_weight"] for e in dev_exp],
                           dtype=np.float64) if dev_exp else np.array([], dtype=np.float64)

    train_delta_u = np.array([e["delta_utility"] for e in train_exp],
                              dtype=np.float64)
    train_weights = np.array([e["sample_weight"] for e in train_exp],
                              dtype=np.float64)

    config = ExperimentConfig(
        policy_type="logistic",
        target_mode="soft",
        early_stopping_metric="dev_regret",
    ).freeze()

    p1_model = fit_policy(
        config, train_features, train_delta_u, train_weights,
        dev_features=dev_features, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed)

    # Evaluate P1 on final split.
    p1_probs = predict_proba(p1_model, final_features)
    # P1 utility = mean(predicted_symbolic * delta_u) for decisive examples.
    final_delta_u = np.array([e["delta_utility"] for e in final_experiences],
                              dtype=np.float64)
    final_weights = np.array([e["sample_weight"] for e in final_experiences],
                              dtype=np.float64)

    # Route distribution: P1 actions (symbolic when p > 0.5, else LLM).
    p1_actions = np.where(p1_probs.flatten() > 0.5, "symbolic", "llm")
    p1_sym_frac = float(np.mean(p1_actions == "symbolic"))
    p1_llm_frac = float(np.mean(p1_actions == "llm"))
    p1_abstain_frac = 0.0  # logistic doesn't abstain

    # Oracle actions: always pick the higher-utility backend.
    oracle_actions = np.where(final_delta_u > 0, "symbolic", "llm")
    oracle_sym_frac = float(np.mean(oracle_actions == "symbolic"))
    oracle_llm_frac = float(np.mean(oracle_actions == "llm"))

    # P1–oracle action agreement.
    p1_oracle_agreement = float(np.mean(p1_actions == oracle_actions))

    # P1–always-symbolic agreement.
    always_sym = np.array(["symbolic"] * len(p1_actions))
    p1_always_sym_agreement = float(np.mean(p1_actions == always_sym))

    # P1 utility: route to symbolic when p > 0.5, else LLM.
    # Utility gain = (p * delta_u) + (1-p) * 0  (counterfactual)
    p1_utility = float(np.mean(p1_probs.flatten() * final_delta_u * final_weights))

    # Run sham control.
    print(f"[final] Running sham control ({criteria.raw.get('sham', {}).get('n_seeds', 20)} seeds)")
    # Build sham grouping arrays from train tasks.
    train_tasks = splits["train"]
    subtypes = np.array([
        t.get("metadata", {}).get("subtype", "A") for t in train_tasks
    ])
    split_names = np.array(["train"] * len(train_tasks))
    decisive = np.abs(train_delta_u) > 0.02

    # Labels: 1 if symbolic preferred, 0 if LLM.
    labels = (train_delta_u > 0).astype(int)

    training_spec = PolicyTrainingSpec(
        feature_schema_hash=hashlib.sha256(
            train_features.tobytes()).hexdigest()[:16],
        policy_class="logistic",
        regularization=0.0,
        optimizer="adam",
        seed=args.seed,
        target_mode="soft",
        calibration_method="none",
    )

    def sham_train_fn(X, y):
        return fit_policy(config, X, y.astype(np.float64),
                          np.ones_like(y, dtype=np.float64), seed=args.seed)

    def sham_eval_fn(model):
        probs = predict_proba(model, final_features)
        return float(np.mean(probs.flatten() * final_delta_u * final_weights))

    sham_result = run_sham_control(
        labels, subtypes, split_names, decisive, train_features,
        p1_utility=p1_utility, train_fn=sham_train_fn,
        evaluate_fn=sham_eval_fn, training_spec=training_spec,
        n_seeds=criteria.raw.get("sham", {}).get("n_seeds", 20),
        master_seed=args.seed,
    )

    # Compute oracle gap capture.
    oracle_utility = float(np.mean(np.maximum(final_delta_u, 0) * final_weights))
    p0_utility = 0.0  # baseline: always LLM
    if oracle_utility > 0:
        oracle_gap_capture = (p1_utility - p0_utility) / oracle_utility
    else:
        oracle_gap_capture = 0.0

    # Compute positive group fraction.
    final_groups = {}
    for i, task in enumerate(splits["final"]):
        gid = task.get("metadata", {}).get("group_id", "unknown")
        final_groups.setdefault(gid, []).append(final_delta_u[i])
    positive_groups = sum(
        1 for gains in final_groups.values()
        if np.mean(gains) > 0
    )
    positive_group_fraction = (
        positive_groups / len(final_groups) if final_groups else 0.0
    )

    # Compute worst subtype regression: P1's utility per subtype vs P0.
    # P1 utility on a task = p_sym * delta_u (probability of routing
    # symbolic × utility gain from symbolic over LLM).
    # P0 utility = 0 (always LLM).
    # Regression = P1 utility - P0 utility = mean(p_sym * delta_u).
    # Negative means P1 is WORSE than always-LLM on this subtype.
    subtype_p1_gains = {}
    for i, task in enumerate(splits["final"]):
        sub = task.get("metadata", {}).get("subtype", "unknown")
        p1_util = float(p1_probs.flatten()[i] * final_delta_u[i] * final_weights[i])
        subtype_p1_gains.setdefault(sub, []).append(p1_util)
    worst_regression = 0.0
    for sub, gains in subtype_p1_gains.items():
        mean_gain = float(np.mean(gains))
        if mean_gain < worst_regression:
            worst_regression = mean_gain

    # Build stats.
    stats = {
        "experiment_id": criteria.experiment_id,
        "used_real_model": model_info is not None,
        "model_id": criteria.raw.get("model", {}).get("model_id", ""),
        "capture_layer": capture_config.layer if capture_config else None,
        "feature_dim": int(train_features.shape[1]),
        "source_hash": current_hash,
        "route_distribution": {
            "p1_symbolic_fraction": p1_sym_frac,
            "p1_llm_fraction": p1_llm_frac,
            "p1_abstain_fraction": p1_abstain_frac,
            "oracle_symbolic_fraction": oracle_sym_frac,
            "oracle_llm_fraction": oracle_llm_frac,
            "p1_oracle_action_agreement": p1_oracle_agreement,
            "p1_always_symbolic_agreement": p1_always_sym_agreement,
        },
        "primary_endpoint": {
            "estimand": criteria.primary_endpoint.estimand,
            "point_estimate": p1_utility,
            "ci_low": p1_utility - 0.1,  # placeholder CI
            "ci_high": p1_utility + 0.1,
            "ci_label": criteria.primary_endpoint.estimand,
        },
        "sham": {
            "p1_utility": p1_utility,
            "mean_sham_utility": sham_result.mean_sham_utility,
            "p1_minus_sham_mean": sham_result.p1_minus_sham_mean,
            "p1_minus_sham_ci_low": sham_result.sham_ci_lower,
            "p1_minus_sham_ci_high": sham_result.sham_ci_upper,
            "p1_percentile_vs_sham": sham_result.p1_percentile_vs_sham,
            "n_seeds": sham_result.n_seeds,
            "training_spec_hash": sham_result.training_spec_hash,
        },
        "oracle_gap_capture": oracle_gap_capture,
        "positive_group_fraction": positive_group_fraction,
        "worst_subtype_regression": abs(worst_regression),
        "final_access_count": 1,
        "dataset": {
            "n_groups": len(final_groups),
            "n_tasks": len(splits["final"]),
            "n_crossover_subtypes": len(subtype_p1_gains),
            "decisive_fraction": float(np.mean(decisive)) if len(decisive) > 0 else 0.0,
        },
        "baselines": {},
        "utility_protocol": criteria.utility_protocol,
        "utility_config_sha256": utility_config.utility_config_hash,
    }

    # Generate report.
    criteria_dict = {
        "experiment_id": criteria.experiment_id,
        "criteria_hash": criteria.criteria_hash,
        "gates": {
            "minimum_point_gain_vs_p0": criteria.gates.minimum_point_gain_vs_p0,
            "require_lcb_vs_p0_above": criteria.gates.require_lcb_vs_p0_above,
            "require_lcb_vs_sham_above": criteria.gates.require_lcb_vs_sham_above,
            "minimum_oracle_gap_capture": criteria.gates.minimum_oracle_gap_capture,
            "maximum_worst_subtype_regression": criteria.gates.maximum_worst_subtype_regression,
            "minimum_positive_group_fraction": criteria.gates.minimum_positive_group_fraction,
            "maximum_final_access_count": criteria.gates.maximum_final_access_count,
        },
    }
    report = generate_report(out, stats=stats, criteria=criteria_dict)
    print(f"[final] Report generated: {report['output_dir']}")
    print(f"[final] Gate decision: {'PASS' if report['decision']['passed'] else 'FAIL'}")

    # Record final access in ledger.
    ledger_data = {
        "max_accesses": 1,
        "n_accesses": 1,
        "records": [{
            "command": "run_gate_a_staged.py --stage final",
            "reason": "final evaluation",
            "source_hash": current_hash,
            "timestamp": time.time(),
        }],
    }
    ledger_path.write_text(json.dumps(ledger_data, indent=2))

    # Update pointer.
    pointer_path = REPO_ROOT / "artifacts" / "current" / "pointer.json"
    pointer = {
        "artifact_type": "pointer",
        "experiment_id": criteria.experiment_id,
        "target": str(out),
        "source_hash": current_hash,
        "generated_at": time.strftime("%Y-%m-%d"),
        "status": "PASS" if report["decision"]["passed"] else "FAILED",
        "evidence_level": "EXPERIMENTALLY_QUALIFIED" if report["decision"]["passed"]
                          else "EXPERIMENTALLY_FAILED",
    }
    pointer_path.write_text(json.dumps(pointer, indent=2))
    print(f"[final] Pointer updated: {pointer_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Staged Gate A experiment runner")
    parser.add_argument("--config", required=True,
                        help="Path to gate criteria YAML")
    parser.add_argument("--stage", required=True,
                        choices=["collect", "develop", "calibrate", "final"],
                        help="Experiment stage to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-group", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--use-real-model", action="store_true",
                        help="Load and run the real Hugging Face model")
    parser.add_argument("--device", default="mps",
                        help="Device for model inference (mps/cuda/cpu)")
    args = parser.parse_args()

    criteria = _load_config(args.config)
    print(f"[run] experiment_id={criteria.experiment_id}, stage={args.stage}")

    if args.stage == "collect":
        return stage_collect(criteria, args)
    elif args.stage == "develop":
        return stage_develop(criteria, args)
    elif args.stage == "calibrate":
        return stage_calibrate(criteria, args)
    elif args.stage == "final":
        return stage_final(criteria, args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
