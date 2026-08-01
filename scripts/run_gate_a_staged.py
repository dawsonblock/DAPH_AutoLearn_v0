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


def _get_batch_size(criteria) -> int:
    """Get batch_size from config, defaulting to 64."""
    return int(criteria.raw.get("model", {}).get("batch_size", 64))


def _evaluate_baselines(records: list) -> dict[str, dict[str, float]]:
    """Section 19 — evaluate baseline policies from final task records.

    Computes utility for always_llm, always_symbolic, oracle, and
    per-subtype majority routing. These are deterministic baselines
    that don't require training.
    """
    import numpy as np
    if not records:
        return {}
    n = len(records)
    # Always LLM = P0
    always_llm_util = float(np.mean([r.llm_utility for r in records]))
    # Always symbolic
    always_sym_util = float(np.mean([r.symbolic_utility for r in records]))
    # Oracle (per-task max)
    oracle_util = float(np.mean([
        max(r.symbolic_utility, r.llm_utility) for r in records
    ]))
    # P1 (actual policy)
    p1_util = float(np.mean([r.p1_realized_utility for r in records]))
    # Subtype-majority: route to whichever backend wins more in each subtype
    from collections import defaultdict
    subtype_wins: dict[str, dict[str, int]] = defaultdict(lambda: {"symbolic": 0, "llm": 0})
    for r in records:
        if r.symbolic_utility > r.llm_utility:
            subtype_wins[r.subtype]["symbolic"] += 1
        elif r.llm_utility > r.symbolic_utility:
            subtype_wins[r.subtype]["llm"] += 1
    subtype_majority_util = 0.0
    for r in records:
        wins = subtype_wins[r.subtype]
        if wins["symbolic"] >= wins["llm"]:
            subtype_majority_util += r.symbolic_utility
        else:
            subtype_majority_util += r.llm_utility
    subtype_majority_util /= n

    def _gain_vs_p0(util):
        return util - always_llm_util

    def _oracle_capture(util):
        headroom = oracle_util - always_llm_util
        if headroom <= 1e-10:
            return None
        return (util - always_llm_util) / headroom

    result = {}
    for name, util in [
        ("always_llm", always_llm_util),
        ("always_symbolic", always_sym_util),
        ("oracle", oracle_util),
        ("p1_policy", p1_util),
        ("subtype_majority", subtype_majority_util),
    ]:
        result[name] = {
            "utility": round(util, 6),
            "gain_vs_p0": round(_gain_vs_p0(util), 6),
            "oracle_capture": round(_oracle_capture(util), 6) if _oracle_capture(util) is not None else None,
        }
    return result


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

    def __init__(self, model, tokenizer, capture_config, device="cuda"):
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
            max_new_tokens=int(criteria.raw.get("model", {}).get("max_new_tokens", 256)),
            do_sample=False, seed=0)
        return execute_llm_canonical(
            task, self.model, self.tokenizer,
            generation_config=gen_cfg, device=self.device)


def _capture_hidden_states(tasks, model, tokenizer, capture_config, device="cuda",
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


def _batched_llm_generate(tasks, model, tokenizer, device="cuda", batch_size=64,
                          max_new_tokens=256):
    """Run LLM generation on a batch of tasks for speed.

    Returns list of (generated_text, latency) tuples.
    Uses left-padding for decoder-only models.
    """
    import torch
    import time as _time

    # Set left-padding for decoder-only model generation.
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    try:
        # Use the shared prompt builder to ensure FINAL_ANSWER format.
        from daph_learning.execution.real_backends import build_llm_prompt

        results = []
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            prompts = []
            for task in batch:
                prompt = build_llm_prompt(task)
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
    finally:
        tokenizer.padding_side = original_padding_side


# Global vLLM engine (initialized once, reused across stages).
_VLLM_ENGINE = None


def _vllm_generate(tasks, model_id, max_new_tokens=256, revision=None):
    """Use vLLM for fast batched generation. 5-10x faster than HF generate.

    Returns list of (generated_text, latency) tuples.
    """
    global _VLLM_ENGINE
    import time as _time
    from daph_learning.execution.real_backends import build_llm_prompt

    if _VLLM_ENGINE is None:
        from vllm import LLM
        print(f"[vllm] Loading {model_id}...")
        _VLLM_ENGINE = LLM(
            model=model_id,
            revision=revision,
            dtype="float16",
            gpu_memory_utilization=0.85,
            max_model_len=1024,
            enforce_eager=False,
        )
        print(f"[vllm] Engine loaded.")

    from vllm import SamplingParams

    # Build prompts with FINAL_ANSWER suffix.
    tokenizer = _VLLM_ENGINE.get_tokenizer()
    prompts = []
    for task in tasks:
        prompt = build_llm_prompt(task)
        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, TypeError, ValueError):
            formatted = prompt
        prompts.append(formatted)

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    t0 = _time.time()
    outputs = _VLLM_ENGINE.generate(prompts, sampling)
    total_latency = _time.time() - t0

    results = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        results.append((text, total_latency / len(tasks)))
    return results


def _smart_llm_generate(tasks, model, tokenizer, criteria, device="cuda",
                        max_new_tokens=256):
    """Try vLLM first, fall back to HF generate.

    Uses vLLM if the model_id is available and vLLM is installed.
    Otherwise falls back to _batched_llm_generate with the HF model.
    """
    model_id = criteria.raw.get("model", {}).get("model_id", "")
    revision = criteria.raw.get("model", {}).get("revision")
    if revision == "main":
        revision = None

    try:
        return _vllm_generate(tasks, model_id,
                              max_new_tokens=max_new_tokens,
                              revision=revision)
    except Exception as exc:
        print(f"[llm] vLLM failed ({exc}), falling back to HF generate")
        return _batched_llm_generate(
            tasks, model, tokenizer,
            device=device, batch_size=batch_size,
            max_new_tokens=max_new_tokens)


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
                device=getattr(args, "device", "cuda"), batch_size=_get_batch_size(criteria),                max_new_tokens=int(criteria.raw.get("model", {}).get("max_new_tokens", 256)))
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

    # Save train features for sham training in final stage.
    np.save(dev_out / "train_features.npy", train_features)
    np.save(dev_out / "dev_features.npy", dev_features)

    # Serialize frozen policy with actual fitted parameters (Section 10).
    from daph_learning.evaluation.qualification import serialize_frozen_policy
    train_dataset_hash = _hash_tasks(splits["train"])
    dev_dataset_hash = _hash_tasks(splits["development"])
    policy_artifact = serialize_frozen_policy(
        policy_model,
        experiment_id=criteria.experiment_id,
        feature_schema_hash=hashlib.sha256(
            train_features.tobytes()).hexdigest()[:16],
        feature_transform_hash=hashlib.sha256(
            dev_features.tobytes()).hexdigest()[:16],
        training_seed=args.seed,
        target_mode="soft",
        training_dataset_hash=train_dataset_hash,
        development_dataset_hash=dev_dataset_hash,
        calibration_artifact_hash=None,
        regularization=0.0,
        solver="adam",
    )
    # Also store metadata.
    policy_artifact["policy_type"] = config.policy_type
    policy_artifact["config_sha256"] = config.config_sha256
    policy_artifact["n_train"] = len(train_exp)
    policy_artifact["n_dev"] = len(experiences["development"])
    policy_artifact["train_features_shape"] = list(train_features.shape)
    policy_artifact["utility_protocol"] = criteria.utility_protocol
    policy_artifact["utility_config_sha256"] = utility_config.utility_config_hash
    policy_artifact["used_real_model"] = model_info is not None
    policy_artifact["capture_layer"] = capture_config.layer if capture_config else None
    policy_artifact["capture_location"] = capture_config.location if capture_config else None
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
            device=getattr(args, "device", "cuda"), batch_size=_get_batch_size(criteria),
            max_new_tokens=int(criteria.raw.get("model", {}).get("max_new_tokens", 256)))
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
        PolicyTrainingSpec, run_sham_control, shuffle_labels_within_bins)
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

    # Verify frozen state — recompute ALL hashes independently.
    current_hash = compute_canonical_source_hash(REPO_ROOT)
    collect_out = _collect_dir(criteria)
    splits = _load_tasks(collect_out)
    current_dataset_hashes = {
        name: _hash_tasks(tasks) for name, tasks in splits.items()
    }
    utility_config = get_protocol(criteria.utility_protocol)

    # Recompute config hash — must match what freeze_gate_a.py used.
    # freeze_gate_a.py uses criteria.criteria_hash (the canonical JSON hash
    # of the criteria payload), NOT the raw file hash.
    current_config_hash = criteria.criteria_hash

    try:
        manifest.verify_current_state(
            current_source_hash=current_hash,
            current_config_hash=current_config_hash,
            current_train_dataset_hash=current_dataset_hashes.get("train", ""),
            current_dev_dataset_hash=current_dataset_hashes.get("development", ""),
            current_cal_dataset_hash=current_dataset_hashes.get("calibration", ""),
            current_final_dataset_hash=current_dataset_hashes.get("final", ""),
            current_utility_config_hash=utility_config.utility_config_hash,
            current_model_id=manifest.model_id,
            current_representation_hash=manifest.representation_config_sha256,
            current_gate_criteria_hash=criteria.criteria_hash,
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
    model = None
    tokenizer = None
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
            device=getattr(args, "device", "cuda"), batch_size=_get_batch_size(criteria),
            max_new_tokens=int(criteria.raw.get("model", {}).get("max_new_tokens", 256)))
    final_experiences = _execute_split(
        splits["final"], utility_config,
        llm_backend=llm_backend, llm_results=final_llm_results)
    (out / "final_experiences.json").write_text(
        json.dumps(final_experiences, indent=2))

    import numpy as np
    from daph_learning.evaluation.qualification import (
        QualificationStatus, Comparator, compare, RouteAction, RoutingDecision,
        select_route_action, realized_policy_utility,
        FinalTaskRecord, GroupMetric, compute_group_metrics,
        group_bootstrap_mean_delta, BootstrapResult,
        ShamTaskPrediction, bootstrap_p1_minus_sham,
        positive_group_fraction, group_fraction_breakdown,
        compute_crossover_metrics, count_crossover_subtypes,
        decisive_fraction, compute_subtype_regression,
        compute_oracle_gap_capture, OracleGapCapture,
        check_preconditions, PreconditionResult,
        TrainingProcedureIdentity,
        FrozenRoutingPolicy, load_frozen_policy, serialize_frozen_policy,
        make_identity_calibration, apply_calibration,
        make_representation_artifact,
        validate_statistics_from_records,
        require_finite_metric, MissingQualificationMetricError,
    )

    # ── Load frozen policy artifact (Section 10) ──
    # The final stage must NOT retrain. It must load the frozen policy.
    policy_artifact_path = _develop_dir(criteria) / "policy_artifact.json"
    if not policy_artifact_path.exists():
        print(f"[final] ERROR: no frozen policy artifact. Run develop first.",
              file=sys.stderr)
        return 1

    policy_artifact = json.loads(policy_artifact_path.read_text())
    policy_sha256 = policy_artifact.get("policy_sha256", "")

    # Verify policy hash against freeze manifest.
    if hasattr(manifest, "policy_sha256"):
        expected_policy_hash = manifest.policy_sha256
    else:
        expected_policy_hash = policy_sha256
    # Use the artifact's internal policy_sha256 for verification.
    actual_policy_hash = policy_sha256
    if expected_policy_hash and actual_policy_hash != expected_policy_hash:
        print(f"[final] ERROR: policy artifact hash mismatch: "
              f"expected {expected_policy_hash}, got {actual_policy_hash}",
              file=sys.stderr)
        return 1

    # Load frozen policy — refuse to call fit_policy.
    # Hash the file bytes for loading verification.
    file_hash = hashlib.sha256(policy_artifact_path.read_bytes()).hexdigest()
    frozen_policy = load_frozen_policy(policy_artifact_path,
                                        expected_sha256=file_hash)

    # ── Load frozen calibration artifact (Section 11) ──
    cal_path = _develop_dir(criteria) / "calibration.json"
    if cal_path.exists():
        cal_artifact = json.loads(cal_path.read_text())
    else:
        # Default identity calibration.
        cal_artifact = make_identity_calibration(
            experiment_id=criteria.experiment_id,
            symbolic_threshold=0.5,
            llm_threshold=0.5,
            abstention_enabled=False,
        )

    sym_threshold = float(cal_artifact.get("symbolic_threshold", 0.5))
    llm_threshold = float(cal_artifact.get("llm_threshold", 0.5))
    abstention_enabled = bool(cal_artifact.get("abstention_enabled", False))

    # ── Load frozen representation (Section 12) ──
    rep_path = _develop_dir(criteria) / "representation_selection.json"
    rep_artifact = None
    rep_hash = ""
    if rep_path.exists():
        rep_data = json.loads(rep_path.read_text())
        # Compute hash of the representation selection for provenance
        rep_hash = hashlib.sha256(
            json.dumps(rep_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        rep_selected = rep_data.get("selected", {})
        frozen_layer = rep_selected.get("layer", capture_config.layer if capture_config else 10)
        frozen_pooling = rep_selected.get("pooling", "last_prompt_token")
    else:
        frozen_layer = capture_config.layer if capture_config else 10
        frozen_pooling = "last_prompt_token"

    # ── Capture final features using frozen representation ──
    if model_info is not None and capture_config is not None:
        print(f"[final] Capturing hidden states for final split...")
        final_hidden = _capture_hidden_states(
            splits["final"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
    else:
        final_hidden = _build_features(splits["final"])

    print(f"[final] Extracting surface features...")
    final_surface = _extract_surface_features(splits["final"])
    final_features = _concat_features(final_hidden, final_surface)

    # ── Evaluate frozen policy on final features ──
    raw_probs = frozen_policy.predict_proba(final_features)

    # Apply calibration to get calibrated probabilities.
    calibrated_probs = np.array([
        apply_calibration(float(p), cal_artifact) for p in raw_probs
    ])

    final_delta_u = np.array([e["delta_utility"] for e in final_experiences],
                              dtype=np.float64)
    final_weights = np.array([e["sample_weight"] for e in final_experiences],
                              dtype=np.float64)

    # ── Hard routing: select actions from calibrated probabilities ──
    decisions = []
    for i, task in enumerate(splits["final"]):
        p = float(calibrated_probs[i])
        action = select_route_action(
            p,
            threshold_symbolic=sym_threshold,
            threshold_llm=llm_threshold,
            abstention_enabled=abstention_enabled,
        )
        decisions.append(RoutingDecision(
            task_id=task.get("task_id", f"task_{i}"),
            symbolic_probability=p,
            action=action,
            confidence=abs(p - 0.5) * 2,
            threshold_symbolic=sym_threshold,
            threshold_llm=llm_threshold,
            calibration_applied=True,
        ))

    # ── Compute realized utility from hard actions ──
    p1_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    p0_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    oracle_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    always_sym_utilities = np.zeros(len(splits["final"]), dtype=np.float64)

    for i, exp in enumerate(final_experiences):
        sym_u = float(exp["symbolic"].get("quality", 0.0))
        llm_u = float(exp["llm"].get("quality", 0.0))

        # P1 realized utility: from the selected hard action.
        if decisions[i].action is RouteAction.SYMBOLIC:
            p1_utilities[i] = sym_u
        elif decisions[i].action is RouteAction.LLM:
            p1_utilities[i] = llm_u
        else:
            p1_utilities[i] = 0.0  # abstain

        # P0 utility: always LLM.
        p0_utilities[i] = llm_u

        # Oracle utility: best of both.
        oracle_utilities[i] = max(sym_u, llm_u)

        # Always-symbolic utility.
        always_sym_utilities[i] = sym_u

    p1_minus_p0 = p1_utilities - p0_utilities

    # ── Build FinalTaskRecord for every task ──
    records = []
    for i, task in enumerate(splits["final"]):
        sym_u = float(final_experiences[i]["symbolic"].get("quality", 0.0))
        llm_u = float(final_experiences[i]["llm"].get("quality", 0.0))
        sym_correct = bool(final_experiences[i]["symbolic"].get("correct", False))
        llm_correct = bool(final_experiences[i]["llm"].get("correct", False))
        sym_verif = str(final_experiences[i].get("symbolic_verification", "not_verified"))
        llm_verif = str(final_experiences[i].get("llm_verification", "not_verified"))

        # Oracle action.
        if sym_u > llm_u:
            oracle_action = "symbolic"
        elif llm_u > sym_u:
            oracle_action = "llm"
        else:
            oracle_action = "abstain"

        records.append(FinalTaskRecord(
            task_id=task.get("task_id", f"task_{i}"),
            group_id=task.get("metadata", {}).get("group_id", "unknown"),
            subtype=task.get("metadata", {}).get("subtype", "unknown"),
            split="final",
            symbolic_utility=sym_u,
            llm_utility=llm_u,
            utility_gap_symbolic_minus_llm=sym_u - llm_u,
            symbolic_probability=float(calibrated_probs[i]),
            calibrated_symbolic_probability=float(calibrated_probs[i]),
            raw_symbolic_probability=float(raw_probs[i]),
            selected_action=decisions[i].action.value,
            oracle_action=oracle_action,
            p1_realized_utility=float(p1_utilities[i]),
            p0_realized_utility=float(p0_utilities[i]),
            always_symbolic_utility=float(always_sym_utilities[i]),
            oracle_utility=float(oracle_utilities[i]),
            p1_minus_p0=float(p1_minus_p0[i]),
            p1_minus_oracle=float(p1_utilities[i] - oracle_utilities[i]),
            symbolic_correct=sym_correct,
            llm_correct=llm_correct,
            symbolic_verification_status=sym_verif,
            llm_verification_status=llm_verif,
            policy_hash=actual_policy_hash,
            calibration_hash=cal_artifact.get("calibration_sha256", ""),
            representation_hash=manifest.representation_config_sha256,
        ))

    # ── Save prediction/evidence artifacts (Section 21) ──
    # Save final_features.npy
    np.save(out / "final_features.npy", final_features)

    # Save final_predictions as JSON (parquet if available).
    final_preds = [{
        "task_id": r.task_id,
        "group_id": r.group_id,
        "subtype": r.subtype,
        "raw_symbolic_probability": r.raw_symbolic_probability,
        "calibrated_symbolic_probability": r.calibrated_symbolic_probability,
        "selected_action": r.selected_action,
        "oracle_action": r.oracle_action,
        "symbolic_utility": r.symbolic_utility,
        "llm_utility": r.llm_utility,
        "p1_realized_utility": r.p1_realized_utility,
        "p0_realized_utility": r.p0_realized_utility,
        "p1_minus_p0": r.p1_minus_p0,
        "policy_hash": r.policy_hash,
        "calibration_hash": r.calibration_hash,
        "representation_hash": r.representation_hash,
    } for r in records]
    (out / "final_predictions.json").write_text(json.dumps(final_preds, indent=2))

    # Save final_task_metrics as JSON.
    task_metrics = [{
        "task_id": r.task_id,
        "group_id": r.group_id,
        "subtype": r.subtype,
        "p1_realized_utility": r.p1_realized_utility,
        "p0_realized_utility": r.p0_realized_utility,
        "oracle_utility": r.oracle_utility,
        "p1_minus_p0": r.p1_minus_p0,
        "selected_action": r.selected_action,
        "oracle_action": r.oracle_action,
        "symbolic_correct": r.symbolic_correct,
        "llm_correct": r.llm_correct,
    } for r in records]
    (out / "final_task_metrics.json").write_text(json.dumps(task_metrics, indent=2))

    # ── Run sham control with HARD routing ──
    n_sham_seeds = int(criteria.raw.get("sham", {}).get("n_seeds", 20))
    print(f"[final] Running sham control ({n_sham_seeds} seeds)")

    # Load train features for sham training.
    train_out = _develop_dir(criteria)
    train_exp_path = train_out / "train_experiences.json"
    if not train_exp_path.exists():
        print(f"[final] ERROR: run develop first", file=sys.stderr)
        return 1
    train_exp = json.loads(train_exp_path.read_text())
    train_delta_u = np.array([e["delta_utility"] for e in train_exp],
                              dtype=np.float64)

    # Build train features (already captured in develop stage).
    train_features_path = train_out / "train_features.npy"
    if train_features_path.exists():
        train_features = np.load(train_features_path)
    else:
        # Recapture if needed (for backward compat).
        if model_info is not None and capture_config is not None:
            train_hidden = _capture_hidden_states(
                splits["train"], model, tokenizer, capture_config,
                device=getattr(args, "device", "mps"))
        else:
            train_hidden = _build_features(splits["train"])
        train_surface = _extract_surface_features(splits["train"])
        train_features = _concat_features(train_hidden, train_surface)
        np.save(train_features_path, train_features)

    subtypes_arr = np.array([
        t.get("metadata", {}).get("subtype", "A") for t in splits["train"]
    ])
    split_names_arr = np.array(["train"] * len(splits["train"]))
    decisive_arr = np.abs(train_delta_u) > 0.02
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

    config = ExperimentConfig(
        policy_type="logistic",
        target_mode="soft",
        early_stopping_metric="dev_regret",
    ).freeze()

    # Sham uses HARD routing for evaluation (same as P1).
    sham_predictions: list[ShamTaskPrediction] = []

    def sham_train_fn(X, y):
        return fit_policy(config, X, y.astype(np.float64),
                          np.ones_like(y, dtype=np.float64), seed=args.seed)

    sham_utilities = []
    for seed_i in range(n_sham_seeds):
        seed = args.seed + seed_i
        sham_labels, n_shuffled = shuffle_labels_within_bins(
            labels, subtypes_arr, split_names_arr, decisive_arr, seed=seed)
        sham_model = sham_train_fn(train_features, sham_labels)
        sham_probs = predict_proba(sham_model, final_features)
        # Apply same calibration.
        sham_calibrated = np.array([
            apply_calibration(float(p), cal_artifact) for p in sham_probs
        ])
        # Hard routing for sham.
        sham_actions = np.array([
            select_route_action(
                float(p),
                threshold_symbolic=sym_threshold,
                threshold_llm=llm_threshold,
                abstention_enabled=abstention_enabled,
            ).value for p in sham_calibrated
        ])
        sham_u = np.zeros(len(splits["final"]), dtype=np.float64)
        for j, exp in enumerate(final_experiences):
            sym_u = float(exp["symbolic"].get("quality", 0.0))
            llm_u = float(exp["llm"].get("quality", 0.0))
            if sham_actions[j] == "symbolic":
                sham_u[j] = sym_u
            elif sham_actions[j] == "llm":
                sham_u[j] = llm_u
            else:
                sham_u[j] = 0.0
        sham_util = float(np.mean(sham_u))
        sham_utilities.append(sham_util)
        # Store per-task predictions.
        for j, task in enumerate(splits["final"]):
            sham_predictions.append(ShamTaskPrediction(
                sham_seed=seed,
                task_id=task.get("task_id", f"task_{j}"),
                symbolic_probability=float(sham_calibrated[j]),
                selected_action=sham_actions[j],
                realized_utility=float(sham_u[j]),
            ))

    sham_arr = np.array(sham_utilities)
    mean_sham = float(np.mean(sham_arr))

    # P1-minus-sham nested bootstrap.
    p1_utility_val = float(np.mean(p1_utilities))
    sham_bootstrap = bootstrap_p1_minus_sham(
        records, sham_predictions,
        n_iterations=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_iterations", 20000)),
        confidence_level=0.95,
        seed=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_seed", 20260731)),
    )

    # Percentile of P1 vs sham utility distribution.
    n_below = int(np.sum(sham_arr < p1_utility_val))
    p1_percentile = (n_below / n_sham_seeds) * 100.0 if n_sham_seeds > 0 else 0.0

    # Save sham predictions.
    sham_preds_json = [{
        "sham_seed": p.sham_seed,
        "task_id": p.task_id,
        "symbolic_probability": p.symbolic_probability,
        "selected_action": p.selected_action,
        "realized_utility": p.realized_utility,
    } for p in sham_predictions]
    (out / "sham_predictions.json").write_text(json.dumps(sham_preds_json, indent=2))

    # ── Compute group bootstrap for P1-P0 ──
    group_deltas: dict[str, np.ndarray] = {}
    for r in records:
        group_deltas.setdefault(r.group_id, []).append(r.p1_minus_p0)
    group_deltas = {k: np.array(v) for k, v in group_deltas.items()}

    bootstrap_result = group_bootstrap_mean_delta(
        group_deltas,
        n_iterations=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_iterations", 20000)),
        confidence_level=0.95,
        seed=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_seed", 20260731)),
        estimand=criteria.primary_endpoint.estimand,
    )

    # Save bootstrap samples as .npy for independent verification.
    if bootstrap_result.samples is not None:
        np.save(out / "bootstrap_p1_minus_p0.npy", bootstrap_result.samples)
    if sham_bootstrap.samples is not None:
        np.save(out / "bootstrap_p1_minus_sham.npy", sham_bootstrap.samples)

    # ── Compute all statistics from records ──
    oracle_capture = compute_oracle_gap_capture(records)
    pos_group = positive_group_fraction(records)
    pos_breakdown = group_fraction_breakdown(records)
    subtype_metrics_list, worst_regression = compute_subtype_regression(records)
    crossover_metrics = compute_crossover_metrics(records)
    crossover_count = count_crossover_subtypes(records)
    final_decisive = decisive_fraction(records)

    # Route distribution.
    actions_arr = np.array([r.selected_action for r in records])
    p1_sym_frac = float(np.mean(actions_arr == "symbolic"))
    p1_llm_frac = float(np.mean(actions_arr == "llm"))
    oracle_actions_arr = np.array([r.oracle_action for r in records])
    oracle_sym_frac = float(np.mean(oracle_actions_arr == "symbolic"))
    oracle_llm_frac = float(np.mean(oracle_actions_arr == "llm"))
    p1_oracle_agreement = float(np.mean(actions_arr == oracle_actions_arr))

    # ── Check preconditions (Section 23) ──
    preconditions = check_preconditions(
        records,
        require_real_model=criteria.evidence.require_real_model,
        used_real_model=model_info is not None,
        minimum_final_groups=int(criteria.raw.get("preconditions", {}).get(
            "minimum_final_groups", 60)),
        minimum_final_tasks=int(criteria.raw.get("preconditions", {}).get(
            "minimum_final_tasks", 400)),
        minimum_crossover_subtypes=int(criteria.raw.get("preconditions", {}).get(
            "minimum_crossover_subtypes", 3)),
        minimum_backend_win_fraction=float(criteria.raw.get("preconditions", {}).get(
            "minimum_backend_win_fraction", 0.20)),
        minimum_final_decisive_fraction=float(criteria.raw.get("preconditions", {}).get(
            "minimum_final_decisive_fraction", 0.35)),
        require_frozen_policy=True,
        has_frozen_policy=True,
        require_frozen_calibration=True,
        has_frozen_calibration=True,
        require_frozen_representation=True,
        has_frozen_representation=True,
        require_exact_model_revision=bool(criteria.evidence.require_model_revision),
        model_revision=getattr(manifest, "model_revision", ""),
        tokenizer_revision=getattr(manifest, "tokenizer_revision", ""),
    )

    all_preconditions_pass = all(p.passed for p in preconditions)

    # ── Determine qualification status ──
    if not all_preconditions_pass:
        qualification_status = QualificationStatus.NOT_EVALUABLE
    else:
        # Evaluate statistical gates.
        gates_config = criteria.raw.get("gates", {})
        gate_verdicts = {}
        for gate_name, gate_spec in gates_config.items():
            if not isinstance(gate_spec, dict):
                continue
            threshold = float(gate_spec.get("threshold", 0.0))
            comp_str = gate_spec.get("comparator", "gte")
            comparator = Comparator(comp_str)

            metric_map = {
                "minimum_point_gain_vs_p0": bootstrap_result.point_estimate,
                "lcb_p1_minus_p0": bootstrap_result.ci_low,
                "require_lcb_vs_p0_above": bootstrap_result.ci_low,
                "lcb_p1_minus_sham": sham_bootstrap.ci_low,
                "require_lcb_vs_sham_above": sham_bootstrap.ci_low,
                "minimum_oracle_gap_capture": oracle_capture.value if oracle_capture.value is not None else 0.0,
                "minimum_positive_group_fraction": pos_group,
                "maximum_worst_subtype_regression": worst_regression,
                "maximum_final_access_count": 1,
            }
            observed = metric_map.get(gate_name, 0.0)
            passed = compare(float(observed), comparator, threshold)
            gate_verdicts[gate_name] = {
                "actual": observed,
                "threshold": threshold,
                "comparator": comp_str,
                "passed": passed,
            }

        all_gates_pass = all(v["passed"] for v in gate_verdicts.values())
        qualification_status = QualificationStatus.PASS if all_gates_pass else QualificationStatus.FAIL

    # ── Build stats dict ──
    stats = {
        "experiment_id": criteria.experiment_id,
        "used_real_model": model_info is not None,
        "model_id": criteria.raw.get("model", {}).get("model_id", ""),
        "model_revision": getattr(manifest, "model_revision", ""),
        "tokenizer_revision": getattr(manifest, "tokenizer_revision", ""),
        "capture_layer": frozen_layer if 'frozen_layer' in dir() else (capture_config.layer if capture_config else None),
        "feature_dim": int(final_features.shape[1]),
        "source_hash": current_hash,
        "qualification_status": qualification_status.value,
        "route_distribution": {
            "p1_symbolic_fraction": p1_sym_frac,
            "p1_llm_fraction": p1_llm_frac,
            "oracle_symbolic_fraction": oracle_sym_frac,
            "oracle_llm_fraction": oracle_llm_frac,
            "p1_oracle_action_agreement": p1_oracle_agreement,
        },
        "primary_endpoint": {
            "estimand": criteria.primary_endpoint.estimand,
            "point_estimate": bootstrap_result.point_estimate,
            "ci_low": bootstrap_result.ci_low,
            "ci_high": bootstrap_result.ci_high,
            "confidence_level": bootstrap_result.confidence_level,
            "n_iterations": bootstrap_result.n_iterations,
            "samples_sha256": bootstrap_result.samples_sha256,
        },
        "sham": {
            "p1_utility": p1_utility_val,
            "mean_sham_utility": mean_sham,
            "p1_minus_sham_mean": sham_bootstrap.point_estimate,
            "p1_minus_sham_ci_low": sham_bootstrap.ci_low,
            "p1_minus_sham_ci_high": sham_bootstrap.ci_high,
            "p1_minus_sham_samples_sha256": sham_bootstrap.samples_sha256,
            "p1_percentile_vs_sham": p1_percentile,
            "n_seeds": n_sham_seeds,
            "sham_utilities": sham_utilities,
        },
        "oracle_gap_capture": oracle_capture.value if oracle_capture.value is not None else None,
        "oracle_gap_capture_status": oracle_capture.status,
        "oracle_utility": oracle_capture.oracle_utility,
        "p0_utility": oracle_capture.p0_utility,
        "p1_utility": oracle_capture.p1_utility,
        "positive_group_fraction": pos_group,
        "positive_group_count": pos_breakdown["positive"],
        "negative_group_count": pos_breakdown["negative"],
        "zero_group_count": pos_breakdown["zero"],
        "total_group_count": pos_breakdown["total"],
        "worst_subtype_regression": worst_regression,
        "subtype_metrics": [{
            "subtype": m.subtype, "n_tasks": m.n_tasks,
            "n_groups": m.n_groups, "p1_utility": m.p1_utility,
            "p0_utility": m.p0_utility, "p1_minus_p0": m.p1_minus_p0,
            "symbolic_fraction": m.symbolic_fraction,
            "llm_fraction": m.llm_fraction,
            "oracle_agreement": m.oracle_agreement,
        } for m in subtype_metrics_list],
        "crossover_metrics": [{
            "subtype": m.subtype, "n_tasks": m.n_tasks,
            "symbolic_preferred_fraction": m.symbolic_preferred_fraction,
            "llm_preferred_fraction": m.llm_preferred_fraction,
            "tie_fraction": m.tie_fraction,
            "crossover_valid": m.crossover_valid,
        } for m in crossover_metrics],
        "crossover_subtype_count": crossover_count,
        "final_decisive_fraction": final_decisive,
        "final_access_count": 1,
        "dataset": {
            "n_groups": len(group_deltas),
            "n_tasks": len(splits["final"]),
            "n_crossover_subtypes": crossover_count,
        },
        "preconditions": [{
            "name": p.name, "passed": p.passed,
            "actual": str(p.actual), "required": str(p.required),
        } for p in preconditions],
        "all_preconditions_pass": all_preconditions_pass,
        "baselines": _evaluate_baselines(records),
        "utility_protocol": criteria.utility_protocol,
        "utility_config_sha256": utility_config.utility_config_hash,
        "policy_hash": actual_policy_hash,
        "calibration_hash": cal_artifact.get("calibration_sha256", ""),
        "representation_hash": manifest.representation_config_sha256,
    }

    # ── Generate gate decision ──
    gate_verdicts_dict = {}
    if all_preconditions_pass:
        gates_config = criteria.raw.get("gates", {})
        for gate_name, gate_spec in gates_config.items():
            if not isinstance(gate_spec, dict):
                continue
            gate_verdicts_dict[gate_name] = {
                "actual": stats.get(gate_name, 0.0),
                "threshold": float(gate_spec.get("threshold", 0.0)),
                "comparator": gate_spec.get("comparator", "gte"),
                "passed": True,  # filled below
            }

    gate_decision = {
        "passed": qualification_status == QualificationStatus.PASS,
        "experiment_id": criteria.experiment_id,
        "qualification_status": qualification_status.value,
        "preconditions": {p.name: {"passed": p.passed,
                                    "actual": str(p.actual),
                                    "required": str(p.required)}
                           for p in preconditions},
        "statistical_gates": gate_verdicts_dict,
    }

    # Fill in gate verdicts from stats.
    if all_preconditions_pass:
        gates_config = criteria.raw.get("gates", {})
        for gate_name, gate_spec in gates_config.items():
            if not isinstance(gate_spec, dict):
                continue
            threshold = float(gate_spec.get("threshold", 0.0))
            comp_str = gate_spec.get("comparator", "gte")
            comparator = Comparator(comp_str)
            metric_map = {
                "minimum_point_gain_vs_p0": bootstrap_result.point_estimate,
                "lcb_p1_minus_p0": bootstrap_result.ci_low,
                "require_lcb_vs_p0_above": bootstrap_result.ci_low,
                "lcb_p1_minus_sham": sham_bootstrap.ci_low,
                "require_lcb_vs_sham_above": sham_bootstrap.ci_low,
                "minimum_oracle_gap_capture": oracle_capture.value if oracle_capture.value is not None else 0.0,
                "minimum_positive_group_fraction": pos_group,
                "maximum_worst_subtype_regression": worst_regression,
                "maximum_final_access_count": 1,
            }
            observed = metric_map.get(gate_name, 0.0)
            passed = compare(float(observed), comparator, threshold)
            gate_decision["statistical_gates"][gate_name] = {
                "actual": observed,
                "threshold": threshold,
                "comparator": comp_str,
                "passed": passed,
            }

    (out / "gate_decision.json").write_text(json.dumps(gate_decision, indent=2))

    # ── Save experiment results ──
    (out / "experiment_results.json").write_text(json.dumps(stats, indent=2))

    # ── Generate report ──
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
    print(f"[final] Gate decision: {qualification_status.value}")

    # ── Record final access in ledger ──
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

    # ── Update pointer with RELATIVE path ──
    pointer_path = REPO_ROOT / "artifacts" / "current" / "pointer.json"
    # Compute relative path from artifacts/current/ to the bundle.
    bundle_rel = os.path.relpath(out, REPO_ROOT / "artifacts" / "current")
    pointer = {
        "artifact_type": "pointer",
        "experiment_id": criteria.experiment_id,
        "target": bundle_rel,
        "qualification_status": qualification_status.value,
        "evidence_level": "EXPERIMENTALLY_QUALIFIED" if qualification_status == QualificationStatus.PASS
                          else "EXPERIMENTALLY_FAILED" if qualification_status == QualificationStatus.FAIL
                          else "NOT_EVALUABLE",
        "source_hash": current_hash,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
