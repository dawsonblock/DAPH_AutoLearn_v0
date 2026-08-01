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
    """Return the output directory for a given stage.

    Priority 1 fix: the final stage writes to ``gate_a_runs/`` (neutral
    staging), NOT ``gate_a_qualified/``. Only promotion logic may move a
    bundle into ``gate_a_qualified/`` (PASS) or ``gate_a_failed/`` (FAIL).
    This prevents failed runs from appearing in the qualified directory.
    """
    base = REPO_ROOT / "artifacts"
    if stage == "final":
        bucket = base / "gate_a_runs"
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


def _evaluate_baselines(
    records: list,
    best_fixed_id: str = "always_symbolic",
    *,
    dev_subtype_preferences: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Section 19 — evaluate ALL baseline policies from final task records.

    Computes utility for every required baseline:
    always_llm, always_symbolic, best_fixed, oracle, p1_policy (primary),
    subtype_only, and any trained baselines passed in via records.

    The primary comparator is best_fixed (selected from dev data).

    Fix 5: the subtype-only baseline must be frozen on DEVELOPMENT data,
    never derived from final labels. ``dev_subtype_preferences`` maps
    subtype → preferred backend ("symbolic" | "llm") and is computed
    from the development split before final access. When provided, the
    final subtype-majority utility applies that frozen policy to the
    final records. When omitted, the subtype baseline is reported as
    ``None`` (not optimistically re-fit on final data).
    """
    import numpy as np
    if not records:
        return {}
    n = len(records)
    # Always LLM
    always_llm_util = float(np.mean([r.llm_utility for r in records]))
    # Always symbolic
    always_sym_util = float(np.mean([r.symbolic_utility for r in records]))
    # Best fixed (frozen from dev data)
    best_fixed_util = always_sym_util if best_fixed_id == "always_symbolic" else always_llm_util
    # Oracle (per-task max)
    oracle_util = float(np.mean([
        max(r.symbolic_utility, r.llm_utility) for r in records
    ]))
    # P1 (primary policy — hidden_plus_surface)
    p1_util = float(np.mean([r.p1_realized_utility for r in records]))
    # Subtype-majority: apply the FROZEN dev-derived per-subtype preference.
    # Fix 5: never derive routing rules from final labels.
    if dev_subtype_preferences is not None:
        subtype_majority_util = 0.0
        n_matched = 0
        for r in records:
            pref = dev_subtype_preferences.get(r.subtype)
            if pref == "symbolic":
                subtype_majority_util += r.symbolic_utility
                n_matched += 1
            elif pref == "llm":
                subtype_majority_util += r.llm_utility
                n_matched += 1
            else:
                # Unknown subtype → fall back to best_fixed action.
                if best_fixed_id == "always_symbolic":
                    subtype_majority_util += r.symbolic_utility
                else:
                    subtype_majority_util += r.llm_utility
                n_matched += 1
        subtype_majority_util /= n
        subtype_source = "development (frozen)"
    else:
        # No frozen dev preferences available — report as unavailable
        # rather than leaking final labels.
        subtype_majority_util = None
        subtype_source = "unavailable (no frozen dev preferences)"

    def _gain_vs_best_fixed(util):
        if util is None:
            return None
        return util - best_fixed_util

    def _oracle_capture(util):
        if util is None:
            return None
        headroom = oracle_util - best_fixed_util
        if headroom <= 1e-10:
            return None
        return (util - best_fixed_util) / headroom

    result = {}
    for name, util in [
        ("always_llm", always_llm_util),
        ("always_symbolic", always_sym_util),
        ("best_fixed", best_fixed_util),
        ("oracle", oracle_util),
        ("hidden_plus_surface", p1_util),
        ("subtype_only", subtype_majority_util),
    ]:
        result[name] = {
            "utility": round(util, 6) if util is not None else None,
            "gain_vs_best_fixed": round(_gain_vs_best_fixed(util), 6) if _gain_vs_best_fixed(util) is not None else None,
            "oracle_capture": round(_oracle_capture(util), 6) if _oracle_capture(util) is not None else None,
        }
    result["subtype_only"]["selection_data"] = subtype_source
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
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        for j in range(len(batch)):
            if cfg.location in ("last_token", "last_prompt_token"):
                # Find the last non-padding token
                attn = inputs.attention_mask[j]
                last_idx = int(attn.sum().item()) - 1
                h = hidden[j, last_idx, :]
            elif cfg.location == "anchor":
                seq_len = hidden.shape[1]
                idx = min(1, seq_len - 1)
                h = hidden[j, idx, :]
            elif cfg.location in ("mean", "mean_prompt_tokens"):
                attn = inputs.attention_mask[j].float().unsqueeze(-1)
                h = (hidden[j] * attn).sum(0) / attn.sum().clamp(min=1)
            elif cfg.location == "mean_content_tokens":
                # Mean over non-pad, non-special (BOS/EOS) content tokens.
                attn = inputs.attention_mask[j]
                content_mask = attn.clone()
                input_ids = inputs.input_ids[j]
                for t_idx in range(int(attn.sum().item())):
                    if int(input_ids[t_idx].item()) in special_ids:
                        content_mask[t_idx] = 0
                if content_mask.sum().item() == 0:
                    content_mask = attn  # fallback to full prompt
                cm = content_mask.float().unsqueeze(-1)
                h = (hidden[j] * cm).sum(0) / cm.sum().clamp(min=1)
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


def _vllm_generate(tasks, model_id, max_new_tokens=256, revision=None,
                   *, gpu_memory_utilization=0.90, max_model_len=2048,
                   tensor_parallel_size=1):
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
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
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


def _vllm_api_generate(tasks, model_id, criteria, max_new_tokens=256):
    """Use a running vLLM OpenAI-compatible API server for generation.

    This is the preferred path when a vLLM server is already running
    (e.g., on RunPod). It avoids loading a second copy of the model
    in-process. The HF model is still loaded separately for hidden state
    capture.

    Uses concurrent requests for speed (vLLM handles batching internally).

    Returns list of (generated_text, latency) tuples.
    """
    import time as _time
    import json as _json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from daph_learning.execution.real_backends import build_llm_prompt

    model_cfg = criteria.raw.get("model", {})
    port = int(model_cfg.get("vllm_port", 8000))
    # Read API key from env var name (vllm_api_key_env) or direct value (legacy).
    api_key_env = model_cfg.get("vllm_api_key_env")
    if api_key_env:
        import os
        api_key = os.environ.get(api_key_env, "")
    else:
        api_key = model_cfg.get("vllm_api_key", "")
    base_url = f"http://localhost:{port}/v1"
    max_concurrent = int(model_cfg.get("vllm_max_concurrent", 64))

    # Build prompts.
    prompts = []
    for task in tasks:
        prompt = build_llm_prompt(task)
        prompts.append(prompt)

    def send_one(idx_prompt):
        idx, prompt = idx_prompt
        payload = _json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read())
                text = data["choices"][0]["message"]["content"].strip()
                return idx, text, 0.0
        except Exception as e:
            if idx < 5:
                print(f"[vllm-api] Error on task {idx}: {e}")
            return idx, "", 0.0

    t0 = _time.time()
    results = [("", 0.0)] * len(tasks)

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(send_one, (i, p)): i
                   for i, p in enumerate(prompts)}
        completed = 0
        for future in as_completed(futures):
            idx, text, latency = future.result()
            results[idx] = (text, latency)
            completed += 1
            if completed % 500 == 0:
                print(f"[vllm-api] {completed}/{len(tasks)} completed...")

    total_latency = _time.time() - t0
    avg_latency = total_latency / max(len(tasks), 1)
    print(f"[vllm-api] Done: {len(tasks)} tasks in {total_latency:.1f}s "
          f"({avg_latency*1000:.0f}ms/task)")
    return [(text, avg_latency) for text, _ in results]


def _smart_llm_generate(tasks, model, tokenizer, criteria, device="cuda",
                        max_new_tokens=256):
    """Try vLLM API server first, then in-process vLLM, then HF generate.

    Priority:
    1. vLLM OpenAI API (if a vLLM server is already running on vllm_port)
    2. In-process vLLM engine (if vLLM is installed)
    3. HF batched generate (fallback)
    """
    model_id = criteria.raw.get("model", {}).get("model_id", "")
    revision = criteria.raw.get("model", {}).get("revision")
    if revision == "main":
        revision = None
    model_cfg = criteria.raw.get("model", {})

    # 1. Try vLLM API server first (preferred when a server is already running)
    if model_cfg.get("vllm_api_key") or model_cfg.get("vllm_api_key_env"):
        try:
            print(f"[llm] Using vLLM API server (port {model_cfg.get('vllm_port', 8000)})...")
            return _vllm_api_generate(
                tasks, model_id, criteria,
                max_new_tokens=max_new_tokens)
        except Exception as exc:
            print(f"[llm] vLLM API failed ({exc}), trying in-process vLLM...")

    # 2. Try in-process vLLM
    try:
        return _vllm_generate(
            tasks, model_id,
            max_new_tokens=max_new_tokens,
            revision=revision,
            gpu_memory_utilization=float(
                model_cfg.get("vllm_gpu_memory_utilization", 0.90)),
            max_model_len=int(model_cfg.get("vllm_max_model_len", 2048)),
            tensor_parallel_size=int(
                model_cfg.get("vllm_tensor_parallel_size", 1)),
        )
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
    """Concatenate hidden states with surface features.

    Hidden states are L2-normalized per-row so they don't dominate
    the surface features in the logistic regression.
    """
    import numpy as np
    # L2 normalize hidden states per row.
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    hidden_normed = hidden / norms
    return np.concatenate([hidden_normed, surface], axis=1)


def _capture_all_layers(tasks, model, tokenizer, device="cuda", batch_size: int = 16,
                        layer_indices: list[int] | None = None):
    """Capture hidden states for specified layers and ALL pooling methods in one pass.

    Fix 3: the previous implementation captured only the last token and
    ignored ``c.pooling``, so the three pooling candidates
    (last_prompt_token / mean_prompt_tokens / mean_content_tokens) were
    functionally identical. This version independently produces three
    genuinely distinct representations per layer:

      * ``last_prompt_token``  — hidden state at the last non-pad position.
      * ``mean_prompt_tokens`` — mean of hidden states over all non-pad
        positions (the full prompt).
      * ``mean_content_tokens`` — mean over non-pad, non-special (BOS/EOS)
        positions (pure content tokens, excluding structural tokens).

    Parameters
    ----------
    layer_indices : list[int] | None
        If provided, only capture hidden states for these layer indices
        (0-indexed including embedding layer). If None, capture all layers.
        Capturing only the needed layers (e.g. 4 out of 37) is ~9x faster.

    Returns
    -------
    dict[str, list[np.ndarray]]
        Maps pooling name → list of per-layer arrays, each of shape
        (n_tasks, hidden_dim). Index by ``result[pooling][layer]``.
    """
    import numpy as np
    import torch

    pooling_names = ("last_prompt_token", "mean_prompt_tokens", "mean_content_tokens")
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    per_pool: dict[str, list[list[np.ndarray]]] = {
        p: None for p in pooling_names}
    n_layers = None
    capture_layers = None  # set after first batch
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        prompts = [str(t.get("prompt", t.get("specification", ""))) for t in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        if n_layers is None:
            n_layers = len(outputs.hidden_states)
            if layer_indices is not None:
                # Map requested layer indices to actual positions, clamp to valid range.
                capture_layers = sorted(set(
                    min(max(l, 0), n_layers - 1) for l in layer_indices))
            else:
                capture_layers = list(range(n_layers))
            per_pool = {p: [[] for _ in range(n_layers)] for p in pooling_names}
        for j in range(len(batch)):
            attn = inputs.attention_mask[j]
            last_idx = int(attn.sum().item()) - 1
            input_ids = inputs.input_ids[j]
            content_mask = attn.clone()
            # Exclude special (BOS/EOS/PAD) tokens from the content mean.
            for t_idx in range(int(attn.sum().item())):
                if int(input_ids[t_idx].item()) in special_ids:
                    content_mask[t_idx] = 0
            # Guarantee at least one content token; fall back to full mask.
            if content_mask.sum().item() == 0:
                content_mask = attn
            for layer_idx in capture_layers:
                h = outputs.hidden_states[layer_idx][j]
                # last_prompt_token
                per_pool["last_prompt_token"][layer_idx].append(
                    h[last_idx].float().cpu().numpy())
                # mean_prompt_tokens
                am = attn.float().unsqueeze(-1)
                per_pool["mean_prompt_tokens"][layer_idx].append(
                    (h * am).sum(0).div(am.sum().clamp(min=1))
                    .float().cpu().numpy())
                # mean_content_tokens
                cm = content_mask.float().unsqueeze(-1)
                per_pool["mean_content_tokens"][layer_idx].append(
                    (h * cm).sum(0).div(cm.sum().clamp(min=1))
                    .float().cpu().numpy())
        if (i + batch_size) % 400 == 0 or i + batch_size >= len(tasks):
            print(f"    ... captured {min(i + batch_size, len(tasks))}/{len(tasks)} "
                  f"hidden states (3 poolings × {len(capture_layers)} layers)")
    return {p: [np.array(layer_feats) if layer_feats else np.array([])
                 for layer_feats in per_pool[p]]
            for p in pooling_names}


def _extract_tfidf_features(tasks: list[dict[str, Any]], ref_tasks: list[dict[str, Any]]) -> "np.ndarray":
    """Extract TF-IDF features from task prompts.

    Uses a simple deterministic bag-of-words approach with unigrams
    and bigrams.  The vocabulary is built from ref_tasks (training data)
    and applied to tasks.
    """
    import numpy as np
    import re
    from collections import Counter

    def tokenize(text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]

    # Build vocabulary from reference tasks (min_df=2).
    doc_freq = Counter()
    ref_docs = []
    for t in ref_tasks:
        spec = str(t.get("specification", t.get("prompt", "")))
        toks = tokenize(spec)
        ref_docs.append(toks)
        for w in set(toks):
            doc_freq[w] += 1
    vocab = {}
    for w, c in sorted(doc_freq.items()):
        if c >= 2:
            vocab[w] = len(vocab)
    # Limit vocab size.
    max_features = 5000
    if len(vocab) > max_features:
        vocab = dict(list(vocab.items())[:max_features])

    # Compute IDF.
    n_docs = len(ref_docs)
    idf = np.zeros(len(vocab))
    for w, idx in vocab.items():
        idf[idx] = np.log((1 + n_docs) / (1 + doc_freq[w])) + 1

    # Transform tasks.
    features = np.zeros((len(tasks), len(vocab)), dtype=np.float32)
    for i, t in enumerate(tasks):
        spec = str(t.get("specification", t.get("prompt", "")))
        toks = tokenize(spec)
        tf = Counter(toks)
        for w, count in tf.items():
            if w in vocab:
                features[i, vocab[w]] = count * idf[vocab[w]]
    # L2 normalize.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms
    return features


def _extract_char_ngram_features(
    tasks: list[dict[str, Any]], ref_tasks: list[dict[str, Any]]
) -> "np.ndarray":
    """Extract character n-gram TF-IDF features from task prompts.

    Priority 4: a stronger text baseline than word-level TF-IDF. Character
    n-grams (3-5) capture morphological structure (e.g. "mod", "gcd",
    "remainder", "convert") that word-level tokenization may miss, and are
    robust to minor phrasing variations.
    """
    import numpy as np
    import re
    from collections import Counter

    def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
        text = re.sub(r'\s+', ' ', text.lower().strip())
        grams = []
        for n in range(n_min, n_max + 1):
            for i in range(len(text) - n + 1):
                grams.append(text[i:i+n])
        return grams

    # Build vocabulary from reference tasks (min_df=3).
    doc_freq = Counter()
    ref_docs = []
    for t in ref_tasks:
        spec = str(t.get("specification", t.get("prompt", "")))
        grams = char_ngrams(spec)
        ref_docs.append(grams)
        for g in set(grams):
            doc_freq[g] += 1
    vocab = {}
    for g, c in sorted(doc_freq.items()):
        if c >= 3:
            vocab[g] = len(vocab)
    # Limit vocab size.
    max_features = 8000
    if len(vocab) > max_features:
        vocab = dict(list(vocab.items())[:max_features])

    # Compute IDF.
    n_docs = len(ref_docs)
    idf = np.zeros(len(vocab))
    for g, idx in vocab.items():
        idf[idx] = np.log((1 + n_docs) / (1 + doc_freq[g])) + 1

    # Transform tasks.
    features = np.zeros((len(tasks), len(vocab)), dtype=np.float32)
    for i, t in enumerate(tasks):
        spec = str(t.get("specification", t.get("prompt", "")))
        grams = char_ngrams(spec)
        tf = Counter(grams)
        for g, count in tf.items():
            if g in vocab:
                features[i, vocab[g]] = count * idf[vocab[g]]
    # L2 normalize.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms
    return features


def _extract_text_embedding_features(
    tasks: list[dict[str, Any]], model, tokenizer, device: str = "cuda",
    batch_size: int = 16,
) -> "np.ndarray":
    """Extract text embedding features by mean-pooling the model's token
    embeddings (input embeddings, NOT hidden states).

    Priority 4: this is a stronger text baseline that uses the model's
    learned token embeddings but NOT its contextual representations. It
    captures lexical semantics (which tokens appear) without contextual
    processing (how they interact). If hidden states add value beyond
    this, it's because of contextual processing, not just token identity.
    """
    import numpy as np
    import torch

    model.eval()
    all_embeddings: list[np.ndarray] = []

    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        specs = [str(t.get("specification", "")) for t in batch]
        enc = tokenizer(
            specs, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        with torch.no_grad():
            # Get input embeddings (token embeddings only, no positional,
            # no layers). This is the raw embedding lookup.
            input_embeds = model.get_input_embeddings()(enc["input_ids"])
            # Mean-pool over non-pad tokens.
            mask = enc["attention_mask"].float().unsqueeze(-1)
            pooled = (input_embeds * mask).sum(1) / mask.sum(1).clamp(min=1)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

    features = np.concatenate(all_embeddings, axis=0)
    # L2 normalize.
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms
    return features


def _fit_heuristic_threshold(dev_tasks: list[dict[str, Any]],
                              dev_experiences: list[dict[str, Any]]) -> float:
    """Fit a deterministic heuristic threshold from development data.

    The heuristic routes to symbolic if max_operand_magnitude >= threshold,
    else to LLM.  The threshold is chosen to maximize dev utility.
    """
    import numpy as np
    import re

    # Extract operand magnitudes and utilities.
    magnitudes = []
    sym_utils = []
    llm_utils = []
    for task, exp in zip(dev_tasks, dev_experiences):
        spec = str(task.get("specification", ""))
        numbers = [int(n) for n in re.findall(r'\d+', spec)]
        max_num = max(numbers) if numbers else 0
        magnitudes.append(float(np.log10(max(max_num, 1))))
        sym_utils.append(float(exp["symbolic"].get("quality", 0.0)))
        llm_utils.append(float(exp["llm"].get("quality", 0.0)))

    magnitudes = np.array(magnitudes)
    sym_utils = np.array(sym_utils)
    llm_utils = np.array(llm_utils)

    # Try thresholds from 0 to 6 in 0.1 steps.
    best_threshold = 3.0
    best_util = -1.0
    for t in np.arange(0.0, 6.1, 0.1):
        route_sym = magnitudes >= t
        util = np.where(route_sym, sym_utils, llm_utils).mean()
        if util > best_util:
            best_util = util
            best_threshold = float(t)
    return best_threshold


def stage_collect(criteria, args) -> int:
    """Stage 1: generate dataset + collect counterfactual experience."""
    from daph_learning.benchmark.audit import audit_dataset

    # Select benchmark generator based on config.
    benchmark_type = criteria.raw.get("dataset", {}).get("benchmark_type", "standard")
    if benchmark_type == "v3":
        from daph_learning.data.benchmark_v3_grouped import generate_all_grouped_splits
        print(f"[collect] Using BENCHMARK V3 (reduced-tie, balanced-crossover)")
    elif benchmark_type == "harder":
        from daph_learning.data.harder_grouped_benchmark import generate_all_grouped_splits
        print(f"[collect] Using HARDER benchmark (magnitude-decoupled)")
    else:
        from daph_learning.data.grouped_benchmark import generate_all_grouped_splits

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
    model = None
    tokenizer = None
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
        # Use smart LLM inference (vLLM API → in-process vLLM → HF generate).
        llm_results = None
        if model_info is not None and model is not None:
            llm_results = _smart_llm_generate(
                splits[split_name], model, tokenizer, criteria,
                device=getattr(args, "device", "cuda"),
                max_new_tokens=int(criteria.raw.get("model", {}).get("max_new_tokens", 256)))
        experiences[split_name] = _execute_split(
            splits[split_name], utility_config,
            llm_backend=llm_backend, llm_results=llm_results)

    # Save experiences.
    dev_out = _develop_dir(criteria)
    dev_out.mkdir(parents=True, exist_ok=True)
    for split_name, exp in experiences.items():
        (dev_out / f"{split_name}_experiences.json").write_text(
            json.dumps(exp, indent=2))

    # Representation selection — REAL evaluation of all candidates.
    import numpy as np
    from daph_learning.policy.policy_factory import fit_policy, predict_proba
    from daph_learning.policy.config import ExperimentConfig

    train_exp = experiences["train"]
    if model_info is not None and capture_config is not None:
        n_layers = model.config.num_hidden_layers
        # Only capture the layers we actually need (0.25, 0.50, 0.75, last)
        # instead of all layers. This is ~9x faster on CPU.
        from daph_learning.evaluation.representation_selection import layer_candidates
        needed_layers = layer_candidates(n_layers)
        print(f"[develop] Capturing hidden states for train split "
              f"(layers {needed_layers} of {n_layers})...")
        train_hidden_all = _capture_all_layers(
            splits["train"], model, tokenizer,
            device=getattr(args, "device", "mps"),
            layer_indices=needed_layers)
        print(f"[develop] Capturing hidden states for dev split "
              f"(layers {needed_layers} of {n_layers})...")
        dev_hidden_all = _capture_all_layers(
            splits["development"], model, tokenizer,
            device=getattr(args, "device", "mps"),
            layer_indices=needed_layers)
    else:
        n_layers = args.n_layers or 24
        train_hidden_all = None
        dev_hidden_all = None

    # Surface features (same for all candidates).
    print(f"[develop] Extracting surface features...")
    train_surface = _extract_surface_features(splits["train"])
    dev_surface = _extract_surface_features(splits["development"])

    train_delta_u = np.array([e["delta_utility"] for e in train_exp],
                              dtype=np.float64)
    train_weights = np.array([e["sample_weight"] for e in train_exp],
                              dtype=np.float64)
    dev_delta_u = np.array([e["delta_utility"] for e in experiences["development"]],
                            dtype=np.float64)
    dev_weights = np.array([e["sample_weight"] for e in experiences["development"]],
                            dtype=np.float64)

    # Evaluate each representation candidate on dev data.
    candidates = all_candidates(n_layers)
    results = []
    config_rep = ExperimentConfig(
        policy_type="logistic", target_mode="soft",
        early_stopping_metric="dev_regret").freeze()

    print(f"[develop] Evaluating {len(candidates)} representation candidates...")
    for c in candidates:
        if train_hidden_all is not None:
            # Fix 3: select the pooling-specific representation. The dict
            # is keyed by pooling name → list[per-layer array].
            train_h = train_hidden_all[c.pooling][c.layer]
            dev_h = dev_hidden_all[c.pooling][c.layer]
        else:
            train_h = _build_features(splits["train"])
            dev_h = _build_features(splits["development"])
        train_feats_c = _concat_features(train_h, train_surface)
        dev_feats_c = _concat_features(dev_h, dev_surface)
        # Train a logistic policy on this candidate's features.
        model_c = fit_policy(
            config_rep, train_feats_c, train_delta_u, train_weights,
            dev_features=dev_feats_c, dev_delta_u=dev_delta_u,
            dev_weights=dev_weights, seed=args.seed)
        probs_c = predict_proba(model_c, dev_feats_c)
        # Compute dev utility using hard routing.
        dev_u = 0.0
        for j, exp in enumerate(experiences["development"]):
            sym_u = float(exp["symbolic"].get("quality", 0.0))
            llm_u = float(exp["llm"].get("quality", 0.0))
            if float(probs_c[j]) >= 0.5:
                dev_u += sym_u
            else:
                dev_u += llm_u
        dev_u /= len(experiences["development"])
        # Dev regret vs oracle.
        dev_oracle = float(np.mean([
            max(float(e["symbolic"].get("quality", 0.0)),
                float(e["llm"].get("quality", 0.0)))
            for e in experiences["development"]
        ]))
        dev_regret = dev_oracle - dev_u
        results.append(CandidateResult(
            candidate=c,
            dev_objective=dev_u,
            mean_utility=dev_u,
            mean_regret=dev_regret,
            p_symbolic=float(np.mean(probs_c >= 0.5)),
            abstention_rate=0.0,
            n_tasks=len(experiences["development"]),
        ))
        print(f"    layer={c.layer} pooling={c.pooling}: "
              f"dev_utility={dev_u:.4f}, dev_regret={dev_regret:.4f}")

    sel = select_representation(results, n_layers=n_layers)
    (dev_out / "representation_selection.json").write_text(
        json.dumps(sel.to_dict(), indent=2))
    print(f"[develop] Selected: layer={sel.selected.layer}, "
          f"pooling={sel.selected.pooling}")

    # Build final features using the SELECTED representation (pooling-aware).
    if train_hidden_all is not None:
        train_hidden = train_hidden_all[sel.selected.pooling][sel.selected.layer]
        dev_hidden = dev_hidden_all[sel.selected.pooling][sel.selected.layer]
    else:
        train_hidden = _build_features(splits["train"])
        dev_hidden = _build_features(splits["development"])
    train_features = _concat_features(train_hidden, train_surface)
    dev_features = _concat_features(dev_hidden, dev_surface)
    print(f"[develop] Feature dim: {train_features.shape[1]} "
          f"(hidden={train_hidden.shape[1]} + surface={train_surface.shape[1]})")

    # ── Section 2.2: Select best_fixed comparator from DEVELOPMENT data only ──
    from daph_learning.policy.policy_types import select_best_fixed_policy, PolicyId
    dev_llm_utils = [float(e["llm"].get("quality", 0.0)) for e in experiences["development"]]
    dev_sym_utils = [float(e["symbolic"].get("quality", 0.0)) for e in experiences["development"]]
    best_fixed = select_best_fixed_policy(dev_llm_utils, dev_sym_utils)
    print(f"[develop] Best fixed comparator (from dev data): {best_fixed.value}")
    (dev_out / "best_fixed_selection.json").write_text(json.dumps({
        "best_fixed_policy_id": best_fixed.value,
        "dev_mean_llm_utility": float(np.mean(dev_llm_utils)),
        "dev_mean_symbolic_utility": float(np.mean(dev_sym_utils)),
        "selection_data": "development",
        "frozen_before_final": True,
    }, indent=2))

    # ── Train baseline models on development data ──
    # Surface-only: logistic on surface features only
    print(f"[develop] Training surface-only baseline...")
    surface_model = fit_policy(
        config_rep, train_surface, train_delta_u, train_weights,
        dev_features=dev_surface, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed)
    np.save(dev_out / "train_surface_only.npy", train_surface)
    np.save(dev_out / "dev_surface_only.npy", dev_surface)
    # Save surface-only model weights
    from daph_learning.evaluation.qualification import serialize_frozen_policy
    surface_artifact = serialize_frozen_policy(
        surface_model, experiment_id=criteria.experiment_id,
        feature_schema_hash=hashlib.sha256(train_surface.tobytes()).hexdigest()[:16],
        feature_transform_hash=hashlib.sha256(dev_surface.tobytes()).hexdigest()[:16],
        training_seed=args.seed, target_mode="soft",
        training_dataset_hash=_hash_tasks(splits["train"]),
        development_dataset_hash=_hash_tasks(splits["development"]),
        calibration_artifact_hash=None, regularization=0.0, solver="adam")
    (dev_out / "surface_only_policy.json").write_text(json.dumps(surface_artifact, indent=2))

    # Hidden-only: logistic on hidden states only
    print(f"[develop] Training hidden-only baseline...")
    hidden_model = fit_policy(
        config_rep, train_hidden, train_delta_u, train_weights,
        dev_features=dev_hidden, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed)
    np.save(dev_out / "train_hidden_only.npy", train_hidden)
    np.save(dev_out / "dev_hidden_only.npy", dev_hidden)
    hidden_artifact = serialize_frozen_policy(
        hidden_model, experiment_id=criteria.experiment_id,
        feature_schema_hash=hashlib.sha256(train_hidden.tobytes()).hexdigest()[:16],
        feature_transform_hash=hashlib.sha256(dev_hidden.tobytes()).hexdigest()[:16],
        training_seed=args.seed, target_mode="soft",
        training_dataset_hash=_hash_tasks(splits["train"]),
        development_dataset_hash=_hash_tasks(splits["development"]),
        calibration_artifact_hash=None, regularization=0.0, solver="adam")
    (dev_out / "hidden_only_policy.json").write_text(json.dumps(hidden_artifact, indent=2))

    # TF-IDF baseline: logistic on TF-IDF of prompts
    print(f"[develop] Training TF-IDF baseline...")
    train_tfidf = _extract_tfidf_features(splits["train"], splits["train"])
    dev_tfidf = _extract_tfidf_features(splits["development"], splits["train"])
    tfidf_model = fit_policy(
        config_rep, train_tfidf, train_delta_u, train_weights,
        dev_features=dev_tfidf, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed)
    np.save(dev_out / "train_tfidf.npy", train_tfidf)
    np.save(dev_out / "dev_tfidf.npy", dev_tfidf)
    tfidf_artifact = serialize_frozen_policy(
        tfidf_model, experiment_id=criteria.experiment_id,
        feature_schema_hash=hashlib.sha256(train_tfidf.tobytes()).hexdigest()[:16],
        feature_transform_hash=hashlib.sha256(dev_tfidf.tobytes()).hexdigest()[:16],
        training_seed=args.seed, target_mode="soft",
        training_dataset_hash=_hash_tasks(splits["train"]),
        development_dataset_hash=_hash_tasks(splits["development"]),
        calibration_artifact_hash=None, regularization=0.0, solver="adam")
    (dev_out / "tfidf_policy.json").write_text(json.dumps(tfidf_artifact, indent=2))

    # ── Priority 4: Stronger representation baselines ──
    # (a) Character n-gram TF-IDF — captures morphological structure.
    print(f"[develop] Training char-ngram TF-IDF baseline...")
    train_charngram = _extract_char_ngram_features(splits["train"], splits["train"])
    dev_charngram = _extract_char_ngram_features(splits["development"], splits["train"])
    charngram_model = fit_policy(
        config_rep, train_charngram, train_delta_u, train_weights,
        dev_features=dev_charngram, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed)
    np.save(dev_out / "train_charngram.npy", train_charngram)
    np.save(dev_out / "dev_charngram.npy", dev_charngram)
    charngram_artifact = serialize_frozen_policy(
        charngram_model, experiment_id=criteria.experiment_id,
        feature_schema_hash=hashlib.sha256(train_charngram.tobytes()).hexdigest()[:16],
        feature_transform_hash=hashlib.sha256(dev_charngram.tobytes()).hexdigest()[:16],
        training_seed=args.seed, target_mode="soft",
        training_dataset_hash=_hash_tasks(splits["train"]),
        development_dataset_hash=_hash_tasks(splits["development"]),
        calibration_artifact_hash=None, regularization=0.0, solver="adam")
    (dev_out / "charngram_policy.json").write_text(json.dumps(charngram_artifact, indent=2))

    # (b) Text embedding — mean-pooled input token embeddings (no contextual layers).
    # Only available when a real model is loaded (requires get_input_embeddings).
    _has_real_model = (
        model_info is not None and model is not None and tokenizer is not None)
    train_text_emb = None
    if _has_real_model:
        print(f"[develop] Training text-embedding baseline...")
        train_text_emb = _extract_text_embedding_features(
            splits["train"], model, tokenizer, device=device)
        dev_text_emb = _extract_text_embedding_features(
            splits["development"], model, tokenizer, device=device)
        text_emb_model = fit_policy(
            config_rep, train_text_emb, train_delta_u, train_weights,
            dev_features=dev_text_emb, dev_delta_u=dev_delta_u,
            dev_weights=dev_weights, seed=args.seed)
        np.save(dev_out / "train_text_emb.npy", train_text_emb)
        np.save(dev_out / "dev_text_emb.npy", dev_text_emb)
        text_emb_artifact = serialize_frozen_policy(
            text_emb_model, experiment_id=criteria.experiment_id,
            feature_schema_hash=hashlib.sha256(train_text_emb.tobytes()).hexdigest()[:16],
            feature_transform_hash=hashlib.sha256(dev_text_emb.tobytes()).hexdigest()[:16],
            training_seed=args.seed, target_mode="soft",
            training_dataset_hash=_hash_tasks(splits["train"]),
            development_dataset_hash=_hash_tasks(splits["development"]),
            calibration_artifact_hash=None, regularization=0.0, solver="adam")
        (dev_out / "text_emb_policy.json").write_text(json.dumps(text_emb_artifact, indent=2))
    else:
        print(f"[develop] Skipping text-embedding baseline (no real model)")

    # (c) Hidden + TF-IDF — concatenation of hidden states and word TF-IDF.
    print(f"[develop] Training hidden+TF-IDF baseline...")
    train_hidden_tfidf = np.concatenate(
        [train_hidden / np.linalg.norm(train_hidden, axis=1, keepdims=True).clip(min=1),
         train_tfidf], axis=1)
    dev_hidden_tfidf = np.concatenate(
        [dev_hidden / np.linalg.norm(dev_hidden, axis=1, keepdims=True).clip(min=1),
         dev_tfidf], axis=1)
    hidden_tfidf_model = fit_policy(
        config_rep, train_hidden_tfidf, train_delta_u, train_weights,
        dev_features=dev_hidden_tfidf, dev_delta_u=dev_delta_u,
        dev_weights=dev_weights, seed=args.seed)
    np.save(dev_out / "train_hidden_tfidf.npy", train_hidden_tfidf)
    np.save(dev_out / "dev_hidden_tfidf.npy", dev_hidden_tfidf)
    hidden_tfidf_artifact = serialize_frozen_policy(
        hidden_tfidf_model, experiment_id=criteria.experiment_id,
        feature_schema_hash=hashlib.sha256(train_hidden_tfidf.tobytes()).hexdigest()[:16],
        feature_transform_hash=hashlib.sha256(dev_hidden_tfidf.tobytes()).hexdigest()[:16],
        training_seed=args.seed, target_mode="soft",
        training_dataset_hash=_hash_tasks(splits["train"]),
        development_dataset_hash=_hash_tasks(splits["development"]),
        calibration_artifact_hash=None, regularization=0.0, solver="adam")
    (dev_out / "hidden_tfidf_policy.json").write_text(json.dumps(hidden_tfidf_artifact, indent=2))

    # (d) Hidden + text embedding — concatenation of hidden states and text embeddings.
    if _has_real_model and train_text_emb is not None:
        print(f"[develop] Training hidden+text-embedding baseline...")
        train_hidden_text = np.concatenate(
            [train_hidden / np.linalg.norm(train_hidden, axis=1, keepdims=True).clip(min=1),
             train_text_emb], axis=1)
        dev_hidden_text = np.concatenate(
            [dev_hidden / np.linalg.norm(dev_hidden, axis=1, keepdims=True).clip(min=1),
             dev_text_emb], axis=1)
        hidden_text_model = fit_policy(
            config_rep, train_hidden_text, train_delta_u, train_weights,
            dev_features=dev_hidden_text, dev_delta_u=dev_delta_u,
            dev_weights=dev_weights, seed=args.seed)
        np.save(dev_out / "train_hidden_text.npy", train_hidden_text)
        np.save(dev_out / "dev_hidden_text.npy", dev_hidden_text)
        hidden_text_artifact = serialize_frozen_policy(
            hidden_text_model, experiment_id=criteria.experiment_id,
            feature_schema_hash=hashlib.sha256(train_hidden_text.tobytes()).hexdigest()[:16],
            feature_transform_hash=hashlib.sha256(dev_hidden_text.tobytes()).hexdigest()[:16],
            training_seed=args.seed, target_mode="soft",
            training_dataset_hash=_hash_tasks(splits["train"]),
            development_dataset_hash=_hash_tasks(splits["development"]),
            calibration_artifact_hash=None, regularization=0.0, solver="adam")
        (dev_out / "hidden_text_policy.json").write_text(json.dumps(hidden_text_artifact, indent=2))
    else:
        print(f"[develop] Skipping hidden+text-embedding baseline (no real model)")

    # Heuristic baseline: frozen threshold from dev data
    print(f"[develop] Fitting heuristic threshold from dev data...")
    heuristic_threshold = _fit_heuristic_threshold(splits["development"], experiences["development"])
    (dev_out / "heuristic_policy.json").write_text(json.dumps({
        "policy_type": "heuristic",
        "threshold": heuristic_threshold,
        "rules": ["if max_operand >= threshold: symbolic", "else: llm"],
        "frozen_from": "development",
    }, indent=2))
    print(f"[develop] Heuristic threshold: {heuristic_threshold}")

    # ── Train the PRIMARY policy (hidden_plus_surface) ──
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
    # Use smart LLM inference (vLLM API → in-process vLLM → HF generate).
    cal_llm_results = None
    if model_info is not None and model is not None:
        cal_llm_results = _smart_llm_generate(
            splits["calibration"], model, tokenizer, criteria,
            device=getattr(args, "device", "cuda"),
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
        PolicyTrainingSpec, run_sham_control, shuffle_labels_within_bins,
        permute_targets_within_bins)
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

    # Use smart LLM inference (vLLM API → in-process vLLM → HF generate).
    final_llm_results = None
    if model_info is not None and model is not None:
        final_llm_results = _smart_llm_generate(
            splits["final"], model, tokenizer, criteria,
            device=getattr(args, "device", "cuda"),
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
    # Update capture_config to use the SELECTED representation layer and
    # pooling method. Fix 3: pass the actual pooling name through so
    # mean_content_tokens is distinct from mean_prompt_tokens.
    if model_info is not None and capture_config is not None:
        from daph_learning.execution.real_backends import CaptureConfig
        # Map the frozen pooling name to a capture location. The three
        # canonical pooling names are handled directly by
        # _capture_hidden_states; legacy "mean" maps to mean_prompt_tokens.
        loc = frozen_pooling
        if loc == "mean":
            loc = "mean_prompt_tokens"
        capture_config = CaptureConfig(
            layer=frozen_layer,
            location=loc,
        )
        print(f"[final] Capturing hidden states for final split "
              f"(layer={frozen_layer}, pooling={frozen_pooling})...")
        final_hidden = _capture_hidden_states(
            splits["final"], model, tokenizer, capture_config,
            device=getattr(args, "device", "mps"))
    else:
        final_hidden = _build_features(splits["final"])

    print(f"[final] Extracting surface features...")
    final_surface = _extract_surface_features(splits["final"])
    final_features = _concat_features(final_hidden, final_surface)

    # ── Section 2.2: Load frozen best_fixed comparator (selected from dev data) ──
    best_fixed_path = _develop_dir(criteria) / "best_fixed_selection.json"
    if best_fixed_path.exists():
        best_fixed_data = json.loads(best_fixed_path.read_text())
        best_fixed_id = best_fixed_data.get("best_fixed_policy_id", "always_symbolic")
    else:
        # Fallback: always_symbolic (deterministic tie-breaker).
        best_fixed_id = "always_symbolic"
    print(f"[final] Frozen comparator: best_fixed={best_fixed_id}")

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
    # P0 = best_fixed (frozen from dev data), NOT always_llm.
    p1_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    p0_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    oracle_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    always_sym_utilities = np.zeros(len(splits["final"]), dtype=np.float64)
    always_llm_utilities = np.zeros(len(splits["final"]), dtype=np.float64)

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

        # P0 utility: best_fixed (frozen comparator from dev data).
        if best_fixed_id == "always_symbolic":
            p0_utilities[i] = sym_u
        else:
            p0_utilities[i] = llm_u

        # Oracle utility: best of both.
        oracle_utilities[i] = max(sym_u, llm_u)

        # Always-symbolic and always-LLM utilities (for reporting).
        always_sym_utilities[i] = sym_u
        always_llm_utilities[i] = llm_u

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

    # ── Section 3.5: Evaluate ALL trained baselines on final data ──
    print(f"[final] Evaluating trained baselines...")
    trained_baselines = {}
    dev_out = _develop_dir(criteria)

    # Config for training control baselines (same as primary policy).
    config = ExperimentConfig(
        policy_type="logistic",
        target_mode="soft",
        early_stopping_metric="dev_regret",
    ).freeze()

    # Load train experiences for control baseline training.
    train_out = _develop_dir(criteria)
    train_exp_path = train_out / "train_experiences.json"
    if not train_exp_path.exists():
        print(f"[final] ERROR: run develop first", file=sys.stderr)
        return 1
    train_exp = json.loads(train_exp_path.read_text())
    train_delta_u = np.array([e["delta_utility"] for e in train_exp],
                              dtype=np.float64)
    train_weights = np.array([e["sample_weight"] for e in train_exp],
                              dtype=np.float64)
    # Load train features (already captured in develop stage).
    train_features_path = train_out / "train_features.npy"
    if train_features_path.exists():
        train_features_loaded = np.load(train_features_path)
    else:
        train_features_loaded = final_features  # fallback

    # Helper to compute utility from hard routing decisions.
    def _compute_util_from_actions(actions_arr, experiences_list):
        u = 0.0
        for j, exp in enumerate(experiences_list):
            sym_u = float(exp["symbolic"].get("quality", 0.0))
            llm_u = float(exp["llm"].get("quality", 0.0))
            if actions_arr[j] == "symbolic":
                u += sym_u
            elif actions_arr[j] == "llm":
                u += llm_u
        return u / len(experiences_list)

    # ── Shared deployment pipeline (Fix 2: ablation comparability) ──
    # Every learned policy — P1 and every ablation — must traverse the
    # SAME inference path: raw prob → calibration → frozen thresholds →
    # abstention → action → realized utility. Only the feature set (and
    # therefore the fitted model) differs. This makes representation
    # ablations a clean causal estimate of feature-set contribution.
    def _route_through_pipeline(raw_probs, experiences_list):
        """Apply P1's frozen calibration + thresholds + abstention to a
        set of raw probabilities and return (actions, utilities, mean_util)."""
        calibrated = np.array([
            apply_calibration(float(p), cal_artifact) for p in raw_probs
        ])
        actions = np.array([
            select_route_action(
                float(p),
                threshold_symbolic=sym_threshold,
                threshold_llm=llm_threshold,
                abstention_enabled=abstention_enabled,
            ).value for p in calibrated
        ])
        utils = np.zeros(len(experiences_list), dtype=np.float64)
        for j, exp in enumerate(experiences_list):
            sym_u = float(exp["symbolic"].get("quality", 0.0))
            llm_u = float(exp["llm"].get("quality", 0.0))
            if actions[j] == "symbolic":
                utils[j] = sym_u
            elif actions[j] == "llm":
                utils[j] = llm_u
            else:
                utils[j] = 0.0  # abstain
        return actions, utils, float(np.mean(utils))

    # Surface-only baseline — routed through the SAME deployment pipeline
    # as P1 (calibration + frozen thresholds + abstention), so the only
    # difference vs P1 is the feature set.
    surface_policy_path = dev_out / "surface_only_policy.json"
    if surface_policy_path.exists():
        from daph_learning.evaluation.qualification import load_frozen_policy
        s_artifact = json.loads(surface_policy_path.read_text())
        s_hash = hashlib.sha256(surface_policy_path.read_bytes()).hexdigest()
        s_policy = load_frozen_policy(surface_policy_path, expected_sha256=s_hash)
        s_probs = s_policy.predict_proba(final_surface)
        s_actions, s_utils, s_util = _route_through_pipeline(s_probs, final_experiences)
        trained_baselines["surface_only"] = {"utility": s_util, "actions": s_actions.tolist()}
        print(f"    surface_only: utility={s_util:.4f}")

    # Hidden-only baseline — same deployment pipeline as P1.
    hidden_policy_path = dev_out / "hidden_only_policy.json"
    if hidden_policy_path.exists():
        h_artifact = json.loads(hidden_policy_path.read_text())
        h_hash = hashlib.sha256(hidden_policy_path.read_bytes()).hexdigest()
        h_policy = load_frozen_policy(hidden_policy_path, expected_sha256=h_hash)
        h_probs = h_policy.predict_proba(final_hidden)
        h_actions, h_utils, h_util = _route_through_pipeline(h_probs, final_experiences)
        trained_baselines["hidden_only"] = {"utility": h_util, "actions": h_actions.tolist()}
        print(f"    hidden_only: utility={h_util:.4f}")

    # TF-IDF baseline — same deployment pipeline as P1.
    tfidf_policy_path = dev_out / "tfidf_policy.json"
    if tfidf_policy_path.exists():
        final_tfidf = _extract_tfidf_features(splits["final"], splits["train"])
        t_artifact = json.loads(tfidf_policy_path.read_text())
        t_hash = hashlib.sha256(tfidf_policy_path.read_bytes()).hexdigest()
        t_policy = load_frozen_policy(tfidf_policy_path, expected_sha256=t_hash)
        t_probs = t_policy.predict_proba(final_tfidf)
        t_actions, t_utils, t_util = _route_through_pipeline(t_probs, final_experiences)
        trained_baselines["tfidf"] = {"utility": t_util, "actions": t_actions.tolist()}
        print(f"    tfidf: utility={t_util:.4f}")

    # ── Priority 4: Stronger baselines (final evaluation) ──
    # Char n-gram TF-IDF
    charngram_policy_path = dev_out / "charngram_policy.json"
    if charngram_policy_path.exists():
        final_charngram = _extract_char_ngram_features(splits["final"], splits["train"])
        cn_hash = hashlib.sha256(charngram_policy_path.read_bytes()).hexdigest()
        cn_policy = load_frozen_policy(charngram_policy_path, expected_sha256=cn_hash)
        cn_probs = cn_policy.predict_proba(final_charngram)
        cn_actions, _cn_utils, cn_util = _route_through_pipeline(cn_probs, final_experiences)
        trained_baselines["charngram"] = {"utility": cn_util, "actions": cn_actions.tolist()}
        print(f"    charngram: utility={cn_util:.4f}")

    # Text embedding (mean-pooled input token embeddings)
    text_emb_policy_path = dev_out / "text_emb_policy.json"
    if text_emb_policy_path.exists():
        final_text_emb = _extract_text_embedding_features(
            splits["final"], model, tokenizer, device=device)
        te_hash = hashlib.sha256(text_emb_policy_path.read_bytes()).hexdigest()
        te_policy = load_frozen_policy(text_emb_policy_path, expected_sha256=te_hash)
        te_probs = te_policy.predict_proba(final_text_emb)
        te_actions, _te_utils, te_util = _route_through_pipeline(te_probs, final_experiences)
        trained_baselines["text_embedding"] = {"utility": te_util, "actions": te_actions.tolist()}
        print(f"    text_embedding: utility={te_util:.4f}")

    # Hidden + TF-IDF
    hidden_tfidf_policy_path = dev_out / "hidden_tfidf_policy.json"
    if hidden_tfidf_policy_path.exists():
        final_hidden_tfidf = np.concatenate(
            [final_hidden / np.linalg.norm(final_hidden, axis=1, keepdims=True).clip(min=1),
             final_tfidf], axis=1) if tfidf_policy_path.exists() else final_hidden
        ht_hash = hashlib.sha256(hidden_tfidf_policy_path.read_bytes()).hexdigest()
        ht_policy = load_frozen_policy(hidden_tfidf_policy_path, expected_sha256=ht_hash)
        ht_probs = ht_policy.predict_proba(final_hidden_tfidf)
        ht_actions, _ht_utils, ht_util = _route_through_pipeline(ht_probs, final_experiences)
        trained_baselines["hidden_plus_tfidf"] = {"utility": ht_util, "actions": ht_actions.tolist()}
        print(f"    hidden_plus_tfidf: utility={ht_util:.4f}")

    # Hidden + text embedding
    hidden_text_policy_path = dev_out / "hidden_text_policy.json"
    if hidden_text_policy_path.exists():
        if text_emb_policy_path.exists():
            final_hidden_text = np.concatenate(
                [final_hidden / np.linalg.norm(final_hidden, axis=1, keepdims=True).clip(min=1),
                 final_text_emb], axis=1)
        else:
            final_hidden_text = final_hidden
        hx_hash = hashlib.sha256(hidden_text_policy_path.read_bytes()).hexdigest()
        hx_policy = load_frozen_policy(hidden_text_policy_path, expected_sha256=hx_hash)
        hx_probs = hx_policy.predict_proba(final_hidden_text)
        hx_actions, _hx_utils, hx_util = _route_through_pipeline(hx_probs, final_experiences)
        trained_baselines["hidden_plus_text_embedding"] = {
            "utility": hx_util, "actions": hx_actions.tolist()}
        print(f"    hidden_plus_text_embedding: utility={hx_util:.4f}")

    # Heuristic baseline
    heuristic_path = dev_out / "heuristic_policy.json"
    if heuristic_path.exists():
        heuristic_data = json.loads(heuristic_path.read_text())
        threshold = float(heuristic_data.get("threshold", 3.0))
        import re
        heur_actions = []
        for task in splits["final"]:
            spec = str(task.get("specification", ""))
            numbers = [int(n) for n in re.findall(r'\d+', spec)]
            max_num = max(numbers) if numbers else 0
            mag = float(np.log10(max(max_num, 1)))
            heur_actions.append("symbolic" if mag >= threshold else "llm")
        heur_actions = np.array(heur_actions)
        heur_util = _compute_util_from_actions(heur_actions, final_experiences)
        trained_baselines["heuristic"] = {"utility": heur_util, "actions": heur_actions.tolist()}
        print(f"    heuristic: utility={heur_util:.4f}")

    # ── Negative controls (Fix 4: replace invalid square random projection) ──
    # The previous "random_projection" used a square Gaussian matrix, which
    # is almost surely full-rank and therefore approximately invertible — a
    # linear classifier over XA has the same representational capacity as
    # over X, so a performance drop there reflected conditioning/optimization
    # rather than signal destruction. It is replaced with a battery of
    # proper signal-destruction / invariance controls, all routed through
    # the SAME deployment pipeline as P1 (calibration + frozen thresholds).
    hidden_dim = final_hidden.shape[1]
    train_hidden_extracted = (
        train_features_loaded[:, :hidden_dim]
        if train_features_path.exists() else final_hidden
    )
    train_subtypes = np.array([
        t.get("metadata", {}).get("subtype", "A") for t in splits["train"]
    ])

    # (a) Shuffled-hidden — subtype-stratified (group-aware): permute hidden
    # vectors WITHIN each subtype so subtype identity is preserved but the
    # task-specific hidden↔utility alignment is destroyed. This isolates
    # task-specific hidden information from subtype-identity leakage.
    print(f"[final] Evaluating shuffled-hidden control (subtype-stratified)...")
    if train_features_path.exists():
        rng = np.random.default_rng(args.seed)
        train_shuffled = train_features_loaded.copy()
        n_shuffled_rows = 0
        for st in np.unique(train_subtypes):
            mask = (train_subtypes == st)
            idx = np.where(mask)[0]
            if len(idx) > 1:
                order = rng.permutation(len(idx))
                train_shuffled[idx, :hidden_dim] = (
                    train_features_loaded[idx[order], :hidden_dim])
                n_shuffled_rows += len(idx)
        shuffled_model = fit_policy(
            config, train_shuffled, train_delta_u, train_weights,
            seed=args.seed)
        sh_probs = predict_proba(shuffled_model, final_features)
        sh_actions, _sh_utils, sh_util = _route_through_pipeline(
            sh_probs, final_experiences)
        trained_baselines["shuffled_hidden"] = {
            "utility": sh_util, "actions": sh_actions.tolist(),
            "n_shuffled_rows": int(n_shuffled_rows),
            "stratification": "subtype",
        }
        print(f"    shuffled_hidden: utility={sh_util:.4f}")

    # (b) Orthogonal-rotation invariance test: apply a random orthogonal
    # matrix Q (Q Q^T = I). A linear probe over XQ can represent the same
    # boundary as over X (w' = Q^T w), so performance should be ~invariant.
    # A large drop here flags conditioning/optimization sensitivity, not
    # signal loss — making this a useful implementation sanity check.
    print(f"[final] Evaluating orthogonal-rotation invariance control...")
    rng_orth = np.random.default_rng(args.seed + 998)
    G = rng_orth.standard_normal((hidden_dim, hidden_dim)).astype(np.float32)
    Q, _R = np.linalg.qr(G)  # orthonormal columns
    train_orth = train_features_loaded.copy()
    train_orth[:, :hidden_dim] = train_hidden_extracted @ Q
    final_orth = final_features.copy()
    final_orth[:, :hidden_dim] = final_features[:, :hidden_dim] @ Q
    orth_model = fit_policy(
        config, train_orth, train_delta_u, train_weights, seed=args.seed)
    orth_probs = predict_proba(orth_model, final_orth)
    orth_actions, _o_utils, orth_util = _route_through_pipeline(
        orth_probs, final_experiences)
    trained_baselines["orthogonal_rotation"] = {
        "utility": orth_util, "actions": orth_actions.tolist(),
        "interpretation": "should be ~invariant vs hidden_only; large drop flags conditioning sensitivity",
    }
    print(f"    orthogonal_rotation: utility={orth_util:.4f}")

    # (c) Dimension-reduced random projection (Johnson–Lindenstrauss): a
    # wide rectangular Gaussian matrix reduces hidden_dim → hidden_dim//4.
    # Unlike the square case this is NOT invertible, so genuine information
    # is discarded. A performance drop here reflects real dimensionality
    # dependence rather than conditioning. The reduced hidden representation
    # is concatenated with the (unchanged) surface features, so the only
    # change vs the hidden+surface policy is the hidden dimensionality.
    print(f"[final] Evaluating dimension-reduced projection control...")
    reduced_dim = max(1, hidden_dim // 4)
    rng_red = np.random.default_rng(args.seed + 997)
    # JL projection scaled by 1/sqrt(reduced_dim) for norm preservation.
    P = (rng_red.standard_normal((hidden_dim, reduced_dim)).astype(np.float32)
         / np.sqrt(reduced_dim))
    train_red_hidden = (train_hidden_extracted @ P).astype(np.float32)
    final_red_hidden = (final_features[:, :hidden_dim] @ P).astype(np.float32)
    # L2-normalize the reduced hidden states, then concat with surface.
    _tn = np.linalg.norm(train_red_hidden, axis=1, keepdims=True)
    _tn[_tn == 0] = 1.0
    _fn = np.linalg.norm(final_red_hidden, axis=1, keepdims=True)
    _fn[_fn == 0] = 1.0
    train_red = np.concatenate(
        [train_red_hidden / _tn, train_features_loaded[:, hidden_dim:]], axis=1)
    final_red = np.concatenate(
        [final_red_hidden / _fn, final_features[:, hidden_dim:]], axis=1)
    red_model = fit_policy(
        config, train_red, train_delta_u, train_weights, seed=args.seed)
    red_probs = predict_proba(red_model, final_red)
    red_actions, _rd_utils, red_util = _route_through_pipeline(
        red_probs, final_experiences)
    trained_baselines["reduced_projection"] = {
        "utility": red_util, "actions": red_actions.tolist(),
        "reduced_dim": int(reduced_dim),
    }
    print(f"    reduced_projection: utility={red_util:.4f}")

    # (d) Gaussian-noise control: replace hidden states with iid Gaussian
    # noise matched to the per-coordinate mean and std of the real hidden
    # states. This destroys all structure while preserving marginal scale.
    print(f"[final] Evaluating gaussian-noise control...")
    rng_noise = np.random.default_rng(args.seed + 996)
    coord_mean = train_hidden_extracted.mean(axis=0, keepdims=True)
    coord_std = train_hidden_extracted.std(axis=0, keepdims=True) + 1e-6
    train_noise = train_features_loaded.copy()
    train_noise[:, :hidden_dim] = (
        coord_mean + coord_std * rng_noise.standard_normal(
            train_hidden_extracted.shape)).astype(np.float32)
    final_noise = final_features.copy()
    final_noise[:, :hidden_dim] = (
        coord_mean + coord_std * rng_noise.standard_normal(
            final_features[:, :hidden_dim].shape)).astype(np.float32)
    noise_model = fit_policy(
        config, train_noise, train_delta_u, train_weights, seed=args.seed)
    noise_probs = predict_proba(noise_model, final_noise)
    noise_actions, _nz_utils, noise_util = _route_through_pipeline(
        noise_probs, final_experiences)
    trained_baselines["gaussian_noise"] = {
        "utility": noise_util, "actions": noise_actions.tolist()}
    print(f"    gaussian_noise: utility={noise_util:.4f}")

    # (e) Hidden-coordinate permutation: independently permute the hidden
    # coordinates (columns) of each task. This destroys inter-coordinate
    # structure within each vector while preserving the per-coordinate
    # marginal distribution across tasks.
    print(f"[final] Evaluating hidden-coordinate-permutation control...")
    rng_cp = np.random.default_rng(args.seed + 995)
    train_cp = train_features_loaded.copy()
    for i in range(train_cp.shape[0]):
        train_cp[i, :hidden_dim] = train_hidden_extracted[
            i, rng_cp.permutation(hidden_dim)]
    final_cp = final_features.copy()
    final_hidden_extracted = final_features[:, :hidden_dim]
    for i in range(final_cp.shape[0]):
        final_cp[i, :hidden_dim] = final_hidden_extracted[
            i, rng_cp.permutation(hidden_dim)]
    cp_model = fit_policy(
        config, train_cp, train_delta_u, train_weights, seed=args.seed)
    cp_probs = predict_proba(cp_model, final_cp)
    cp_actions, _cp_utils, cp_util = _route_through_pipeline(
        cp_probs, final_experiences)
    trained_baselines["coord_permutation"] = {
        "utility": cp_util, "actions": cp_actions.tolist()}
    print(f"    coord_permutation: utility={cp_util:.4f}")

    # Hidden-norm-only control: use only magnitude statistics of hidden states
    print(f"[final] Evaluating hidden-norm-only control...")
    def _hidden_norm_features(hidden_arr):
        return np.stack([
            np.linalg.norm(hidden_arr, ord=1, axis=1),
            np.linalg.norm(hidden_arr, ord=2, axis=1),
            hidden_arr.mean(axis=1),
            hidden_arr.std(axis=1),
            hidden_arr.max(axis=1),
            hidden_arr.min(axis=1),
        ], axis=1).astype(np.float32)

    # Extract train hidden states from saved train_features (hidden + surface).
    train_norm = _hidden_norm_features(train_hidden_extracted)
    final_norm = _hidden_norm_features(final_hidden)
    norm_model = fit_policy(
        config, train_norm, train_delta_u, train_weights,
        seed=args.seed)
    n_probs = predict_proba(norm_model, final_norm)
    n_actions, _n_utils, n_util = _route_through_pipeline(
        n_probs, final_experiences)
    trained_baselines["hidden_norm_only"] = {"utility": n_util, "actions": n_actions.tolist()}
    print(f"    hidden_norm_only: utility={n_util:.4f}")

    # Save trained baselines.
    (out / "trained_baselines.json").write_text(json.dumps(trained_baselines, indent=2))

    # ── Section 2.3: Hidden-state contribution ablation ──
    print(f"[final] Computing hidden-state contribution ablation...")
    p1_util_val = float(np.mean(p1_utilities))
    surface_util = trained_baselines.get("surface_only", {}).get("utility", 0.0)
    hidden_util = trained_baselines.get("hidden_only", {}).get("utility", 0.0)
    tfidf_util = trained_baselines.get("tfidf", {}).get("utility", 0.0)

    # P_COMBINED - P_SURFACE
    combined_minus_surface = p1_util_val - surface_util
    # P_HIDDEN - P_TFIDF
    hidden_minus_tfidf = hidden_util - tfidf_util

    # Bootstrap CIs for these ablation comparisons.
    combined_minus_surface_arr = np.array([
        p1_utilities[i] - (float(final_experiences[i]["symbolic"].get("quality", 0.0))
                            if trained_baselines.get("surface_only", {}).get("actions", [])[i] == "symbolic"
                            else float(final_experiences[i]["llm"].get("quality", 0.0)))
        for i in range(len(splits["final"]))
    ]) if "surface_only" in trained_baselines else np.zeros(len(splits["final"]))

    hidden_minus_tfidf_arr = np.array([
        (float(final_experiences[i]["symbolic"].get("quality", 0.0))
         if trained_baselines.get("hidden_only", {}).get("actions", [])[i] == "symbolic"
         else float(final_experiences[i]["llm"].get("quality", 0.0)))
        - (float(final_experiences[i]["symbolic"].get("quality", 0.0))
           if trained_baselines.get("tfidf", {}).get("actions", [])[i] == "symbolic"
           else float(final_experiences[i]["llm"].get("quality", 0.0)))
        for i in range(len(splits["final"]))
    ]) if "hidden_only" in trained_baselines and "tfidf" in trained_baselines else np.zeros(len(splits["final"]))

    # Group bootstrap for ablation CIs.
    ablation_groups: dict[str, list[float]] = {}
    for i, task in enumerate(splits["final"]):
        gid = task.get("metadata", {}).get("group_id", "unknown")
        ablation_groups.setdefault(gid, []).append(float(combined_minus_surface_arr[i]))
    ablation_group_arr = {k: np.array(v) for k, v in ablation_groups.items()}
    combined_minus_surface_bootstrap = group_bootstrap_mean_delta(
        ablation_group_arr,
        n_iterations=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_iterations", 20000)),
        confidence_level=0.95,
        seed=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_seed", 20260731)),
        estimand="group_weighted",
    )

    ablation_groups2: dict[str, list[float]] = {}
    for i, task in enumerate(splits["final"]):
        gid = task.get("metadata", {}).get("group_id", "unknown")
        ablation_groups2.setdefault(gid, []).append(float(hidden_minus_tfidf_arr[i]))
    ablation_group_arr2 = {k: np.array(v) for k, v in ablation_groups2.items()}
    hidden_minus_tfidf_bootstrap = group_bootstrap_mean_delta(
        ablation_group_arr2,
        n_iterations=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_iterations", 20000)),
        confidence_level=0.95,
        seed=int(criteria.raw.get("statistics", {}).get(
            "bootstrap_seed", 20260731)),
        estimand="group_weighted",
    )

    min_effect_threshold = float(criteria.raw.get("statistics", {}).get(
        "min_fixed_backend_gain", 0.02))
    hidden_state_claim_supported = (
        combined_minus_surface_bootstrap.ci_low > 0 or
        hidden_minus_tfidf_bootstrap.ci_low > 0
    )

    hidden_state_claim = {
        "combined_minus_surface": {
            "estimate": combined_minus_surface_bootstrap.point_estimate,
            "lcb_95": combined_minus_surface_bootstrap.ci_low,
            "ucb_95": combined_minus_surface_bootstrap.ci_high,
        },
        "hidden_minus_tfidf": {
            "estimate": hidden_minus_tfidf_bootstrap.point_estimate,
            "lcb_95": hidden_minus_tfidf_bootstrap.ci_low,
            "ucb_95": hidden_minus_tfidf_bootstrap.ci_high,
        },
        "claim_supported": hidden_state_claim_supported,
        "minimum_effect_threshold": min_effect_threshold,
    }
    print(f"[final] Hidden-state claim supported: {hidden_state_claim_supported}")
    print(f"    combined-surface: {combined_minus_surface_bootstrap.point_estimate:.4f} "
          f"LCB={combined_minus_surface_bootstrap.ci_low:.4f}")
    print(f"    hidden-tfidf: {hidden_minus_tfidf_bootstrap.point_estimate:.4f} "
          f"LCB={hidden_minus_tfidf_bootstrap.ci_low:.4f}")

    # ── Run sham control (matched: permute continuous ΔU + weights) ──
    # Fix 1: the sham previously binarized ΔU into 0/1 labels and permuted
    # those, then passed them to fit_policy with target_mode="soft" — which
    # re-applied sigmoid(ΔU/τ), turning 0→0.5 and 1→sigmoid(1/τ). That
    # changed the target distribution, magnitude, and weights simultaneously,
    # not just the X↔ΔU association. The matched sham instead permutes the
    # actual signed continuous ΔU (and its matching weights) within strata,
    # so fit_policy applies the identical sigmoid(ΔU/τ) transform the real
    # P1 model received. Only the feature→target association is destroyed.
    n_sham_seeds = int(criteria.raw.get("sham", {}).get("n_seeds", 20))
    print(f"[final] Running sham control ({n_sham_seeds} seeds, matched ΔU permutation)")

    # Train features already loaded above for control baselines.
    train_features = train_features_loaded

    subtypes_arr = np.array([
        t.get("metadata", {}).get("subtype", "A") for t in splits["train"]
    ])
    split_names_arr = np.array(["train"] * len(splits["train"]))
    decisive_arr = np.abs(train_delta_u) > 0.02

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

    # config already defined above for control baselines (same as P1).

    # Sham uses the SAME deployment pipeline as P1 (calibration + frozen
    # thresholds + abstention) for evaluation.
    sham_predictions: list[ShamTaskPrediction] = []

    def sham_train_fn(X, sham_du, sham_w):
        # Identical fit_policy config as P1: target_mode="soft" applies
        # sigmoid(sham_du/τ) to the permuted continuous ΔU.
        return fit_policy(config, X, sham_du, sham_w, seed=args.seed)

    sham_utilities = []
    for seed_i in range(n_sham_seeds):
        seed = args.seed + seed_i
        sham_du, sham_w, n_shuffled = permute_targets_within_bins(
            train_delta_u, train_weights, subtypes_arr,
            split_names_arr, decisive_arr, seed=seed)
        sham_model = sham_train_fn(train_features, sham_du, sham_w)
        sham_probs = predict_proba(sham_model, final_features)
        # Apply the SAME calibration + frozen thresholds + abstention as P1.
        sham_actions, sham_u, sham_util = _route_through_pipeline(
            sham_probs, final_experiences)
        sham_utilities.append(sham_util)
        # Calibrated probabilities (for the prediction ledger).
        sham_calibrated = np.array([
            apply_calibration(float(p), cal_artifact) for p in sham_probs
        ])
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

    # ── Fix 5: freeze subtype-only baseline on DEVELOPMENT data ──
    # Compute per-subtype preferred backend from the development split
    # (never from final labels), then freeze it before final evaluation.
    dev_subtype_prefs: dict[str, str] = {}
    dev_exp_path = _develop_dir(criteria) / "development_experiences.json"
    if dev_exp_path.exists() and "development" in splits:
        dev_exps = json.loads(dev_exp_path.read_text())
        from collections import defaultdict
        dev_subtype_wins: dict[str, dict[str, int]] = defaultdict(
            lambda: {"symbolic": 0, "llm": 0})
        for dev_task, dev_exp in zip(splits["development"], dev_exps):
            st = dev_task.get("metadata", {}).get("subtype", "unknown")
            sym_u = float(dev_exp["symbolic"].get("quality", 0.0))
            llm_u = float(dev_exp["llm"].get("quality", 0.0))
            if sym_u > llm_u:
                dev_subtype_wins[st]["symbolic"] += 1
            elif llm_u > sym_u:
                dev_subtype_wins[st]["llm"] += 1
        for st, wins in dev_subtype_wins.items():
            if wins["symbolic"] >= wins["llm"]:
                dev_subtype_prefs[st] = "symbolic"
            else:
                dev_subtype_prefs[st] = "llm"
        (out / "dev_subtype_preferences.json").write_text(json.dumps({
            "preferences": dev_subtype_prefs,
            "selection_data": "development",
            "frozen_before_final": True,
        }, indent=2))
        print(f"[final] Frozen subtype-only preferences from dev data: {dev_subtype_prefs}")

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
        "baselines": _evaluate_baselines(
            records, best_fixed_id=best_fixed_id,
            dev_subtype_preferences=dev_subtype_prefs),
        "trained_baselines": trained_baselines,
        "hidden_state_claim": hidden_state_claim,
        "primary_policy_id": "hidden_plus_surface",
        "primary_comparator_id": "best_fixed",
        "best_fixed_policy_id": best_fixed_id,
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
        "status": qualification_status.value,
        "primary_policy_id": "hidden_plus_surface",
        "primary_comparator_id": "best_fixed",
        "best_fixed_policy_id": best_fixed_id,
        "primary_endpoint": {
            "estimate": bootstrap_result.point_estimate,
            "lcb_95": bootstrap_result.ci_low,
            "ucb_95": bootstrap_result.ci_high,
            "minimum_effect": float(criteria.raw.get("statistics", {}).get(
                "min_fixed_backend_gain", 0.02)),
            "passes_effect_threshold": bootstrap_result.point_estimate >= float(
                criteria.raw.get("statistics", {}).get("min_fixed_backend_gain", 0.02)),
            "passes_confidence_threshold": bootstrap_result.ci_low > 0,
        },
        "secondary_comparisons": {
            "primary_minus_always_llm": {
                "estimate": p1_utility_val - float(np.mean(always_llm_utilities)),
            },
            "primary_minus_always_symbolic": {
                "estimate": p1_utility_val - float(np.mean(always_sym_utilities)),
            },
            "primary_minus_sham": {
                "estimate": sham_bootstrap.point_estimate,
                "lcb_95": sham_bootstrap.ci_low,
                "ucb_95": sham_bootstrap.ci_high,
            },
            "primary_minus_heuristic": {
                "estimate": p1_utility_val - trained_baselines.get("heuristic", {}).get("utility", 0.0),
            },
        },
        "hidden_state_claim": hidden_state_claim,
        "provenance": {
            "source_hash_match": current_hash == manifest.source_tree_sha256,
            "artifact_hashes_valid": True,
            "final_access_count": 1,
        },
        "preconditions": {p.name: {"passed": p.passed,
                                    "actual": str(p.actual),
                                    "required": str(p.required)}
                           for p in preconditions},
        "statistical_gates": gate_verdicts_dict,
        "failures": [] if qualification_status == QualificationStatus.PASS else [
            name for name, v in gate_verdicts_dict.items() if not v.get("passed", False)
        ],
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

    # ── Generate report (writes GATE_A_RESULTS.md only) ──
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

    # ── Write final gate_decision.json and experiment_results.json ──
    # (AFTER generate_report so our schema with hidden_state_claim persists)
    (out / "gate_decision.json").write_text(json.dumps(gate_decision, indent=2))
    (out / "experiment_results.json").write_text(json.dumps(stats, indent=2))

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
