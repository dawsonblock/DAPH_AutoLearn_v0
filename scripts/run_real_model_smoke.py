#!/usr/bin/env python
"""Section 21 — minimal real-model smoke (Priority 0 slice).

A SMALL, HONEST real-model smoke that exercises the new canonical
``FINAL_ANSWER:`` verifier and the new artifact validator on a real
Hugging Face causal language model loaded from the local cache (no
download). It:

* loads a real HF causal LM (Qwen2.5-0.5B-Instruct) on MPS/CPU;
* captures an actual hidden state via ``capture_task_representation``;
* executes BOTH backends (symbolic + LLM) on a few arithmetic tasks;
* verifies actual generated answers with the canonical verifier;
* writes a REAL_MODEL_SMOKE evidence bundle;
* validates the bundle with ``validate_artifact_bundle``;
* NEVER marks the result Gate A qualified.

This is NOT a Gate A run. It does not train a policy, does not compute
group-aware confidence bounds, and does not produce qualification
evidence. It only demonstrates that the real-model path, the eval-free
symbolic path, the canonical verifier, and the artifact validator work
end-to-end on a real model.

Usage::

    python scripts/run_real_model_smoke.py [--model Qwen/Qwen2.5-0.5B-Instruct]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-tasks", type=int, default=6)
    parser.add_argument(
        "--out-root", default=str(REPO_ROOT / "artifacts" / "real_model_smoke"))
    args = parser.parse_args()

    # Force offline mode so we only use the locally cached model.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from daph_learning.provenance import compute_canonical_source_hash
    from daph_learning.evaluation.canonical_verifier import (
        CanonicalIntegerVerifier, VerificationStatus,
    )
    from daph_learning.evaluation.artifact_integrity import (
        validate_artifact_bundle,
    )

    source_hash = compute_canonical_source_hash(REPO_ROOT)
    run_id = "smoke_" + hashlib.sha256(
        (source_hash + str(time.time())).encode()).hexdigest()[:12]

    print(f"[smoke] source_hash={source_hash[:16]}... run_id={run_id}")
    print(f"[smoke] loading model {args.model} (offline, from cache)")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"[smoke] SKIP: torch/transformers unavailable: {exc}",
              file=sys.stderr)
        return 0  # not a failure — smoke is opportunistic

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16).to(device).eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] SKIP: could not load model from cache: {exc}",
              file=sys.stderr)
        return 0  # opportunistic — not a failure

    n_layers = model.config.num_hidden_layers
    print(f"[smoke] device={device} n_layers={n_layers}")

    from daph_learning.execution.real_backends import (
        CaptureConfig, LLMGenerationConfig,
        capture_task_representation,
        execute_llm_backend, execute_symbolic_backend,
    )

    # A few structured arithmetic tasks. The prompt requires the canonical
    # FINAL_ANSWER field so the canonical verifier can score the LLM output.
    base_tasks = [
        {"a": 8137, "b": 296, "op": "*", "expected": 8137 * 296},
        {"a": 1234, "b": 5678, "op": "+", "expected": 1234 + 5678},
        {"a": 9999, "b": 100, "op": "-", "expected": 9999 - 100},
        {"a": 4096, "b": 8, "op": "*", "expected": 4096 * 8},
        {"a": 1000, "b": 7, "op": "//", "expected": 1000 // 7},
        {"a": 7777, "b": 13, "op": "%", "expected": 7777 % 13},
    ][: args.n_tasks]

    cap_cfg = CaptureConfig(layer=min(10, n_layers), location="last_token")
    gen_cfg = LLMGenerationConfig(max_new_tokens=48, do_sample=False)
    verifier = CanonicalIntegerVerifier()

    outcomes = []
    for i, t in enumerate(base_tasks):
        prompt = (
            f"Calculate exactly: {t['a']} {t['op']} {t['b']}.\n"
            f"Respond with FINAL_ANSWER: <integer> on the last line.\n"
            f"FINAL_ANSWER must contain only the integer result."
        )
        task = {
            "task_id": f"smoke_{i:02d}",
            "prompt": prompt,
            "specification": f"Compute {t['a']} {t['op']} {t['b']}",
            "capability_ids": ["integer_arithmetic"],
            "inputs": {"a": t["a"], "b": t["b"], "op": t["op"]},
            "expected": t["expected"],
            "metadata": {"subtype": "A"},
        }

        # Capture hidden state once (pre-execution).
        try:
            h = capture_task_representation(
                task, model, tokenizer, config=cap_cfg, device=device)
            h_shape = list(h.shape)
        except Exception as exc:  # noqa: BLE001
            h_shape = None
            print(f"[smoke] task {i}: hidden-state capture failed: {exc}")

        # Symbolic backend (eval-free bounded AST evaluator).
        sym_outcome, sym_text = execute_symbolic_backend(task)
        # LLM backend (real generation).
        llm_outcome, llm_text = execute_llm_backend(
            task, model, tokenizer, generation_config=gen_cfg, device=device)

        # Verify both with the canonical verifier.
        sym_status = verifier.verify(
            task, sym_text, execution_succeeded=sym_outcome.execution_success)
        llm_status = verifier.verify(
            task, llm_text, execution_succeeded=llm_outcome.execution_success)

        outcomes.append({
            "task_id": task["task_id"],
            "expected": t["expected"],
            "hidden_state_shape": h_shape,
            "symbolic": {
                "output": sym_text,
                "execution_success": sym_outcome.execution_success,
                "latency_sec": round(sym_outcome.latency_sec, 4),
                "verification_status": sym_status.value,
            },
            "llm": {
                "output": (llm_text or "")[:200],
                "execution_success": llm_outcome.execution_success,
                "latency_sec": round(llm_outcome.latency_sec, 4),
                "verification_status": llm_status.value,
            },
        })
        print(f"[smoke] task {i}: expected={t['expected']} "
              f"sym={sym_status.value} llm={llm_status.value} "
              f"llm_out={(llm_text or '')[:60]!r}")

    # Write the REAL_MODEL_SMOKE evidence bundle.
    out_dir = Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "artifact_type": "real_model_smoke",
        "experiment_id": "daph_gate_a_smoke",
        "run_id": run_id,
        "evidence_level": "REAL_MODEL_SMOKE",
        "status": "SMOKE",
        "source_hash": source_hash,
        "model_id": args.model,
        "model_revision": args.model,
        "tokenizer_revision": args.model,
        "device": device,
        "n_layers": n_layers,
        "n_tasks": len(outcomes),
        "generated_at": time.time(),
        "outcomes": outcomes,
        "note": ("Minimal real-model smoke. NOT a Gate A run. Does not "
                 "train a policy, compute confidence bounds, or produce "
                 "qualification evidence. Never mark as Gate A qualified."),
    }
    (out_dir / "smoke_result.json").write_text(json.dumps(bundle, indent=2))

    # Validate the bundle with the new artifact validator.
    result = validate_artifact_bundle(out_dir, REPO_ROOT)
    (out_dir / "validation.json").write_text(json.dumps({
        "valid": result.valid,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }, indent=2))

    print(f"[smoke] bundle written: {out_dir}")
    print(f"[smoke] bundle valid={result.valid} "
          f"errors={len(result.errors)} warnings={len(result.warnings)}")
    if result.errors:
        for e in result.errors:
            print(f"[smoke]   ERROR: {e}")
    if not result.valid:
        print("[smoke] bundle validation failed", file=sys.stderr)
        return 1
    print("[smoke] OK — real-model smoke completed. NOT Gate A qualified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
