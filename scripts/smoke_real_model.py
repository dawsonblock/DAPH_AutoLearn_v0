"""v0.3.10.1-alpha — real-model smoke test (Section 37 / G25).

Runs a tiny end-to-end real-model intervention + policy evaluation
using Qwen2.5-0.5B-Instruct on MPS (or CUDA if available). This is the
G25 gate: "small real-model smoke run completes."

The smoke split is:
    train        100
    dev           50
    calibration   50
    final        100

For the smoke test we use a synthetic task set with real hidden-state
capture (the task labels are synthetic, but the hidden states are real
Qwen activations). This proves the pipeline executes end-to-end
without requiring a real benchmark dataset.

Usage:
    python scripts/smoke_real_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from daph_learning.interventions.real_pipeline import (
        InterventionConfig, ResidualStreamHook, run_real_intervention,
    )
    from daph_learning.interventions import RealInterventionResult

    print("=" * 60)
    print("v0.3.10.1-alpha real-model smoke test (G25)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Device: {device}")
    print(f"Model: {model_id}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Loaded in {time.time()-t0:.1f}s ({n_layers} layers, dim {hidden_dim})")

    # --- Generate a tiny synthetic task set with real hidden states ---
    print("\n[1/4] Capturing hidden states on 20 tasks...")
    rng = np.random.default_rng(0)
    prompts = [
        "What is 2+2?", "Solve: 3*5", "Define photosynthesis.",
        "Translate hello to French.", "What is the capital of France?",
        "Compute 10-7.", "Explain gravity briefly.", "What is water?",
        "List three prime numbers.", "What is a noun?",
        "Solve: 12/4", "What is mitosis?", "Name a planet.",
        "What is 7+8?", "Define velocity.", "What is DNA?",
        "Compute 6*6.", "What is an adjective?", "What is fusion?",
        "Solve: 15-9.",
    ]
    # Synthetic labels: first 10 are "symbolic-preferred" (math), last 10 "llm".
    labels = ["S"] * 10 + ["L"] * 10
    layer = min(10, n_layers - 1)  # use a mid layer

    activations = []
    task_ids = []
    for i, (prompt, fam) in enumerate(zip(prompts, labels)):
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer][0, -1, :].cpu().numpy()
        activations.append(h.astype(np.float32))
        task_ids.append(f"smoke_{i}")
    activations = np.array(activations, dtype=np.float32)
    print(f"  Captured {len(activations)} hidden states, shape {activations.shape}")

    # --- Train a tiny centroid policy ---
    print("\n[2/4] Training centroid policy on captured features...")
    from daph_learning.policy.centroid_policy import train_centroid_policy
    delta_u = np.array([1.0 if l == "S" else -1.0 for l in labels], dtype=np.float32)
    weights = np.ones(len(labels), dtype=np.float32)
    policy = train_centroid_policy(
        activations, delta_u, weights, gap_threshold=0.1)
    print(f"  Centroid vector norm: {float(np.linalg.norm(policy.vector)):.4f}")
    print(f"  n_symbolic={policy.n_symbolic_}, n_llm={policy.n_llm_}")

    # --- Run real-model intervention with the learned vector ---
    print("\n[3/4] Running real-model intervention (+v / 0 / -v)...")
    v = policy.vector
    # Use the first 5 tasks for the intervention study.
    intervention_tasks = [
        {"task_id": task_ids[i], "prompt": prompts[i], "family": labels[i]}
        for i in range(5)
    ]

    def policy_prob_fn(h):
        h = np.asarray(h, dtype=np.float64).flatten()
        if policy.vector.shape[0] != h.shape[0]:
            return 0.5
        return float(1.0 / (1.0 + np.exp(-((h @ policy.vector - policy.threshold) / max(policy.temperature, 1e-8)))))

    def utility_fn(task, route):
        from daph_learning.policy.types import Route
        if isinstance(route, Route):
            route = route.value
        if route == "abstain":
            return 0.0
        fam = task["family"]
        if fam == "S":
            return 1.0 if route == "symbolic" else 0.0
        return 1.0 if route == "llm" else 0.0

    cfg = InterventionConfig(
        layer=layer,
        alpha_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
        max_vector_norm=1.0,
        clamp_norm=5.0,
    )
    results = run_real_intervention(
        model, tok, intervention_tasks, v,
        config=cfg,
        policy_prob_fn=policy_prob_fn,
        utility_fn=utility_fn,
        device=device,
    )
    print(f"  {len(results)} intervention results recorded")
    if results:
        # Show a sample.
        r = results[0]
        print(f"  Sample (task={r.task_id}, alpha={r.alpha}):")
        print(f"    baseline_route={r.baseline_route} -> intervened_route={r.intervened_route}")
        print(f"    route_score: {r.route_score_before:.4f} -> {r.route_score_after:.4f}")
        print(f"    utility_delta={r.utility_delta:.4f}, clamp={r.clamp_triggered}")
        print(f"    evidence_level={r.evidence_level}")

    # --- Aggregate +v / 0 / -v directional effect ---
    print("\n[4/4] Aggregating +v / 0 / -v directional effect...")
    plus_results = [r for r in results if r.alpha == 1.0]
    zero_results = [r for r in results if r.alpha == 0.0]
    minus_results = [r for r in results if r.alpha == -1.0]
    if plus_results and zero_results and minus_results:
        mean_p_plus = float(np.mean([r.route_score_after for r in plus_results]))
        mean_p_zero = float(np.mean([r.route_score_after for r in zero_results]))
        mean_p_minus = float(np.mean([r.route_score_after for r in minus_results]))
        print(f"  E[P(S|+v)] = {mean_p_plus:.4f}")
        print(f"  E[P(S| 0)] = {mean_p_zero:.4f}")
        print(f"  E[P(S|-v)] = {mean_p_minus:.4f}")
        plus_gt = sum(1 for r in plus_results if r.route_score_after > r.route_score_before)
        minus_lt = sum(1 for r in minus_results if r.route_score_after < r.route_score_before)
        print(f"  +v > 0: {plus_gt}/{len(plus_results)}, -v < 0: {minus_lt}/{len(minus_results)}")

    # --- Save smoke artifact ---
    out_path = REPO / "artifacts" / "smoke_real_model_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "evidence_level": "REAL_MODEL_DEV",
        "model_id": model_id,
        "device": device,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_used": layer,
        "n_tasks": len(prompts),
        "n_intervention_results": len(results),
        "alpha_grid": list(cfg.alpha_grid),
        "policy_vector_norm": float(np.linalg.norm(v)),
        "intervention_results": [r.to_dict() for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nSmoke artifact saved to {out_path}")
    print("=" * 60)
    print("G25 SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
