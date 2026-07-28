"""Matched-norm random-direction control experiment.

This script tests whether the steering vector's **direction** matters, or
only its **magnitude**. It does this by:

1. Loading a real steering vector.
2. Generating N random directions with the same L2 norm.
3. Running the routing pipeline with each random direction.
4. Comparing the routing metrics against the real steering vector.

If the random directions produce the same routing change as the real vector,
then the steering vector's direction is not causally responsible for the
routing change — only the perturbation magnitude matters. This would
undermine the claim that the extracted direction encodes tool-policy
information.

If the real vector produces a significantly different (and better) routing
outcome than the random controls, that is evidence the direction is
meaningful.

See CLAIMS.md §11 (causal ablation) and audit §11.

Usage:

    python scripts/random_direction_control.py \\
        --vector artifacts/invoke_symbolic_tool.npz \\
        --val-tasks data/ood_eval.jsonl \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --n-random 10 \\
        --output artifacts/random_direction_control.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from daph_learning.steering.io import load_vector
from daph_learning.steering.types import SteeringSpec, SteeringVector
from daph_learning.evaluation.routes import evaluate_route_records, load_jsonl
from daph_learning.routing.batched import (
    evaluate_batch_steered_routes as _evaluate_batch_steered_routes,
    as_task_map as _as_task_map,
)
from daph_learning.data.task_utils import load_llm as _load_llm
from daph_learning.data.integrity import canonical_json, task_fingerprint
from daph_learning.evaluation.nulls import empirical_null_test
from daph_learning.steering.hooks import SteeringSafetyLimits


def _make_random_direction(
    reference: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate a random direction with the same L2 norm as the reference."""
    shape = reference.shape
    random_vec = rng.standard_normal(shape).astype(np.float32)
    random_norm = np.linalg.norm(random_vec)
    if random_norm == 0:
        random_vec = np.ones(shape, dtype=np.float32)
        random_norm = float(np.linalg.norm(random_vec))
    reference_norm = float(np.linalg.norm(reference))
    return random_vec * (reference_norm / random_norm)


def _make_random_vector(
    real_vector: SteeringVector,
    rng: np.random.RandomState,
) -> SteeringVector:
    """Create a SteeringVector with a random direction of the same norm,
    preserving the real vector's spec (layer, alpha, anchor, etc.)."""
    random_values = _make_random_direction(real_vector.values, rng)
    # Copy the spec but change the vector_id to mark it as a control.
    spec = SteeringSpec(
        **{
            **real_vector.spec.__dict__,
            "vector_id": f"{real_vector.spec.vector_id}_random_control",
        }
    )
    return SteeringVector(spec=spec, values=random_values)


def run_control_experiment(
    real_vector: SteeringVector,
    tasks: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    *,
    n_random: int = 10,
    seed: int = 42,
    batch_size: int = 32,
    logit_threshold: float = 0.0,
    prompt_format: str = "chat",
    token_resolver: str = "isolated",
    label_scoring: str = "single_token",
    sequence_normalization: str = "mean",
    max_relative_perturbation: float = 0.65,
    max_cosine_shift: float = 0.25,
    label_field: str | None = None,
    label_oracle_kind: str | None = None,
) -> dict:
    """Run the matched-norm random-direction control experiment.

    Returns a dict with:
    - real_metrics: routing metrics with the real steering vector
    - random_metrics: list of routing metrics for each random control
    - random_mean / random_std: aggregate over random controls
    - real_vs_random: comparison summary
    """
    task_map = _as_task_map(tasks)
    task_list = list(task_map.values())
    rng = np.random.RandomState(seed)
    if n_random <= 0:
        raise ValueError("n_random must be positive")
    safety_limits = SteeringSafetyLimits(
        max_relative_perturbation=max_relative_perturbation,
        max_cosine_shift=max_cosine_shift,
    )

    # Real vector
    real_routes = _evaluate_batch_steered_routes(
        task_list,
        model,
        tokenizer,
        vectors=[real_vector],
        alphas=[float(real_vector.spec.alpha)],
        prompt_format=prompt_format,
        batch_size=batch_size,
        threshold=logit_threshold,
        token_resolver=token_resolver,
        label_scoring=label_scoring,
        sequence_normalization=sequence_normalization,
        safety_limits=safety_limits,
    )
    real_metrics = evaluate_route_records(
        task_map,
        real_routes,
        label_field=label_field,
        label_oracle_kind=label_oracle_kind,
    )

    # Random controls
    random_metrics_list = []
    for i in range(n_random):
        random_vec = _make_random_vector(real_vector, rng)
        random_routes = _evaluate_batch_steered_routes(
            task_list,
            model,
            tokenizer,
            vectors=[random_vec],
            alphas=[float(random_vec.spec.alpha)],
            prompt_format=prompt_format,
            batch_size=batch_size,
            threshold=logit_threshold,
            token_resolver=token_resolver,
            label_scoring=label_scoring,
            sequence_normalization=sequence_normalization,
            safety_limits=safety_limits,
        )
        metrics = evaluate_route_records(
            task_map,
            random_routes,
            label_field=label_field,
            label_oracle_kind=label_oracle_kind,
        )
        random_metrics_list.append(metrics)

    # Aggregate random controls
    def _agg(key: str) -> tuple[float, float]:
        vals = [m[key] for m in random_metrics_list]
        return float(np.mean(vals)), float(np.std(vals))

    keys_to_aggregate = ["f1", "precision", "recall", "route_accuracy", "decision_coverage"]
    random_agg = {key: {"mean": _agg(key)[0], "std": _agg(key)[1]} for key in keys_to_aggregate}

    # Comparison: does the real vector outperform random controls?
    real_f1 = real_metrics["f1"]
    random_f1_mean = random_agg["f1"]["mean"]
    null_test = empirical_null_test(
        real_f1,
        [float(metrics["f1"]) for metrics in random_metrics_list],
        minimum_protocol_nulls=500,
    )
    task_digest = hashlib.sha256(
        canonical_json(
            [
                {
                    "task_id": task["task_id"],
                    "fingerprint": task_fingerprint(task),
                }
                for task in task_list
            ]
        ).encode("utf-8")
    ).hexdigest()

    return {
        "n_random": n_random,
        "seed": seed,
        "task_count": len(task_list),
        "task_set_sha256": task_digest,
        "scorer": {
            "label_scoring": label_scoring,
            "sequence_normalization": sequence_normalization,
            "token_resolver": token_resolver,
            "threshold": logit_threshold,
            "batch_size": batch_size,
            "prompt_format": prompt_format,
            "max_relative_perturbation": max_relative_perturbation,
            "max_cosine_shift": max_cosine_shift,
        },
        "real_metrics": real_metrics,
        "random_metrics": random_metrics_list,
        "random_aggregate": random_agg,
        "empirical_null": null_test.to_dict(),
        "protocol_eligible": bool(
            null_test.protocol_eligible and label_scoring == "sequence"
        ),
        "comparison": {
            "real_f1": real_f1,
            "random_f1_mean": random_f1_mean,
            "random_f1_std": random_agg["f1"]["std"],
            "f1_lift": real_f1 - random_f1_mean,
            "empirical_p_value": null_test.empirical_p_value,
            "observed_percentile": null_test.observed_percentile,
            "interpretation": (
                "real_vector_outperforms"
                if null_test.empirical_p_value <= 0.05
                and real_f1 > random_f1_mean
                else "real_vector_within_random_range"
                if null_test.null_interval_95[0]
                <= real_f1
                <= null_test.null_interval_95[1]
                else "real_vector_underperforms"
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Matched-norm random-direction control experiment. Tests whether "
            "the steering vector's direction matters, or only its magnitude. "
            "See CLAIMS.md §11."
        )
    )
    ap.add_argument("--vector", required=True, help="Path to the real steering vector .npz")
    ap.add_argument("--val-tasks", required=True, help="Path to validation tasks JSONL")
    ap.add_argument("--model", required=True, help="Model ID")
    ap.add_argument(
        "--n-random",
        type=int,
        default=999,
        help="Number of random controls (protocol minimum: 500)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--logit-threshold", type=float, default=0.0)
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--route-token-resolver", choices=["isolated", "contextual"], default="isolated")
    ap.add_argument(
        "--label-scoring",
        choices=["sequence", "single_token"],
        default="sequence",
        help="Full sequence scoring is required for protocol-eligible results.",
    )
    ap.add_argument(
        "--sequence-normalization",
        choices=["mean", "sum"],
        default="mean",
    )
    ap.add_argument("--max-relative-perturbation", type=float, default=0.65)
    ap.add_argument("--max-cosine-shift", type=float, default=0.25)
    ap.add_argument("--label-field", default=None)
    ap.add_argument(
        "--label-oracle-kind",
        choices=["capability", "accuracy", "utility", "policy_heuristic"],
        default="policy_heuristic",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    real_vector = load_vector(Path(args.vector))
    tasks = load_jsonl(Path(args.val_tasks))
    model, tokenizer = _load_llm(args.model)

    result = run_control_experiment(
        real_vector,
        tasks,
        model,
        tokenizer,
        n_random=args.n_random,
        seed=args.seed,
        batch_size=args.batch_size,
        logit_threshold=args.logit_threshold,
        prompt_format=args.prompt_format,
        token_resolver=args.route_token_resolver,
        label_scoring=args.label_scoring,
        sequence_normalization=args.sequence_normalization,
        max_relative_perturbation=args.max_relative_perturbation,
        max_cosine_shift=args.max_cosine_shift,
        label_field=args.label_field,
        label_oracle_kind=args.label_oracle_kind,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
