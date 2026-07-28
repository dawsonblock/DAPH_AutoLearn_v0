"""AutoLearn CLI: run the iterative steering-vector learning loop.

This is the script that gives the project its name. It runs the learning
loop described in ``daph_learning.autolearn``:

1. Routes training tasks with the current steering vector.
2. Executes the chosen backend.
3. Learns from misrouted tasks.
4. Updates the steering vector.
5. Evaluates on validation.
6. Repeats.

v0.3.9: the CLI now supports two engines:

- ``--engine counterfactual`` (default): the corrected v0.3.9
  counterfactual learning loop that uses the candidate steering vector for
  candidate routing decisions and the incumbent vector for incumbent routing
  decisions via an injected ``RoutePolicyFn``.
- ``--engine legacy`` (deprecated): the v0.3.8.1 ``run_autolearn_loop``
  implementation, retained for regression comparison until feature parity
  is verified.

Usage:

    daph-autolearn --train-tasks data/train.jsonl \\
        --val-tasks data/val.jsonl \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --layer 24 \\
        --n-iterations 5 \\
        --output artifacts/autolearn_result.json \\
        --vector-output artifacts/autolearn_vector.npz

See CLAIMS.md §18 for what this loop does and does not establish.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from daph_learning.autolearn import AutoLearnConfig, run_autolearn_loop
from daph_learning.steering.io import save_vector
from daph_learning.data.task_utils import load_llm as _load_llm
from daph_learning.evaluation.routes import load_jsonl
from daph_learning.evaluation.protocol import (
    ProtocolPurpose,
    assert_split_access,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the AutoLearn learning loop: iteratively improve a steering "
            "vector by learning from execution outcomes. See CLAIMS.md §18."
        )
    )
    ap.add_argument("--train-tasks", required=True, help="Training tasks JSONL")
    ap.add_argument("--val-tasks", required=True, help="Validation tasks JSONL")
    ap.add_argument("--model", required=True, help="HuggingFace model ID")
    ap.add_argument("--layer", type=int, default=24, help="Transformer layer to steer")
    ap.add_argument("--alpha", type=float, default=1.0, help="Steering alpha")
    ap.add_argument("--anchor", default="ACTION", help="Anchor token")
    ap.add_argument(
        "--capture-token-scope",
        choices=["anchor", "last", "mean_anchor_span", "mean_sequence"],
        default="anchor",
        help=(
            "Activation-pooling method at capture time. 'anchor' (default) "
            "resolves and captures the configured anchor token; 'last' "
            "captures the final prompt token (legacy v0.3.x behaviour); "
            "'mean_anchor_span' mean-pools the anchor's token span; "
            "'mean_sequence' mean-pools the whole sequence."
        ),
    )
    ap.add_argument("--n-iterations", type=int, default=5)
    ap.add_argument("--min-examples", type=int, default=4, help="Min examples per class for update")
    ap.add_argument("--normalization", choices=["l2", "none", "mean_centered"], default="l2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-format", choices=["raw", "chat"], default="raw")
    ap.add_argument(
        "--routing-mode",
        choices=["auto", "logit", "generate"],
        default="auto",
        help=(
            "Routing method: 'auto' tries logit-based, falls back to generate; "
            "'logit' forces logit (fails for multi-token tokenizers); "
            "'generate' forces generate-mode (slower, works with any tokenizer)."
        ),
    )
    ap.add_argument("--label-field", default="route_label")
    ap.add_argument(
        "--label-oracle-kind",
        choices=["capability", "accuracy", "utility", "policy_heuristic"],
        default="policy_heuristic",
    )
    ap.add_argument("--output", required=True, help="Output JSON with learning curve")
    ap.add_argument("--vector-output", help="Save the best steering vector to .npz")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="dev")
    ap.add_argument("--max-relative-perturbation", type=float, default=0.65)
    ap.add_argument("--max-cosine-shift", type=float, default=0.25)
    # v0.3.9: engine selection.
    ap.add_argument(
        "--engine",
        choices=["counterfactual", "legacy"],
        default="counterfactual",
        help=(
            "Learning engine: 'counterfactual' (default) uses the corrected "
            "v0.3.9 loop with real candidate-vs-incumbent policy evaluation; "
            "'legacy' uses the v0.3.8.1 loop (deprecated, for regression "
            "comparison only)."
        ),
    )
    # v0.3.9: counterfactual engine config.
    ap.add_argument("--eta", type=float, default=0.5, help="Trust-region learning rate")
    ap.add_argument("--abstention-band", type=float, default=0.0, help="Delta-U abstention band")
    ap.add_argument("--min-vector-norm", type=float, default=0.0, help="Min steering vector norm")
    ap.add_argument("--max-vector-norm", type=float, default=None, help="Max steering vector norm")
    ap.add_argument("--max-update-norm", type=float, default=None, help="Max update norm per step")
    ap.add_argument("--initial-steering-norm", type=float, default=1.0, help="Bootstrap norm")
    ap.add_argument("--min-capture-coverage", type=float, default=0.0, help="Min capture coverage gate")
    args = ap.parse_args()

    assert_split_access(args.train_split, ProtocolPurpose.VECTOR_CAPTURE)
    assert_split_access(args.val_split, ProtocolPurpose.MODEL_SELECTION)

    train_tasks = load_jsonl(Path(args.train_tasks))
    val_tasks = load_jsonl(Path(args.val_tasks))
    model, tokenizer = _load_llm(args.model)

    if args.engine == "legacy":
        warnings.warn(
            "--engine legacy is deprecated. Use --engine counterfactual (the "
            "default) for the corrected v0.3.9 learning loop. The legacy "
            "engine will be removed after feature parity is verified.",
            DeprecationWarning,
            stacklevel=2,
        )
        _run_legacy_engine(args, train_tasks, val_tasks, model, tokenizer)
    else:
        _run_counterfactual_engine(args, train_tasks, val_tasks, model, tokenizer)


def _run_legacy_engine(args, train_tasks, val_tasks, model, tokenizer) -> None:
    """Run the v0.3.8.1 legacy loop (deprecated)."""
    config = AutoLearnConfig(
        n_iterations=args.n_iterations,
        layer=args.layer,
        alpha=args.alpha,
        anchor=args.anchor,
        capture_token_scope=args.capture_token_scope,
        min_examples_per_update=args.min_examples,
        normalization=args.normalization,
        seed=args.seed,
        max_relative_perturbation=args.max_relative_perturbation,
        max_cosine_shift=args.max_cosine_shift,
    )

    result = run_autolearn_loop(
        train_tasks,
        val_tasks,
        model,
        tokenizer,
        config=config,
        prompt_format=args.prompt_format,
        label_field=args.label_field,
        label_oracle_kind=args.label_oracle_kind,
        routing_mode=args.routing_mode,
    )

    output = {
        "engine": "legacy",
        "config": {
            "n_iterations": config.n_iterations,
            "layer": config.layer,
            "alpha": config.alpha,
            "anchor": config.anchor,
            "capture_token_scope": config.capture_token_scope,
            "normalization": config.normalization,
            "seed": config.seed,
            "min_examples_per_update": config.min_examples_per_update,
            "max_relative_perturbation": config.max_relative_perturbation,
            "max_cosine_shift": config.max_cosine_shift,
        },
        "best_iteration": result.best_iteration,
        "best_val_f1": result.best_val_f1,
        "stop_reason": result.stop_reason,
        "learning_curve": [
            {
                "iteration": m.iteration,
                "n_train": m.n_train,
                "n_correct": m.n_correct,
                "n_misrouted": m.n_misrouted,
                "n_unverifiable": m.n_unverifiable,
                "train_accuracy": m.train_accuracy,
                "val_f1": m.val_f1,
                "val_accuracy": m.val_accuracy,
                "val_precision": m.val_precision,
                "val_recall": m.val_recall,
                "vector_norm": m.vector_norm,
                "n_positive_examples": m.n_positive_examples,
                "n_negative_examples": m.n_negative_examples,
                "updated": m.updated,
                "n_capture_attempts": m.n_capture_attempts,
                "n_capture_successes": m.n_capture_successes,
                "capture_coverage": m.capture_coverage,
                "capture_failure_rate_positive": m.capture_failure_rate_positive,
                "capture_failure_rate_negative": m.capture_failure_rate_negative,
            }
            for m in result.iterations
        ],
        "n_train_tasks": len(train_tasks),
        "n_val_tasks": len(val_tasks),
    }

    _write_output(args, output, result.best_vector)


def _run_counterfactual_engine(args, train_tasks, val_tasks, model, tokenizer) -> None:
    """Run the corrected v0.3.9 counterfactual learning loop."""
    import numpy as np
    from daph_learning.autolearn.trust_region import (
        CounterfactualLearningLoop,
        LoopConfig,
    )
    from daph_learning.autolearn.counterfactual import UtilityConfig
    from daph_learning.autolearn.promotion import PromotionGateConfig
    from daph_learning.autolearn.route_policy import SteeringPolicyConfig
    from daph_learning.autolearn.adapters import (
        make_model_route_policy,
        make_model_capture_fn,
        make_model_execute_fn,
    )

    hidden_size = (
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd", None)
    )
    if hidden_size is None:
        raise ValueError("cannot determine model hidden_size")

    spc = SteeringPolicyConfig(
        layer=args.layer,
        token_location=args.capture_token_scope,
        alpha=args.alpha,
        model_revision=getattr(model.config, "_name_or_path", None),
        tokenizer_revision=getattr(tokenizer, "name_or_path", None),
        vector_version="v0.3.9",
    )

    utility_config = UtilityConfig(
        lambda_risk=1.0,
        abstention_band=args.abstention_band,
    )
    gate_config = PromotionGateConfig()
    config = LoopConfig(
        utility_config=utility_config,
        gate_config=gate_config,
        steering_policy_config=spc,
        eta=args.eta,
        min_vector_norm=args.min_vector_norm,
        max_vector_norm=args.max_vector_norm,
        max_update_norm=args.max_update_norm,
        initial_steering_norm=args.initial_steering_norm,
        min_capture_coverage=args.min_capture_coverage,
    )

    route_fn = make_model_route_policy(model, tokenizer, prompt_format=args.prompt_format)
    capture_fn = make_model_capture_fn(
        model, tokenizer,
        layer=args.layer,
        anchor=args.anchor,
        capture_token_scope=args.capture_token_scope,
        prompt_format=args.prompt_format,
    )
    execute_fn = make_model_execute_fn(model, tokenizer, prompt_format=args.prompt_format)

    loop = CounterfactualLearningLoop(
        seed_vector=np.zeros(hidden_size, dtype=np.float32),
        training_tasks=train_tasks,
        held_out_tasks=val_tasks,
        capture_fn=capture_fn,
        execute_fn=execute_fn,
        route_fn=route_fn,
        config=config,
        model_revision=spc.model_revision,
        tokenizer_revision=spc.tokenizer_revision,
    )
    results = loop.run(max_iterations=args.n_iterations)

    output = {
        "engine": "counterfactual",
        "config": {
            "n_iterations": args.n_iterations,
            "layer": args.layer,
            "alpha": args.alpha,
            "eta": args.eta,
            "abstention_band": args.abstention_band,
            "capture_token_scope": args.capture_token_scope,
            "min_vector_norm": args.min_vector_norm,
            "max_vector_norm": args.max_vector_norm,
            "max_update_norm": args.max_update_norm,
            "initial_steering_norm": args.initial_steering_norm,
            "min_capture_coverage": args.min_capture_coverage,
        },
        "learning_curve": [
            {
                "iteration": r.iteration,
                "promoted": r.promoted,
                "n_train": r.n_train,
                "n_held_out": r.n_held_out,
                "n_abstain": r.n_abstain,
                "decision": r.decision.to_dict(),
                "candidate_vector_norm": float(np.linalg.norm(r.candidate_vector)),
                "incumbent_vector_norm": float(np.linalg.norm(r.incumbent_vector)),
                "capture_attempts": r.capture_attempts,
                "capture_successes": r.capture_successes,
                "capture_failures": r.capture_failures,
                "capture_coverage": r.capture_coverage,
            }
            for r in results
        ],
        "n_train_tasks": len(train_tasks),
        "n_val_tasks": len(val_tasks),
        "final_incumbent_norm": float(np.linalg.norm(loop.incumbent)),
    }

    # Save the final incumbent vector.
    best_vector = loop.incumbent
    _write_output(args, output, best_vector)


def _write_output(args, output, best_vector) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    if args.vector_output and best_vector is not None:
        save_vector(best_vector, Path(args.vector_output))
        print(f"Best vector saved to {args.vector_output}")

    engine = output.get("engine", "unknown")
    print(f"\nAutoLearn complete ({engine} engine).")
    if "learning_curve" in output:
        curve = output["learning_curve"]
        print(f"{len(curve)} iterations.")
        if "best_iteration" in output:
            print(f"Best iteration: {output['best_iteration']} (val F1={output['best_val_f1']:.4f})")
        if "stop_reason" in output:
            print(f"Stop reason: {output['stop_reason']}")
        if "final_incumbent_norm" in output:
            print(f"Final incumbent norm: {output['final_incumbent_norm']:.4f}")
    print(f"\nFull results: {output_path}")


if __name__ == "__main__":
    main()
