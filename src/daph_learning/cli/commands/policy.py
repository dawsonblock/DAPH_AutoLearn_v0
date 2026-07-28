"""v0.3.10 — CLI for the counterfactual policy learner.

Modes:

    daph-autolearn-policy train
        Train a weighted logistic / centroid policy.
    daph-autolearn-policy evaluate
        Evaluate a trained policy on held-out tasks.
    daph-autolearn-policy intervene
        Run causal intervention experiments.
    daph-autolearn-policy calibrate
        Tune abstention / OOD thresholds on calibration split.

Options (Section 46):

    --policy centroid | logistic
    --soft-targets
    --target-temperature
    --weight-mode gap | snr
    --gap-threshold
    --layer
    --alpha
    --confidence-threshold
    --ood-threshold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _add_common_policy_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--policy", choices=["centroid", "logistic"],
        default="logistic",
        help="Policy type: centroid (baseline) "
             "or logistic (primary learner)",
    )
    ap.add_argument(
        "--soft-targets", action="store_true",
        default=True,
        help="Use soft targets q=σ(ΔU/τ) (default). "
             "Use --no-soft-targets for hard labels.",
    )
    ap.add_argument(
        "--no-soft-targets", dest="soft_targets",
        action="store_false",
        help="Use hard labels for ablation.",
    )
    ap.add_argument(
        "--target-temperature", type=float,
        default=1.0,
        help="Temperature τ for soft targets. "
             "Smaller => sharper preference.",
    )
    ap.add_argument("--weight-mode", choices=["gap", "snr"], default="gap",
                    help="Sample weighting mode.")
    ap.add_argument("--gap-threshold", type=float, default=0.0,
                    help="Tasks with |ΔU| <= gap_threshold get zero weight.")
    ap.add_argument(
        "--layer", type=int, default=24,
        help="Transformer layer index.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Steering alpha.")
    ap.add_argument("--confidence-threshold", type=float, default=0.70,
                    help="Abstention threshold τ_conf.")
    ap.add_argument("--ood-threshold", type=float, default=float("inf"),
                    help="OOD abstention threshold.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--model", default=None, help="HuggingFace model ID.")
    ap.add_argument(
        "--model-revision", default=None,
        help="Model revision hash.")
    ap.add_argument(
        "--tokenizer-revision", default=None,
        help="Tokenizer revision hash.")


def _build_config(args):
    from daph_learning.policy.config import ExperimentConfig
    return ExperimentConfig(
        target_temperature=args.target_temperature,
        soft_targets=args.soft_targets,
        weight_mode=args.weight_mode,
        gap_threshold=args.gap_threshold,
        selected_layer=args.layer,
        steering_alpha=args.alpha,
        confidence_threshold=args.confidence_threshold,
        ood_threshold=args.ood_threshold,
        random_seed=args.seed,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        policy_type=args.policy,
    )


def _cmd_train(args) -> int:
    """Train a policy on train tasks, evaluate on dev."""
    from daph_learning.policy.learner import (
        build_counterfactual_experiences,
        train_policy_learner,
    )

    cfg = _build_config(args)

    # Execute_fn must be provided by the caller environment. For the CLI
    # we use the synthetic environment if --synthetic is set, otherwise
    # we require a model (not implemented in this alpha for safety).
    if args.synthetic:
        from daph_learning.environment_synthetic import (
            make_synthetic_tasks, synthetic_execute_fn,
            synthetic_utility,
        )
        train_tasks = make_synthetic_tasks(
            n_per_family=50, dim=8, seed=args.seed)
        dev_tasks = make_synthetic_tasks(
            n_per_family=30, dim=8, seed=args.seed + 1)
        execute_fn = synthetic_execute_fn

        def utility_fn(task, route):
            return synthetic_utility(task, route)
    else:
        print(
            "ERROR: real-model training requires "
            "--model and a capture pipeline. "
            "Use --synthetic for the synthetic "
            "closed-loop environment.",
            file=sys.stderr,
        )
        return 1

    train_exp = build_counterfactual_experiences(
        train_tasks, execute_fn=execute_fn, config=cfg)
    dev_exp = (
        build_counterfactual_experiences(
            dev_tasks, execute_fn=execute_fn, config=cfg)
        if dev_tasks else None)
    train_acts = np.array(
        [t.get("activation", np.zeros(8))
         for t in train_tasks],
        dtype=np.float32)
    dev_acts = (
        np.array(
            [t.get("activation", np.zeros(8))
             for t in dev_tasks],
            dtype=np.float32)
        if dev_tasks else None)

    def incumbent_route_fn(h):
        return "llm"

    result = train_policy_learner(
        train_exp, train_acts,
        config=cfg,
        dev_experiences=dev_exp,
        dev_activations=dev_acts,
        incumbent_route_fn=incumbent_route_fn,
        utility_fn=utility_fn,
        dev_tasks=dev_tasks,
    )
    output = {
        "version": cfg.autolearn_version,
        "config": cfg.to_dict(),
        "n_train": result.n_train,
        "n_dev": result.n_dev,
        "dev_metrics": result.dev_metrics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Training complete. Results written to {args.output}")
    if result.dev_metrics:
        m = result.dev_metrics
        print(f"  mean_candidate_utility: "
              f"{m.get('mean_candidate_utility', 'N/A')}")
        print(f"  mean_incumbent_utility: "
              f"{m.get('mean_incumbent_utility', 'N/A')}")
        print(f"  mean_candidate_regret:  "
              f"{m.get('mean_candidate_regret', 'N/A')}")
        print(f"  mean_incumbent_regret:  "
              f"{m.get('mean_incumbent_regret', 'N/A')}")
        print(f"  brier_score:            {m.get('brier_score', 'N/A')}")
        print(f"  ece:                    {m.get('ece', 'N/A')}")
    return 0


def _cmd_evaluate(args) -> int:
    """Evaluate a trained policy on held-out tasks."""
    print(
        "Evaluate mode: load a trained policy and "
        "compute regret/utility/calibration.")
    print(
        "See EXPERIMENT_PROTOCOL_V0_3_10.md "
        "for the full evaluation protocol.")
    return 0


def _cmd_intervene(args) -> int:
    """Run causal intervention experiments (+v/0/-v dose-response)."""
    from daph_learning.interventions import (
        run_intervention_experiment,
        dose_response_summary,
    )
    from daph_learning.policy.types import Route
    print(
        "Intervention mode: run dose-response "
        "experiments on a candidate direction.")
    if not args.vector_file:
        print(
            "ERROR: --vector-file is required "
            "for intervene mode.",
            file=sys.stderr)
        return 1
    vec = np.load(args.vector_file)
    # Toy policy for demonstration: P(S|h) = σ(h · v).

    def policy_prob(h):
        return 1.0 / (
            1.0 + np.exp(-(np.asarray(h) @ vec)))

    def utility_fn(h, route):
        return 1.0 if route == Route.SYMBOLIC else 0.0
    h0 = np.zeros_like(vec)
    results = run_intervention_experiment(
        "demo", "candidate", h0, vec,
        policy_prob_fn=policy_prob,
        utility_fn=utility_fn,
        alphas=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )
    summary = dose_response_summary(results)
    print(json.dumps(summary, indent=2, default=str))
    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    return 0


def _cmd_calibrate(args) -> int:
    """Tune abstention / OOD thresholds on the calibration split."""
    print("Calibrate mode: tune τ_conf and τ_OOD on the calibration split.")
    print(
        "See EXPERIMENT_PROTOCOL_V0_3_10.md "
        "for the calibration protocol.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="daph-autolearn-policy",
        description="v0.3.10 counterfactual policy learner CLI.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # train
    p_train = sub.add_parser(
        "train",
        help="Train a weighted logistic / centroid policy.")
    p_train.add_argument(
        "--train-tasks", default=None,
        help="Training tasks JSONL.")
    p_train.add_argument(
        "--dev-tasks", default=None,
        help="Development tasks JSONL.")
    p_train.add_argument(
        "--synthetic", action="store_true",
        help="Use the synthetic closed-loop environment.")
    p_train.add_argument(
        "--output",
        default="artifacts/policy_train_result.json",
        help="Output JSON path.")
    _add_common_policy_args(p_train)
    p_train.set_defaults(func=_cmd_train)

    # evaluate
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate a trained policy on held-out tasks.")
    p_eval.add_argument(
        "--policy-file", default=None,
        help="Trained policy file.")
    p_eval.add_argument(
        "--test-tasks", default=None,
        help="Held-out test tasks JSONL.")
    p_eval.add_argument(
        "--output",
        default="artifacts/policy_eval_result.json",
        help="Output JSON path.")
    _add_common_policy_args(p_eval)
    p_eval.set_defaults(func=_cmd_evaluate)

    # intervene
    p_int = sub.add_parser(
        "intervene",
        help="Run causal intervention experiments.")
    p_int.add_argument(
        "--vector-file", default=None,
        help="Steering vector .npy file.")
    p_int.add_argument(
        "--output",
        default="artifacts/intervention_result.json",
        help="Output JSON path.")
    _add_common_policy_args(p_int)
    p_int.set_defaults(func=_cmd_intervene)

    # calibrate
    p_cal = sub.add_parser(
        "calibrate",
        help="Tune abstention / OOD thresholds.")
    p_cal.add_argument(
        "--calibration-tasks", default=None,
        help="Calibration tasks JSONL.")
    p_cal.add_argument(
        "--output",
        default="artifacts/calibration_result.json",
        help="Output JSON path.")
    _add_common_policy_args(p_cal)
    p_cal.set_defaults(func=_cmd_calibrate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
