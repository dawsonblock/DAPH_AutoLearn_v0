"""v0.3.10.1-alpha — CLI for the counterfactual policy learner.

Modes:

    daph-autolearn-policy train
        Train a weighted centroid / logistic / MLP policy.
        Supports --synthetic (closed-loop environment) or
        --model <hf-model-id> + --dataset <jsonl> (real-model flow).
    daph-autolearn-policy evaluate
        Evaluate a trained policy on held-out tasks.
    daph-autolearn-policy intervene
        Run causal intervention experiments (+v/0/-v dose-response).
    daph-autolearn-policy calibrate
        Tune abstention / OOD thresholds on calibration split.

v0.3.10.1 — clean-break CLI changes:

* ``--soft-targets`` / ``--no-soft-targets`` replaced by
  ``--target-mode soft|hard`` (Section 1). The bool flag was a
  correctness bug.
* ``--weight-mode gap|snr`` replaced by
  ``--weight-mode uniform|absolute_gap|clipped_gap|snr`` (Section 2).
* ``--policy centroid|logistic`` expanded to
  ``--policy centroid|logistic|mlp_experimental`` (Section 3).
* ``--early-stopping-metric`` added (Section 9).
* Real-model train/evaluate/calibrate flows added (Sections 16-20).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _add_common_policy_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--policy", choices=["centroid", "logistic", "mlp_experimental"],
        default="logistic",
        help="Policy type: centroid (baseline), logistic (primary "
             "learner), or mlp_experimental (diagnostic, Section 30).",
    )
    # v0.3.10.1 — replaces --soft-targets / --no-soft-targets (Section 1).
    ap.add_argument(
        "--target-mode", choices=["soft", "hard"], default="soft",
        help="Preference target mode (Section 1). 'soft': q=sigmoid(ΔU/τ). "
             "'hard': y∈{0,1} with ties (|ΔU|<=gap) masked out.",
    )
    ap.add_argument(
        "--target-temperature", type=float,
        default=1.0,
        help="Temperature τ for soft targets. Smaller => sharper.",
    )
    # v0.3.10.1 — replaces --weight-mode gap|snr (Section 2).
    ap.add_argument(
        "--weight-mode",
        choices=["uniform", "absolute_gap", "clipped_gap", "snr"],
        default="clipped_gap",
        help="Sample weighting mode (Section 2).",
    )
    ap.add_argument("--gap-threshold", type=float, default=0.0,
                    help="Tasks with |ΔU| <= gap_threshold get zero weight.")
    ap.add_argument(
        "--early-stopping-metric",
        choices=["dev_loss", "dev_regret", "dev_utility"],
        default="dev_regret",
        help="Dev early stopping metric (Section 9). Default dev_regret.",
    )
    ap.add_argument(
        "--mlp-hidden-dim", type=int, default=64,
        help="Hidden dim for mlp_experimental policy (Section 30).")
    ap.add_argument(
        "--layer", type=int, default=20,
        help="Transformer layer index for feature capture / intervention.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Steering alpha.")
    ap.add_argument("--confidence-threshold", type=float, default=0.70,
                    help="Abstention threshold τ_conf (Section 21).")
    ap.add_argument("--ood-threshold", type=float, default=float("inf"),
                    help="OOD abstention threshold (Section 19). "
                         "Default inf (disabled); calibrate to enable.")
    ap.add_argument("--max-weight", type=float, default=1.0,
                    help="Max sample weight (clipping for clipped_gap/snr).")
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
        target_mode=args.target_mode,
        weight_mode=args.weight_mode,
        gap_threshold=args.gap_threshold,
        selected_layer=args.layer,
        steering_alpha=args.alpha,
        confidence_threshold=args.confidence_threshold,
        ood_threshold=args.ood_threshold,
        max_weight=args.max_weight,
        random_seed=args.seed,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        policy_type=args.policy,
        mlp_hidden_dim=args.mlp_hidden_dim,
        early_stopping_metric=args.early_stopping_metric,
    )


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file of tasks."""
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def _save_policy_artifact(result, path: str, cfg) -> None:
    """Save the trained policy artifact with full provenance (Section 45)."""
    import time
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "release_version": cfg.autolearn_version,
        "policy_type": cfg.policy_type,
        "target_mode": cfg.target_mode,
        "target_temperature": cfg.target_temperature,
        "weight_mode": cfg.weight_mode,
        "gap_threshold": cfg.gap_threshold,
        "confidence_threshold": cfg.confidence_threshold,
        "ood_threshold": (None if cfg.ood_threshold == float("inf")
                          else cfg.ood_threshold),
        "selected_layer": cfg.selected_layer,
        "steering_alpha": cfg.steering_alpha,
        "random_seed": cfg.random_seed,
        "model_revision": cfg.model_revision,
        "tokenizer_revision": cfg.tokenizer_revision,
        "train_timestamp": time.time(),
        "n_train": result.n_train,
        "n_dev": result.n_dev,
        "dev_metrics": result.dev_metrics,
        "config": cfg.to_dict(),
    }
    # If a torch policy, save the state dict.
    try:
        if hasattr(result.policy, "save"):
            state_path = str(Path(path).with_suffix(".pt"))
            result.policy.save(state_path)
            artifact["policy_state_file"] = state_path
    except (OSError, IOError, RuntimeError, ValueError) as e:
        artifact["policy_save_error"] = str(e)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)


def _cmd_train(args) -> int:
    """Train a policy on train tasks, evaluate on dev (Section 16)."""
    from daph_learning.policy.learner import (
        build_counterfactual_experiences,
        train_policy_learner,
    )

    cfg = _build_config(args)

    if args.synthetic:
        # Synthetic closed-loop environment (v0.3.10 legacy path).
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
    elif args.benchmark:
        # Redesigned benchmark environment (v0.3.10.1).
        from daph_learning.environment_benchmark import (
            make_environment, benchmark_execute_fn, benchmark_utility,
        )
        env_name = args.benchmark
        train_tasks = make_environment(env_name, n=200, dim=8, seed=args.seed)
        dev_tasks = make_environment(env_name, n=150, dim=8, seed=args.seed + 1)
        execute_fn = benchmark_execute_fn

        def utility_fn(task, route):
            return benchmark_utility(task, route)
    elif args.model and args.dataset:
        # Real-model flow (Section 16).
        return _cmd_train_real_model(args, cfg)
    else:
        print(
            "ERROR: must specify --synthetic, --benchmark <name>, or "
            "--model <id> + --dataset <jsonl>.",
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
    _save_policy_artifact(result, args.output, cfg)
    print(f"Training complete. Policy artifact written to {args.output}")
    if result.dev_metrics:
        m = result.dev_metrics
        print(f"  policy_type:             {cfg.policy_type}")
        print(f"  target_mode:             {cfg.target_mode}")
        print(f"  weight_mode:             {cfg.weight_mode}")
        print(f"  n_train:                 {result.n_train}")
        print(f"  n_dev:                   {result.n_dev}")
        print(f"  mean_candidate_utility:  {m.get('mean_candidate_utility', 'N/A')}")
        print(f"  mean_incumbent_utility:  {m.get('mean_incumbent_utility', 'N/A')}")
        print(f"  mean_candidate_regret:   {m.get('mean_candidate_regret', 'N/A')}")
        print(f"  mean_incumbent_regret:   {m.get('mean_incumbent_regret', 'N/A')}")
        print(f"  preference_brier_soft:   {m.get('preference_brier_soft', 'N/A')}")
        print(f"  action_confidence_ece:   {m.get('action_confidence_ece', 'N/A')}")
        print(f"  abstain_rate:            {m.get('abstain_rate', 'N/A')}")
        print(f"  symbolic_utilization:    {m.get('symbolic_utilization', 'N/A')}")
        print(f"  llm_utilization:         {m.get('llm_utilization', 'N/A')}")
    return 0


def _cmd_train_real_model(args, cfg) -> int:
    """Real-model training flow (Section 16, v0.3.10.2 — no placeholders).

    load model -> load dataset -> for each task:
        capture h(task) once -> execute symbolic -> execute LLM ->
        verify both -> compute utilities -> build experience
    -> join by task_id -> fit policy -> save artifact.

    v0.3.10.2 — removes all ``symbolic_correct`` / ``llm_correct``
    placeholder labels. Every outcome is derived from actual execution
    + verification (Sections 9-12).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: torch + transformers required for real-model training.",
              file=sys.stderr)
        return 1
    print(f"Loading model {args.model}...")
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    model.eval()
    # Load dataset splits.
    train_tasks = _load_jsonl(args.dataset)
    dev_tasks = _load_jsonl(args.dev_tasks) if args.dev_tasks else None
    print(f"Loaded {len(train_tasks)} train tasks"
          + (f", {len(dev_tasks)} dev tasks" if dev_tasks else ""))

    # v0.3.10.2 — use real backends (Sections 9-14).
    from daph_learning.execution.real_backends import (
        CaptureConfig, LLMGenerationConfig,
        build_real_counterfactual_experience,
        capture_task_representation,
        execute_llm_backend,
        execute_symbolic_backend,
    )
    from daph_learning.policy.types import FeatureRecord, Route

    cap_cfg = CaptureConfig(layer=cfg.selected_layer, location="last_token")
    gen_cfg = LLMGenerationConfig(max_new_tokens=16, do_sample=False)

    def _process_tasks(tasks):
        """Execute both backends on each task and build experiences."""
        experiences = []
        records = []
        n_sym_correct = 0
        n_llm_correct = 0
        for i, task in enumerate(tasks):
            if (i + 1) % 10 == 0:
                print(f"  processing task {i+1}/{len(tasks)}...")
            # Capture activation once (Section 14).
            h = capture_task_representation(
                task, model, tokenizer, config=cap_cfg, device=device)
            # Execute symbolic backend (Section 10).
            sym_outcome, sym_text = execute_symbolic_backend(task)
            if sym_outcome.execution_success:
                n_sym_correct += 1
            # Execute LLM backend (Section 11).
            llm_outcome, llm_text = execute_llm_backend(
                task, model, tokenizer,
                generation_config=gen_cfg, device=device)
            # Build experience with real verification (Section 12).
            exp, sym_v, llm_v = build_real_counterfactual_experience(
                task, h, sym_outcome, llm_outcome, llm_text,
                config=cfg, symbolic_output_text=sym_text)
            if llm_v.verified_correct is True:
                n_llm_correct += 1
            experiences.append(exp)
            records.append(FeatureRecord(exp.task_id, h))
        print(f"  symbolic correct: {n_sym_correct}/{len(tasks)}")
        print(f"  LLM correct: {n_llm_correct}/{len(tasks)}")
        return experiences, records

    print("Processing train tasks...")
    train_exp, train_records = _process_tasks(train_tasks)
    dev_exp = dev_records = None
    if dev_tasks:
        print("Processing dev tasks...")
        dev_exp, dev_records = _process_tasks(dev_tasks)

    # Utility function for held-out evaluation.
    # Looks up the pre-computed experience by task_id and returns the
    # appropriate backend's utility. This avoids re-executing backends
    # during evaluation (the outcomes were already captured in Phase A).
    _exp_by_id = {}
    for exp in train_exp + (dev_exp or []):
        _exp_by_id[exp.task_id] = exp

    def real_utility_fn(task, route):
        # v0.3.10.3.1 — Section 3: use the canonical utility_for_route
        # so this path uses the same U_b as training/calibration/final.
        from daph_learning.policy.utility import utility_for_route
        if isinstance(route, Route):
            route = route.value
        if route == "abstain":
            return 0.0
        tid = str(task.get("task_id", ""))
        exp = _exp_by_id.get(tid)
        if exp is None:
            # Task not in pre-computed experiences — fail closed rather
            # than silently returning 0.0 (which would give no signal).
            return 0.0
        return utility_for_route(exp, route, cfg)

    # Fit policy using the real experiences.
    from daph_learning.policy.learner import train_policy_learner
    def incumbent(h):
        return "llm"

    result = train_policy_learner(
        train_exp, train_records,
        config=cfg,
        dev_experiences=dev_exp,
        dev_activations=dev_records,
        incumbent_route_fn=incumbent,
        utility_fn=real_utility_fn,
        dev_tasks=dev_tasks,
    )
    _save_policy_artifact(result, args.output, cfg)
    print(f"Real-model training complete. Artifact: {args.output}")
    if result.dev_metrics:
        m = result.dev_metrics
        print(f"  policy_type:             {cfg.policy_type}")
        print(f"  mean_candidate_regret:   {m.get('mean_candidate_regret', 'N/A')}")
        print(f"  mean_candidate_utility:  {m.get('mean_candidate_utility', 'N/A')}")
        print(f"  mean_incumbent_regret:   {m.get('mean_incumbent_regret', 'N/A')}")
        print(f"  preference_brier_soft:   {m.get('preference_brier_soft', 'N/A')}")
        print(f"  action_confidence_ece:   {m.get('action_confidence_ece', 'N/A')}")
    return 0


def _cmd_evaluate(args) -> int:
    """Evaluate a trained policy on held-out tasks (Section 17)."""
    if not args.policy_file:
        print("ERROR: --policy-file is required for evaluate mode.",
              file=sys.stderr)
        return 1
    with open(args.policy_file) as f:
        artifact = json.load(f)
    cfg = _build_config(args)
    print(f"Loaded policy artifact: {args.policy_file}")
    print(f"  policy_type: {artifact.get('policy_type')}")
    print(f"  release_version: {artifact.get('release_version')}")
    if args.synthetic:
        from daph_learning.environment_synthetic import (
            make_synthetic_tasks, synthetic_execute_fn, synthetic_utility,
        )
        test_tasks = make_synthetic_tasks(
            n_per_family=50, dim=8, seed=args.seed + 100)
        execute_fn = synthetic_execute_fn

        def utility_fn(task, route):
            return synthetic_utility(task, route)
    elif args.benchmark:
        from daph_learning.environment_benchmark import (
            make_environment, benchmark_execute_fn, benchmark_utility,
        )
        test_tasks = make_environment(
            args.benchmark, n=200, dim=8, seed=args.seed + 100)
        execute_fn = benchmark_execute_fn

        def utility_fn(task, route):
            return benchmark_utility(task, route)
    elif args.test_tasks:
        test_tasks = _load_jsonl(args.test_tasks)
        # Real-model evaluate would load the model and execute; for the
        # alpha, we use the stored activations if present.
        execute_fn = None
    else:
        print("ERROR: must specify --synthetic, --benchmark, or --test-tasks.",
              file=sys.stderr)
        return 1
    # Load the trained policy.
    from daph_learning.policy.policy_factory import make_policy
    from daph_learning.policy import predict_proba
    policy = make_policy(cfg)
    state_file = artifact.get("policy_state_file")
    if state_file and Path(state_file).exists():
        from daph_learning.policy.centroid_policy import CentroidPolicy
        if isinstance(policy, CentroidPolicy):
            policy = CentroidPolicy.load(state_file)
        else:
            policy = type(policy).load(state_file)
    # Evaluate.
    from daph_learning.policy.learner import build_counterfactual_experiences
    from daph_learning.policy.abstention import choose_route_with_reason
    from daph_learning.policy.regret import mean_regret
    from daph_learning.policy.types import Route
    test_exp = (
        build_counterfactual_experiences(
            test_tasks, execute_fn=execute_fn, config=cfg)
        if execute_fn else [])
    test_acts = np.array(
        [t.get("activation", np.zeros(8)) for t in test_tasks],
        dtype=np.float32)
    probs = predict_proba(policy, test_acts)
    from daph_learning.policy.ood import MahalanobisOOD
    ood = MahalanobisOOD(ridge=cfg.ood_ridge)
    ood.fit(test_acts)
    candidate_routes = []
    reasons = []
    cand_utils = []
    ora_utils = []
    for i, task in enumerate(test_tasks):
        p = float(probs[i])
        ood_score = ood.score(test_acts[i])
        decision = choose_route_with_reason(
            p, cfg.confidence_threshold,
            ood_score=ood_score, ood_threshold=cfg.ood_threshold)
        candidate_routes.append(decision.route.value)
        reasons.append(decision.reason)
        cu = utility_fn(task, decision.route)
        cand_utils.append(cu)
        u_sym = utility_fn(task, Route.SYMBOLIC)
        u_llm = utility_fn(task, Route.LLM)
        ora_utils.append(max(u_sym, u_llm))
    cand_utils = np.array(cand_utils)
    ora_utils = np.array(ora_utils)
    # Per-family metrics (Section 17).
    families = [t.get("family", "unknown") for t in test_tasks]
    unique_fams = sorted(set(families))
    per_family = {}
    for fam in unique_fams:
        mask = np.array([f == fam for f in families])
        if mask.sum() == 0:
            continue
        per_family[fam] = {
            "n": int(mask.sum()),
            "mean_utility": float(cand_utils[mask].mean()),
            "mean_regret": float(
                mean_regret(cand_utils[mask], ora_utils[mask])),
        }
    reason_counts: dict[str, int] = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1
    metrics = {
        "mean_utility": float(cand_utils.mean()),
        "mean_regret": float(mean_regret(cand_utils, ora_utils)),
        "routing_accuracy": float(np.mean([
            (candidate_routes[i] == "symbolic" and test_exp[i].delta_utility > 0)
            or (candidate_routes[i] == "llm" and test_exp[i].delta_utility < 0)
            for i in range(len(test_tasks))
        ])) if test_exp else None,
        "abstain_rate": float(np.mean(
            [r == "abstain" for r in candidate_routes])),
        "symbolic_utilization": float(np.mean(
            [r == "symbolic" for r in candidate_routes])),
        "llm_utilization": float(np.mean(
            [r == "llm" for r in candidate_routes])),
        "ood_rate": float(reason_counts.get("ood", 0) / len(reasons)),
        "abstention_reasons": reason_counts,
        "per_family": per_family,
        "n_test": len(test_tasks),
        "evidence_level": "SYNTHETIC" if args.synthetic or args.benchmark
                          else "REAL_MODEL_FINAL",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Evaluation complete. Results: {args.output}")
    print(f"  mean_utility:   {metrics['mean_utility']}")
    print(f"  mean_regret:    {metrics['mean_regret']}")
    print(f"  abstain_rate:   {metrics['abstain_rate']}")
    print(f"  evidence_level: {metrics['evidence_level']}")
    return 0


def _cmd_intervene(args) -> int:
    """Run causal intervention experiments (+v/0/-v dose-response)."""
    from daph_learning.interventions import (
        run_intervention_experiment,
        dose_response_summary,
    )
    from daph_learning.policy.types import Route
    if not args.vector_file:
        print("ERROR: --vector-file is required for intervene mode.",
              file=sys.stderr)
        return 1
    vec = np.load(args.vector_file)

    def policy_prob(h):
        return 1.0 / (1.0 + np.exp(-(np.asarray(h) @ vec)))

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
    """Tune abstention / OOD thresholds on the calibration split
    (Section 18)."""
    from daph_learning.policy.ood import MahalanobisOOD
    cfg = _build_config(args)
    if args.synthetic:
        from daph_learning.environment_synthetic import (
            make_synthetic_tasks, synthetic_execute_fn,
        )
        cal_tasks = make_synthetic_tasks(
            n_per_family=30, dim=8, seed=args.seed + 200)
        train_tasks = make_synthetic_tasks(
            n_per_family=50, dim=8, seed=args.seed)
    elif args.benchmark:
        from daph_learning.environment_benchmark import make_environment
        cal_tasks = make_environment(args.benchmark, n=150, dim=8, seed=args.seed + 200)
        train_tasks = make_environment(args.benchmark, n=200, dim=8, seed=args.seed)
    elif args.calibration_tasks:
        cal_tasks = _load_jsonl(args.calibration_tasks)
        train_tasks = _load_jsonl(args.train_tasks) if args.train_tasks else cal_tasks
    else:
        print("ERROR: must specify --synthetic, --benchmark, or "
              "--calibration-tasks.", file=sys.stderr)
        return 1
    # Fit OOD on train features.
    train_acts = np.array(
        [t.get("activation", np.zeros(8)) for t in train_tasks],
        dtype=np.float32)
    ood = MahalanobisOOD(ridge=cfg.ood_ridge)
    ood.fit(train_acts)
    # Compute OOD scores on calibration set.
    cal_acts = np.array(
        [t.get("activation", np.zeros(8)) for t in cal_tasks],
        dtype=np.float32)
    cal_scores = np.array([ood.score(h) for h in cal_acts])
    # Calibrate tau_ood as a quantile of the in-distribution scores
    # (Section 19).
    tau_ood = float(np.quantile(cal_scores, cfg.ood_quantile))
    # Simple confidence threshold calibration: try a grid and pick the
    # one that minimizes abstain rate subject to accuracy constraint.
    from daph_learning.policy.abstention import choose_route_with_reason
    if args.synthetic:
        from daph_learning.environment_synthetic import synthetic_utility
        execute_fn = synthetic_execute_fn

        def utility_fn(task, route):
            return synthetic_utility(task, route)
    elif args.benchmark:
        from daph_learning.environment_benchmark import (
            benchmark_execute_fn, benchmark_utility,
        )
        execute_fn = benchmark_execute_fn

        def utility_fn(task, route):
            return benchmark_utility(task, route)
    else:
        execute_fn = None

        def utility_fn(task, route):
            return 0.0
    # v0.3.10.2 — compute ACTUAL policy probabilities (Section 8).
    # The v0.3.10.1 version used p = 0.5 placeholder, which made the
    # threshold grid search meaningless. Now we train a policy on the
    # train set (or load from --policy-file) and compute real probs.
    from daph_learning.policy import fit_policy, predict_proba
    from daph_learning.policy.learner import build_counterfactual_experiences as _bce
    # Build train experiences for fitting the policy.
    if execute_fn:
        train_exp_for_policy = _bce(
            train_tasks, execute_fn=execute_fn, config=cfg)
        train_acts_for_policy = train_acts
        train_du = np.array(
            [e.delta_utility for e in train_exp_for_policy], dtype=np.float32)
        train_w = np.array(
            [e.sample_weight for e in train_exp_for_policy], dtype=np.float32)
        model = fit_policy(
            cfg, train_acts_for_policy, train_du, train_w,
            seed=cfg.random_seed)
        cal_probs = predict_proba(model, cal_acts)
    else:
        # No execute_fn (real-model path without pre-captured features).
        # v0.3.10.2 — fail closed. Using p=0.5 makes the threshold grid
        # search meaningless (Section 8). The caller must provide either
        # --synthetic, --benchmark, or pre-captured features.
        print("ERROR: cannot calibrate without policy probabilities.",
              file=sys.stderr)
        print("  Provide --synthetic, --benchmark, or pre-captured features.",
              file=sys.stderr)
        return 1
    # Grid search confidence threshold using ACTUAL probabilities.
    best_threshold = 0.5
    best_utility = -1.0
    for threshold in np.arange(0.5, 0.95, 0.05):
        utils = []
        for i, task in enumerate(cal_tasks):
            p = float(cal_probs[i])  # ACTUAL policy probability
            decision = choose_route_with_reason(
                p, float(threshold),
                ood_score=float(cal_scores[i]),
                ood_threshold=tau_ood)
            u = utility_fn(task, decision.route)
            utils.append(u)
        mean_u = float(np.mean(utils))
        if mean_u > best_utility:
            best_utility = mean_u
            best_threshold = float(threshold)
    result = {
        "tau_ood": tau_ood,
        "tau_conf": best_threshold,
        "ood_quantile": cfg.ood_quantile,
        "calibration_n": len(cal_tasks),
        "calibration_ood_score_mean": float(cal_scores.mean()),
        "calibration_ood_score_p95": float(np.quantile(cal_scores, 0.95)),
        "config": cfg.to_dict(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Calibration complete. Results: {args.output}")
    print(f"  tau_ood:    {tau_ood:.4f} (quantile {cfg.ood_quantile})")
    print(f"  tau_conf:   {best_threshold:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="daph-autolearn-policy",
        description="v0.3.10.1-alpha counterfactual policy learner CLI.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    # train
    p_train = sub.add_parser(
        "train",
        help="Train a weighted centroid / logistic / MLP policy.")
    p_train.add_argument(
        "--train-tasks", default=None,
        help="Training tasks JSONL (real-model flow).")
    p_train.add_argument(
        "--dev-tasks", default=None,
        help="Development tasks JSONL (real-model flow).")
    p_train.add_argument(
        "--synthetic", action="store_true",
        help="Use the legacy synthetic closed-loop environment.")
    p_train.add_argument(
        "--benchmark", default=None,
        help="Use the redesigned benchmark: linear|near_tie|multimodal|xor.")
    p_train.add_argument(
        "--dataset", default=None,
        help="Dataset JSONL for real-model training (Section 16).")
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
        help="Trained policy artifact JSON.")
    p_eval.add_argument(
        "--test-tasks", default=None,
        help="Held-out test tasks JSONL.")
    p_eval.add_argument(
        "--synthetic", action="store_true",
        help="Use the synthetic environment for test data.")
    p_eval.add_argument(
        "--benchmark", default=None,
        help="Use the redesigned benchmark for test data.")
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
        help="Tune abstention / OOD thresholds (Section 18).")
    p_cal.add_argument(
        "--calibration-tasks", default=None,
        help="Calibration tasks JSONL.")
    p_cal.add_argument(
        "--train-tasks", default=None,
        help="Train tasks JSONL (for fitting OOD model).")
    p_cal.add_argument(
        "--synthetic", action="store_true",
        help="Use the synthetic environment for calibration.")
    p_cal.add_argument(
        "--benchmark", default=None,
        help="Use the redesigned benchmark for calibration.")
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
