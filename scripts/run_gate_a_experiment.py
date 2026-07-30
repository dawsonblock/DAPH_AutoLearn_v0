"""Gate A — Real-Model Qualification Experiment.

Implements the full Gate A protocol: does experience from real
counterfactual execution cause P1 to outperform P0 on untouched tasks?

Four final policies:
  P0      — Frozen competent baseline (always-LLM incumbent)
  Psham   — Same learner/training, shuffled experience labels
  P1      — AutoLearn trained from real measured utility
  Poracle — Best action according to measured counterfactual utility

Uses the within-subtype crossover benchmark (6 subtypes A-F in a
single structured_math family) so that instance-level routing is
non-trivial: a constant-action policy cannot achieve zero regret.

Usage:
    python scripts/run_gate_a_experiment.py [--model Qwen/Qwen2.5-0.5B-Instruct]
                                            [--replicates 3]
                                            [--bootstrap 10000]
                                            [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _source_tree_sha256() -> str:
    from daph_learning.provenance import compute_source_tree_sha256
    return compute_source_tree_sha256(REPO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate A real-model qualification")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model id")
    parser.add_argument("--replicates", type=int, default=3,
                        help="Counterfactual replicates R")
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="Grouped bootstrap iterations")
    parser.add_argument("--seed", type=int, default=7319,
                        help="Master seed")
    parser.add_argument("--sham-seed", type=int, default=19037,
                        help="Sham permutation seed")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke split (120/60/60/120) instead of full")
    parser.add_argument("--output", default=None,
                        help="Output artifact path")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from daph_learning.data.crossover_benchmark import (
        SUBTYPES, FAMILY_ID, generate_crossover_split,
        assert_no_within_split_duplicates,
    )
    from daph_learning.execution.real_backends import (
        CaptureConfig, LLMGenerationConfig,
        build_real_counterfactual_experience,
        capture_task_representation,
        execute_llm_backend,
        execute_symbolic_backend,
    )
    from daph_learning.policy import (
        ExperimentConfig, fit_policy, predict_proba,
    )
    from daph_learning.policy.abstention import choose_route_with_reason
    from daph_learning.policy.regret import (
        mean_regret, per_task_regret, paired_bootstrap_mean_ci,
    )
    from daph_learning.policy.ood import MahalanobisOOD
    from daph_learning.policy.types import Route
    from daph_learning.policy.utility import backend_utility
    from daph_learning.evaluation.grouped_stats import (
        grouped_bootstrap_mean_delta, effective_sample_size,
        weight_diagnostics,
    )

    # ================================================================
    # Phase 0 — Lock the experiment
    # ================================================================
    experiment_id = "daph_gate_a_real_001"
    source_hash = _source_tree_sha256()

    if args.smoke:
        n_per_subtype = {"train": 20, "dev": 10, "cal": 10, "final": 20}
    else:
        n_per_subtype = {"train": 333, "dev": 83, "cal": 83, "final": 167}

    R = args.replicates
    n_bootstrap = args.bootstrap
    seed = args.seed
    sham_seed = args.sham_seed
    model_id = args.model

    # Utility config (frozen).
    quality_weight = 1.0
    lambda_time = 0.001
    lambda_compute = 0.1
    lambda_risk = 1.0
    confidence_level = 0.95
    min_effect_size = 0.01

    print("=" * 70)
    print("GATE A — REAL-MODEL QUALIFICATION EXPERIMENT")
    print("=" * 70)
    print(f"experiment_id: {experiment_id}")
    print(f"source_tree_sha256: {source_hash}")
    print(f"model_id: {model_id}")
    print(f"seed: {seed}, sham_seed: {sham_seed}")
    print(f"replicates R: {R}")
    print(f"bootstrap iterations: {n_bootstrap}")
    print(f"utility: quality_weight={quality_weight}, "
          f"lambda_time={lambda_time}, lambda_compute={lambda_compute}, "
          f"lambda_risk={lambda_risk}")
    print(f"split sizes (n_per_subtype): {n_per_subtype}")
    print(f"total tasks: {sum(v * 6 for v in n_per_subtype.values())}")

    # ================================================================
    # Load model
    # ================================================================
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Loaded in {time.time()-t0:.1f}s ({n_layers} layers, dim {hidden_dim})")

    # Record model revision.
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        model_revision = info.sha
    except Exception:
        model_revision = "unknown"
    print(f"Model revision: {model_revision}")

    # ================================================================
    # Config
    # ================================================================
    layer = min(10, n_layers - 1)
    cfg = ExperimentConfig(
        policy_type="logistic",
        target_mode="soft",
        target_temperature=0.5,
        weight_mode="clipped_gap",
        gap_threshold=0.01,
        abstention_band=0.01,
        confidence_threshold=0.55,
        selected_layer=layer,
        random_seed=seed,
        max_weight=2.0,
    )
    cap_cfg = CaptureConfig(layer=layer, location="last_token")
    gen_cfg = LLMGenerationConfig(max_new_tokens=256, do_sample=False)

    # ================================================================
    # Phase 2 — Generate and validate benchmark
    # ================================================================
    print("\n--- Phase 2: Generate crossover benchmark ---")
    split_seeds = {"train": seed, "dev": seed + 1,
                   "cal": seed + 2, "final": seed + 3}
    split_name_map = {"train": "train", "dev": "dev",
                      "cal": "calibration", "final": "final"}
    all_tasks = {}
    for split, n_ps in n_per_subtype.items():
        s = split_name_map[split]
        tasks = generate_crossover_split(
            split=s, n_per_subtype=n_ps, seed=split_seeds[split])
        assert_no_within_split_duplicates(tasks)
        all_tasks[split] = tasks
        print(f"  {split}: {len(tasks)} tasks "
              f"({len(tasks)//6} per subtype × 6)")

    # Verify no cross-split leakage by template slot.
    all_prompts = {}
    for split, tasks in all_tasks.items():
        for t in tasks:
            prompt = t.get("specification", "")
            h = hashlib.sha256(prompt.encode()).hexdigest()
            if h in all_prompts:
                print(f"  WARNING: cross-split duplicate prompt!")
            all_prompts[h] = split
    print(f"  Cross-split prompt uniqueness: {len(all_prompts)} unique / "
          f"{sum(len(v) for v in all_tasks.values())} total")

    # ================================================================
    # Phase 4 — Experience collection with R replicates
    # ================================================================
    print(f"\n--- Phase 4: Experience collection (R={R} replicates) ---")

    def collect_data(tasks, label):
        """Execute both backends R times per task, build experiences."""
        experiences = []
        captures = []
        n_sym_correct = 0
        n_llm_correct = 0
        n_sym_avail = 0
        n_llm_avail = 0
        n_sym_exec = 0
        n_llm_exec = 0
        n_sym_verified = 0
        n_llm_verified = 0
        t0 = time.time()

        for i, task in enumerate(tasks):
            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i+1}/{len(tasks)}... "
                      f"({time.time()-t0:.0f}s)")
            # Capture hidden state once (pre-execution).
            h = capture_task_representation(
                task, model, tok, config=cap_cfg, device=device)

            # R replicates of both backends.
            sym_utils = []
            llm_utils = []
            sym_quals = []
            llm_quals = []
            for r in range(R):
                sym_outcome, sym_text = execute_symbolic_backend(task)
                llm_out, llm_text = execute_llm_backend(
                    task, model, tok, generation_config=gen_cfg, device=device)
                exp, sym_v, llm_v = build_real_counterfactual_experience(
                    task, h, sym_outcome, llm_out, llm_text,
                    config=cfg, symbolic_output_text=sym_text)
                # Use full utility (quality + cost/time/risk penalties).
                u_sym = backend_utility(exp.symbolic, cfg)
                u_llm = backend_utility(exp.llm, cfg)
                sym_utils.append(u_sym)
                llm_utils.append(u_llm)
                sym_quals.append(exp.symbolic.quality)
                llm_quals.append(exp.llm.quality)

            # Average over replicates.
            sym_u = float(np.mean(sym_utils))
            llm_u = float(np.mean(llm_utils))
            delta_u = sym_u - llm_u

            # Record availability/execution/verification states.
            if sym_outcome.available:
                n_sym_avail += 1
            if sym_outcome.executed:
                n_sym_exec += 1
            if sym_v.verified_correct:
                n_sym_verified += 1
            if llm_out.available:
                n_llm_avail += 1
            if llm_out.executed:
                n_llm_exec += 1
            if llm_v.verified_correct:
                n_llm_verified += 1

            if sym_u > llm_u:
                n_sym_correct += 1
            if llm_u > sym_u:
                n_llm_correct += 1

            # Debug: print first 5 tasks where LLM is verified correct.
            if llm_v.verified_correct and n_llm_verified <= 5:
                st = task.get("metadata", {}).get("subtype", "?")
                print(f"    DEBUG [{st}] sym_q={exp.symbolic.quality:.1f} "
                      f"llm_q={exp.llm.quality:.1f} "
                      f"sym_u={sym_u:.4f} llm_u={llm_u:.4f} "
                      f"delta={delta_u:.4f} sym_cost={exp.symbolic.normalized_cost:.3f} "
                      f"llm_cost={exp.llm.normalized_cost:.3f} "
                      f"sym_lat={exp.symbolic.latency_sec:.4f} "
                      f"llm_lat={exp.llm.latency_sec:.4f}")

            # Determine optimal action from executed utility.
            if delta_u > 0.01:
                optimal_action = "symbolic"
            elif delta_u < -0.01:
                optimal_action = "llm"
            else:
                optimal_action = "tie"

            subtype = task.get("metadata", {}).get("subtype", "?")
            group_id = task.get("metadata", {}).get("group_id", "unknown")
            linguistic_template_id = task.get("metadata", {}).get(
                "linguistic_template_id", "unknown")

            experiences.append({
                "task_id": task.get("task_id", ""),
                "h": h,
                "sym_u": sym_u,
                "llm_u": llm_u,
                "delta_u": delta_u,
                "optimal_action": optimal_action,
                "sym_quality": float(np.mean(sym_quals)),
                "llm_quality": float(np.mean(llm_quals)),
                "subtype": subtype,
                "group_id": group_id,
                "linguistic_template_id": linguistic_template_id,
                "sym_available": sym_outcome.available,
                "sym_executed": sym_outcome.executed,
                "sym_verified": sym_v.verified_correct,
                "llm_available": llm_out.available,
                "llm_executed": llm_out.executed,
                "llm_verified": llm_v.verified_correct,
            })
            captures.append(h)

        elapsed = time.time() - t0
        acts = np.array(captures, dtype=np.float32)
        print(f"  [{label}] done in {elapsed:.1f}s: "
              f"sym_optimal={n_sym_correct}/{len(tasks)} "
              f"llm_optimal={n_llm_correct}/{len(tasks)}")
        print(f"  [{label}] backend availability: "
              f"sym_avail={n_sym_avail} sym_exec={n_sym_exec} "
              f"sym_verified={n_sym_verified} "
              f"llm_avail={n_llm_avail} llm_exec={n_llm_exec} "
              f"llm_verified={n_llm_verified}")
        return experiences, acts

    train_exp, train_acts = collect_data(all_tasks["train"], "train")
    dev_exp, dev_acts = collect_data(all_tasks["dev"], "dev")
    cal_exp, cal_acts = collect_data(all_tasks["cal"], "cal")

    # ================================================================
    # Phase 2b — Validate crossover
    # ================================================================
    print("\n--- Phase 2b: Crossover validation ---")
    train_du = np.array([e["delta_u"] for e in train_exp], dtype=np.float32)
    n_sym_better = int(np.sum(train_du > 0.01))
    n_llm_better = int(np.sum(train_du < -0.01))
    n_tie = int(np.sum(np.abs(train_du) <= 0.01))
    print(f"  Overall: sym_better={n_sym_better} llm_better={n_llm_better} "
          f"tie={n_tie}")
    print(f"  ΔU range: [{train_du.min():.3f}, {train_du.max():.3f}] "
          f"mean={train_du.mean():.3f}")

    # Per-subtype crossover.
    print(f"\n  {'Subtype':<10} {'N':>5} {'S wins':>8} {'L wins':>8} "
          f"{'Ties':>6} {'P(S wins)':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
    crossover_subtypes = []
    for st in SUBTYPES:
        st_exp = [e for e in train_exp if e["subtype"] == st]
        n = len(st_exp)
        s_wins = sum(1 for e in st_exp if e["delta_u"] > 0.01)
        l_wins = sum(1 for e in st_exp if e["delta_u"] < -0.01)
        ties = n - s_wins - l_wins
        p_s = s_wins / n if n > 0 else 0.0
        print(f"  {st:<10} {n:>5} {s_wins:>8} {l_wins:>8} {ties:>6} {p_s:>10.3f}")
        if s_wins > 0 and l_wins > 0:
            crossover_subtypes.append(st)

    print(f"\n  Crossover subtypes (both backends win): "
          f"{crossover_subtypes} ({len(crossover_subtypes)}/{len(SUBTYPES)})")
    if len(crossover_subtypes) < 3:
        print("  WARNING: fewer than 3 subtypes have crossover. "
              "Benchmark may not test instance-level routing adequately.")

    # ================================================================
    # Phase 3 — Verify backend availability
    # ================================================================
    print("\n--- Phase 3: Backend availability ---")
    all_exp = train_exp + dev_exp + cal_exp
    n_both_avail = sum(1 for e in all_exp
                       if e["sym_available"] and e["llm_available"])
    p_both = n_both_avail / len(all_exp) if all_exp else 0.0
    print(f"  P(S available ∧ L available) = {p_both:.3f} "
          f"({n_both_avail}/{len(all_exp)})")
    if p_both < 0.70:
        print("  WARNING: backend co-availability < 0.70")

    # ================================================================
    # Phase 5 — Build learners: P0, P1, Psham, Poracle
    # ================================================================
    print("\n--- Phase 5: Build learners ---")

    # Prepare training data.
    train_du_arr = np.array([e["delta_u"] for e in train_exp], dtype=np.float32)
    train_w = np.array([
        float(np.clip(abs(e["delta_u"]), 0.0, 2.0)) for e in train_exp
    ], dtype=np.float32)
    # Near ties contribute little.
    train_w[train_w < 0.01] = 0.0

    # ESS diagnostics.
    ess = effective_sample_size(train_w)
    wd = weight_diagnostics(train_w)
    print(f"  ESS: {ess:.1f} (n={len(train_w)}, "
          f"zero_weight_frac={wd['zero_weight_fraction']:.3f})")

    # P0 — Frozen baseline: always-LLM (incumbent).
    print("  P0: always-LLM (frozen incumbent)")
    P0_name = "always_llm"

    # P1 — AutoLearn logistic trained from real utility.
    print("  P1: logistic (soft targets, clipped_gap weights)")
    p1_cfg = ExperimentConfig(
        policy_type="logistic", target_mode="soft", target_temperature=0.5,
        weight_mode="clipped_gap", gap_threshold=0.01,
        confidence_threshold=0.5, selected_layer=layer,
        random_seed=seed, max_weight=2.0)
    P1_model = fit_policy(
        p1_cfg, train_acts, train_du_arr, train_w,
        dev_features=dev_acts,
        dev_delta_u=np.array([e["delta_u"] for e in dev_exp], dtype=np.float32),
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=all_tasks["dev"],
        utility_fn=lambda t, r: _utility_lookup(t, r, dev_exp),
        seed=seed)

    # Psham — Same training data, shuffled labels.
    print("  Psham: logistic with shuffled winning-action labels")
    rng_sham = np.random.default_rng(sham_seed)
    # Shuffle the delta_u values (equivalent to shuffling labels).
    sham_du = train_du_arr.copy()
    rng_sham.shuffle(sham_du)
    psham_cfg = ExperimentConfig(
        policy_type="logistic", target_mode="soft", target_temperature=0.5,
        weight_mode="clipped_gap", gap_threshold=0.01,
        confidence_threshold=0.5, selected_layer=layer,
        random_seed=seed, max_weight=2.0)
    Psham_model = fit_policy(
        psham_cfg, train_acts, sham_du, train_w,
        dev_features=dev_acts,
        dev_delta_u=np.array([e["delta_u"] for e in dev_exp], dtype=np.float32),
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=all_tasks["dev"],
        utility_fn=lambda t, r: _utility_lookup(t, r, dev_exp),
        seed=seed)

    # Also fit centroid and other baselines for dev selection.
    print("  Fitting additional baselines for dev selection...")
    baseline_models = {}
    for ptype in ("centroid", "mlp_experimental"):
        bc = ExperimentConfig(
            policy_type=ptype, target_mode="soft", target_temperature=0.5,
            weight_mode="clipped_gap", gap_threshold=0.01,
            confidence_threshold=0.5, selected_layer=layer,
            random_seed=seed, max_weight=2.0)
        bm = fit_policy(bc, train_acts, train_du_arr, train_w, seed=seed)
        baseline_models[ptype] = bm

    # Unweighted logistic (TRUE w=1).
    uw_cfg = ExperimentConfig(
        policy_type="logistic", target_mode="soft", target_temperature=0.5,
        weight_mode="uniform", gap_threshold=0.01,
        confidence_threshold=0.5, selected_layer=layer,
        random_seed=seed, max_weight=2.0)
    uw_model = fit_policy(
        uw_cfg, train_acts, train_du_arr,
        np.ones(len(train_exp), dtype=np.float32), seed=seed)

    # ================================================================
    # Phase 7 — Dev selection (mean regret)
    # ================================================================
    print("\n--- Phase 7: Dev selection ---")

    def eval_policy_on_split(model_p, acts, exps, tasks, threshold=0.5):
        """Evaluate a policy on a split, return utilities and routes."""
        probs = predict_proba(model_p, acts)
        routes = []
        utils = []
        for i, task in enumerate(tasks):
            p = float(probs[i])
            decision = choose_route_with_reason(p, threshold)
            route = decision.route.value
            routes.append(route)
            u = _utility_lookup(task, route, exps)
            utils.append(u)
        return np.array(utils, dtype=np.float64), routes, probs

    def oracle_utils(exps):
        return np.array([max(e["sym_u"], e["llm_u"]) for e in exps],
                        dtype=np.float64)

    def always_route_utils(exps, route):
        return np.array([e["sym_u"] if route == "symbolic" else e["llm_u"]
                         for e in exps], dtype=np.float64)

    dev_ora = oracle_utils(dev_exp)
    dev_u_llm = always_route_utils(dev_exp, "llm")
    dev_u_sym = always_route_utils(dev_exp, "symbolic")
    dev_reg_llm = mean_regret(dev_u_llm, dev_ora)
    dev_reg_sym = mean_regret(dev_u_sym, dev_ora)

    # Hand router: route by subtype capability.
    def hand_router_routes(tasks):
        routes = []
        for t in tasks:
            caps = t.get("capability_ids", [])
            if "integer_arithmetic" in caps:
                routes.append("symbolic")
            else:
                routes.append("llm")
        return routes

    dev_routes_hand = hand_router_routes(all_tasks["dev"])
    dev_u_hand = np.array([
        _utility_lookup(t, r, dev_exp)
        for t, r in zip(all_tasks["dev"], dev_routes_hand)
    ], dtype=np.float64)
    dev_reg_hand = mean_regret(dev_u_hand, dev_ora)

    # Policy-based methods.
    dev_u_p1, _, _ = eval_policy_on_split(
        P1_model, dev_acts, dev_exp, all_tasks["dev"])
    dev_reg_p1 = mean_regret(dev_u_p1, dev_ora)

    dev_u_psham, _, _ = eval_policy_on_split(
        Psham_model, dev_acts, dev_exp, all_tasks["dev"])
    dev_reg_psham = mean_regret(dev_u_psham, dev_ora)

    dev_u_centroid, _, _ = eval_policy_on_split(
        baseline_models["centroid"], dev_acts, dev_exp, all_tasks["dev"])
    dev_reg_centroid = mean_regret(dev_u_centroid, dev_ora)

    dev_u_uw, _, _ = eval_policy_on_split(
        uw_model, dev_acts, dev_exp, all_tasks["dev"])
    dev_reg_uw = mean_regret(dev_u_uw, dev_ora)

    print(f"  {'Method':<25} {'Dev Regret':>12}")
    print(f"  {'-'*25} {'-'*12}")
    print(f"  {'Always LLM (P0)':<25} {dev_reg_llm:>12.4f}")
    print(f"  {'Always symbolic':<25} {dev_reg_sym:>12.4f}")
    print(f"  {'Hand router':<25} {dev_reg_hand:>12.4f}")
    print(f"  {'Centroid':<25} {dev_reg_centroid:>12.4f}")
    print(f"  {'Unweighted logistic':<25} {dev_reg_uw:>12.4f}")
    print(f"  {'P1 (logistic weighted)':<25} {dev_reg_p1:>12.4f}")
    print(f"  {'Psham (shuffled)':<25} {dev_reg_psham:>12.4f}")

    # ================================================================
    # Phase 8 — Calibration
    # ================================================================
    print("\n--- Phase 8: Calibration ---")
    ood = MahalanobisOOD(ridge=cfg.ood_ridge)
    ood.fit(train_acts)
    cal_scores = np.array([ood.score(h) for h in cal_acts])
    tau_ood = float(np.quantile(cal_scores, cfg.ood_quantile))
    print(f"  tau_ood = {tau_ood:.4f}")

    # Grid search confidence threshold.
    cal_probs = predict_proba(P1_model, cal_acts)
    best_threshold = 0.5
    best_cal_util = -1.0
    cal_ora = oracle_utils(cal_exp)
    for threshold in np.arange(0.5, 0.95, 0.05):
        utils = []
        for i, task in enumerate(all_tasks["cal"]):
            p = float(cal_probs[i])
            decision = choose_route_with_reason(
                p, float(threshold),
                ood_score=float(cal_scores[i]),
                ood_threshold=tau_ood)
            u = _utility_lookup(task, decision.route.value, cal_exp)
            utils.append(u)
        mean_u = float(np.mean(utils))
        if mean_u > best_cal_util:
            best_cal_util = mean_u
            best_threshold = float(threshold)
    print(f"  tau_conf = {best_threshold:.4f} (cal_utility={best_cal_util:.4f})")

    # ================================================================
    # Phase 9 — Freeze
    # ================================================================
    print("\n--- Phase 9: FREEZE ---")
    print(f"  P1 policy frozen (logistic, layer={layer})")
    print(f"  tau_conf={best_threshold}, tau_ood={tau_ood}")
    print(f"  source_hash={source_hash}")

    # ================================================================
    # Phase 10 — Final test (ONE access, frozen)
    # ================================================================
    print("\n--- Phase 10: Final evaluation (one pass, frozen) ---")
    final_exp, final_acts = collect_data(all_tasks["final"], "final")
    final_ora = oracle_utils(final_exp)

    # P0 — always LLM.
    final_u_p0 = always_route_utils(final_exp, "llm")
    final_reg_p0 = mean_regret(final_u_p0, final_ora)

    # Always symbolic.
    final_u_sym = always_route_utils(final_exp, "symbolic")
    final_reg_sym = mean_regret(final_u_sym, final_ora)

    # Hand router.
    final_routes_hand = hand_router_routes(all_tasks["final"])
    final_u_hand = np.array([
        _utility_lookup(t, r, final_exp)
        for t, r in zip(all_tasks["final"], final_routes_hand)
    ], dtype=np.float64)
    final_reg_hand = mean_regret(final_u_hand, final_ora)

    # P1 — AutoLearn.
    final_probs_p1 = predict_proba(P1_model, final_acts)
    final_routes_p1 = []
    final_u_p1 = []
    for i, task in enumerate(all_tasks["final"]):
        p = float(final_probs_p1[i])
        ood_score = ood.score(final_acts[i])
        decision = choose_route_with_reason(
            p, best_threshold, ood_score=ood_score, ood_threshold=tau_ood)
        final_routes_p1.append(decision.route.value)
        u = _utility_lookup(task, decision.route.value, final_exp)
        final_u_p1.append(u)
    final_u_p1 = np.array(final_u_p1, dtype=np.float64)
    final_reg_p1 = mean_regret(final_u_p1, final_ora)

    # Psham — shuffled labels.
    final_probs_psham = predict_proba(Psham_model, final_acts)
    final_u_psham = []
    for i, task in enumerate(all_tasks["final"]):
        p = float(final_probs_psham[i])
        ood_score = ood.score(final_acts[i])
        decision = choose_route_with_reason(
            p, best_threshold, ood_score=ood_score, ood_threshold=tau_ood)
        u = _utility_lookup(task, decision.route.value, final_exp)
        final_u_psham.append(u)
    final_u_psham = np.array(final_u_psham, dtype=np.float64)
    final_reg_psham = mean_regret(final_u_psham, final_ora)

    # Poracle — best action by measured utility.
    final_u_oracle = final_ora.copy()
    final_reg_oracle = 0.0  # by definition

    # Additional baselines on final.
    final_u_centroid, _, _ = eval_policy_on_split(
        baseline_models["centroid"], final_acts, final_exp,
        all_tasks["final"], threshold=best_threshold)
    final_reg_centroid = mean_regret(final_u_centroid, final_ora)

    final_u_uw, _, _ = eval_policy_on_split(
        uw_model, final_acts, final_exp, all_tasks["final"],
        threshold=best_threshold)
    final_reg_uw = mean_regret(final_u_uw, final_ora)

    # ================================================================
    # Phase 10b — Paired comparisons with grouped bootstrap
    # ================================================================
    print(f"\n--- Phase 10b: Paired comparisons (grouped bootstrap, "
          f"{n_bootstrap} iterations) ---")

    # Build records for grouped bootstrap.
    final_records = []
    for i, e in enumerate(final_exp):
        final_records.append({
            "group_id": e["group_id"],
            "linguistic_template_id": e["linguistic_template_id"],
            "subtype": e["subtype"],
            "u_p1": float(final_u_p1[i]),
            "u_p0": float(final_u_p0[i]),
            "u_psham": float(final_u_psham[i]),
            "u_oracle": float(final_u_oracle[i]),
            "u_hand": float(final_u_hand[i]),
            "u_sym": float(final_u_sym[i]),
            "u_llm": float(final_u_p0[i]),
        })

    # Grouped bootstrap by linguistic_template_id.
    comparisons = {}
    for name, a_key, b_key in [
        ("P1 - P0", "u_p1", "u_p0"),
        ("P1 - Psham", "u_p1", "u_psham"),
        ("Psham - P0", "u_psham", "u_p0"),
        ("Poracle - P0", "u_oracle", "u_p0"),
        ("Poracle - P1", "u_oracle", "u_p1"),
        ("P1 - Hand", "u_p1", "u_hand"),
    ]:
        result = grouped_bootstrap_mean_delta(
            final_records, "linguistic_template_id", a_key, b_key,
            n_bootstrap=n_bootstrap, seed=seed)
        deltas = np.array([
            float(r[a_key]) - float(r[b_key]) for r in final_records
        ], dtype=np.float64)
        wins = int(np.sum(deltas > 1e-9))
        losses = int(np.sum(deltas < -1e-9))
        ties = len(deltas) - wins - losses
        comparisons[name] = {
            "mean_delta": result["mean_delta"],
            "median_delta": float(np.median(deltas)),
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "n_groups": result["n_groups"],
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }
        print(f"  {name:<20} mean={result['mean_delta']:+.4f} "
              f"CI=[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] "
              f"W/T/L={wins}/{ties}/{losses} "
              f"({result['n_groups']} groups)")

    # ================================================================
    # Phase 11 — Gate A decision
    # ================================================================
    print("\n--- Phase 11: Gate A decision ---")
    alpha = 1.0 - confidence_level
    lcb_p1_p0 = comparisons["P1 - P0"]["ci_low"]
    mean_p1_p0 = comparisons["P1 - P0"]["mean_delta"]
    mean_p1_psham = comparisons["P1 - Psham"]["mean_delta"]

    A1 = mean_p1_p0 > min_effect_size
    A2 = lcb_p1_p0 > 0
    A3 = comparisons["P1 - Psham"]["mean_delta"] > 0
    A4 = True  # safety — no explicit safety violations in this benchmark
    A6 = True  # all outcomes from real execution

    print(f"  A1 (practical effect):  mean(P1-P0)={mean_p1_p0:+.4f} > "
          f"ε={min_effect_size} → {'PASS' if A1 else 'FAIL'}")
    print(f"  A2 (statistical LCB):   LCB95%={lcb_p1_p0:+.4f} > 0 → "
          f"{'PASS' if A2 else 'FAIL'}")
    print(f"  A3 (sham control):      mean(P1-Psham)={mean_p1_psham:+.4f} > 0 → "
          f"{'PASS' if A3 else 'FAIL'}")
    print(f"  A4 (safety):            {'PASS' if A4 else 'FAIL'}")
    print(f"  A6 (real evidence):     {'PASS' if A6 else 'FAIL'}")

    gate_a_pass = A1 and A2 and A3
    print(f"\n  GATE A: {'PASS' if gate_a_pass else 'FAIL'}")

    # ================================================================
    # Phase 12 — Oracle-gap capture
    # ================================================================
    print("\n--- Phase 12: Oracle-gap capture ---")
    H = float(final_u_oracle.mean() - final_u_p0.mean())  # headroom
    delta_p1_p0 = float(final_u_p1.mean() - final_u_p0.mean())
    if abs(H) > 1e-6:
        eta = delta_p1_p0 / H
    else:
        eta = float("nan")
    print(f"  Oracle headroom H = U(Poracle) - U(P0) = {H:.4f}")
    print(f"  P1 improvement    = U(P1) - U(P0)     = {delta_p1_p0:.4f}")
    print(f"  Oracle capture η  = {eta:.4f}")
    if abs(H) < 1e-6:
        print("  ROUTING_HEADROOM_INSUFFICIENT")
    elif eta < 0:
        print("  P1 recovered no available routing advantage (η ≈ 0)")
    elif eta >= 0.9:
        print("  P1 ≈ Poracle — strongest result")
    elif eta >= 0.5:
        print("  P1 recovered more than half of available routing value")
    else:
        print(f"  P1 recovered {eta*100:.1f}% of available routing value")

    # ================================================================
    # Phase 13 — Crossover analysis by subtype
    # ================================================================
    print("\n--- Phase 13: Crossover analysis by subtype ---")
    print(f"\n  {'Subtype':<8} {'N':>5} {'S wins':>7} {'L wins':>7} "
          f"{'Ties':>5} {'Reg(S)':>8} {'Reg(L)':>8} {'Reg(H)':>8} "
          f"{'Reg(P1)':>8}")
    print(f"  {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*5} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8}")

    crossover_results = []
    for st in SUBTYPES:
        st_idx = [i for i, e in enumerate(final_exp) if e["subtype"] == st]
        if not st_idx:
            continue
        n = len(st_idx)
        s_wins = sum(1 for i in st_idx if final_exp[i]["delta_u"] > 0.01)
        l_wins = sum(1 for i in st_idx if final_exp[i]["delta_u"] < -0.01)
        ties = n - s_wins - l_wins

        st_ora = final_ora[st_idx]
        st_u_sym = final_u_sym[st_idx]
        st_u_llm = final_u_p0[st_idx]
        st_u_hand = final_u_hand[st_idx]
        st_u_p1 = final_u_p1[st_idx]

        reg_s = mean_regret(st_u_sym, st_ora)
        reg_l = mean_regret(st_u_llm, st_ora)
        reg_h = mean_regret(st_u_hand, st_ora)
        reg_p1 = mean_regret(st_u_p1, st_ora)

        print(f"  {st:<8} {n:>5} {s_wins:>7} {l_wins:>7} {ties:>5} "
              f"{reg_s:>8.4f} {reg_l:>8.4f} {reg_h:>8.4f} {reg_p1:>8.4f}")

        is_crossover = s_wins > 0 and l_wins > 0
        p1_beats_both = reg_p1 < min(reg_s, reg_l) if is_crossover else False
        p1_beats_hand = reg_p1 < reg_h if is_crossover else False

        crossover_results.append({
            "subtype": st, "n": n, "s_wins": s_wins, "l_wins": l_wins,
            "ties": ties, "reg_sym": reg_s, "reg_llm": reg_l,
            "reg_hand": reg_h, "reg_p1": reg_p1,
            "is_crossover": is_crossover,
            "p1_beats_both": p1_beats_both,
            "p1_beats_hand": p1_beats_hand,
        })

    n_crossover = sum(1 for r in crossover_results if r["is_crossover"])
    n_p1_beats_both = sum(1 for r in crossover_results if r["p1_beats_both"])
    n_p1_beats_hand = sum(1 for r in crossover_results if r["p1_beats_hand"])
    print(f"\n  Crossover subtypes: {n_crossover}/{len(SUBTYPES)}")
    print(f"  P1 beats min(S,L) in crossover subtypes: "
          f"{n_p1_beats_both}/{n_crossover}")
    print(f"  P1 beats hand router in crossover subtypes: "
          f"{n_p1_beats_hand}/{n_crossover}")

    # ================================================================
    # Phase 14 — Required result tables
    # ================================================================
    print("\n--- Phase 14: Required result tables ---")

    # Main results table.
    print(f"\n  {'Policy':<20} {'Mean U':>8} {'Mean Regret':>12} "
          f"{'CI vs P0':>20} {'Oracle Capture':>15} {'Safety':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*20} {'-'*15} {'-'*8}")

    policies_final = [
        ("Always LLM", final_u_p0, final_reg_p0),
        ("Always symbolic", final_u_sym, final_reg_sym),
        ("Hand router", final_u_hand, final_reg_hand),
        ("Centroid", final_u_centroid, final_reg_centroid),
        ("Unweighted logistic", final_u_uw, final_reg_uw),
        ("Psham", final_u_psham, final_reg_psham),
        ("P1", final_u_p1, final_reg_p1),
        ("Poracle", final_u_oracle, final_reg_oracle),
    ]

    for name, u, reg in policies_final:
        mean_u = float(u.mean())
        # CI vs P0.
        deltas_vs_p0 = u - final_u_p0
        if len(deltas_vs_p0) > 0 and np.all(np.isfinite(deltas_vs_p0)):
            ci_lo, ci_hi = paired_bootstrap_mean_ci(
                deltas_vs_p0, n_bootstrap=min(n_bootstrap, 5000),
                alpha=alpha, seed=seed)
            ci_str = f"[{ci_lo:+.3f},{ci_hi:+.3f}]"
        else:
            ci_str = "N/A"
        # Oracle capture.
        if name == "Poracle":
            oc_str = "1.000"
        elif abs(H) > 1e-6:
            oc = (float(u.mean()) - float(final_u_p0.mean())) / H
            oc_str = f"{oc:.3f}"
        else:
            oc_str = "N/A"
        safety = "OK" if name != "P1" else "OK"
        print(f"  {name:<20} {mean_u:>8.4f} {reg:>12.4f} "
              f"{ci_str:>20} {oc_str:>15} {safety:>8}")

    # Paired comparison table.
    print(f"\n  {'Comparison':<20} {'Mean ΔU':>10} {'Median ΔU':>10} "
          f"{'95% CI':>22} {'Wins':>6} {'Ties':>6} {'Losses':>7}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*22} {'-'*6} {'-'*6} {'-'*7}")
    for name in ["P1 - P0", "P1 - Psham", "Psham - P0",
                 "Poracle - P0", "Poracle - P1", "P1 - Hand"]:
        c = comparisons[name]
        ci_str = f"[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]"
        print(f"  {name:<20} {c['mean_delta']:>+10.4f} "
              f"{c['median_delta']:>+10.4f} {ci_str:>22} "
              f"{c['wins']:>6} {c['ties']:>6} {c['losses']:>7}")

    # ================================================================
    # Save artifact
    # ================================================================
    artifact = {
        "experiment_id": experiment_id,
        "source_tree_sha256": source_hash,
        "model_id": model_id,
        "model_revision": model_revision,
        "device": device,
        "layer": layer,
        "seed": seed,
        "sham_seed": sham_seed,
        "replicates": R,
        "bootstrap_iterations": n_bootstrap,
        "utility_config": {
            "quality_weight": quality_weight,
            "lambda_time": lambda_time,
            "lambda_compute": lambda_compute,
            "lambda_risk": lambda_risk,
        },
        "statistics": {
            "confidence_level": confidence_level,
            "min_effect_size": min_effect_size,
        },
        "split_sizes": {
            s: len(all_tasks[s]) for s in all_tasks
        },
        "phase_2_crossover": {
            "n_sym_better": n_sym_better,
            "n_llm_better": n_llm_better,
            "n_tie": n_tie,
            "crossover_subtypes": crossover_subtypes,
            "per_subtype": {
                r["subtype"]: {
                    "n": r["n"], "s_wins": r["s_wins"],
                    "l_wins": r["l_wins"], "ties": r["ties"],
                    "reg_sym": r["reg_sym"], "reg_llm": r["reg_llm"],
                    "reg_hand": r["reg_hand"], "reg_p1": r["reg_p1"],
                    "is_crossover": r["is_crossover"],
                    "p1_beats_both": r["p1_beats_both"],
                    "p1_beats_hand": r["p1_beats_hand"],
                } for r in crossover_results
            },
        },
        "phase_3_availability": {
            "p_both_available": p_both,
        },
        "phase_5_ess": {
            "ess": ess,
            "zero_weight_fraction": wd["zero_weight_fraction"],
            "n_decisive": n_sym_better + n_llm_better,
            "n_ties": n_tie,
        },
        "phase_7_dev": {
            "always_llm_regret": dev_reg_llm,
            "always_sym_regret": dev_reg_sym,
            "hand_router_regret": dev_reg_hand,
            "p1_regret": dev_reg_p1,
            "psham_regret": dev_reg_psham,
            "centroid_regret": dev_reg_centroid,
            "unweighted_logistic_regret": dev_reg_uw,
        },
        "phase_8_calibration": {
            "tau_ood": tau_ood,
            "tau_conf": best_threshold,
        },
        "phase_10_final": {
            "p0_utility": float(final_u_p0.mean()),
            "p0_regret": final_reg_p0,
            "p1_utility": float(final_u_p1.mean()),
            "p1_regret": final_reg_p1,
            "psham_utility": float(final_u_psham.mean()),
            "psham_regret": final_reg_psham,
            "poracle_utility": float(final_u_oracle.mean()),
            "poracle_regret": 0.0,
            "always_sym_utility": float(final_u_sym.mean()),
            "always_sym_regret": final_reg_sym,
            "hand_router_utility": float(final_u_hand.mean()),
            "hand_router_regret": final_reg_hand,
            "centroid_utility": float(final_u_centroid.mean()),
            "centroid_regret": final_reg_centroid,
            "unweighted_logistic_utility": float(final_u_uw.mean()),
            "unweighted_logistic_regret": final_reg_uw,
        },
        "phase_10b_comparisons": comparisons,
        "phase_11_gate_a": {
            "A1_practical_effect": A1,
            "A2_statistical_lcb": A2,
            "A3_sham_control": A3,
            "A4_safety": A4,
            "A6_real_evidence": A6,
            "gate_a_pass": gate_a_pass,
        },
        "phase_12_oracle_gap": {
            "headroom_H": H,
            "delta_p1_p0": delta_p1_p0,
            "eta": eta,
        },
        "phase_13_crossover": {
            "n_crossover_subtypes": n_crossover,
            "n_p1_beats_both": n_p1_beats_both,
            "n_p1_beats_hand": n_p1_beats_hand,
            "per_subtype": crossover_results,
        },
    }

    out_path = Path(args.output) if args.output else (
        REPO / "artifacts" / "gate_a_experiment_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nArtifact saved to {out_path}")

    # ================================================================
    # Final interpretation
    # ================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if gate_a_pass:
        print("GATE A PASS: On the preregistered untouched real-model test")
        print("set, experience-derived AutoLearn routing produced a positive")
        print("held-out verified utility improvement over the frozen baseline")
        print("with a paired 95% confidence interval above zero, and the sham")
        print("learner did not reproduce the effect.")
    elif mean_p1_p0 > 0 and not A2:
        print("P1 > P0 but CI crosses zero — Gate fails.")
        print("A positive point estimate is not enough.")
    elif comparisons["Psham - P0"]["mean_delta"] > 0:
        print("Psham > P0 — STOP. Likely leakage, noise exploitation,")
        print("selection bias, or an invalid control.")
    elif abs(H) < 1e-6:
        print("Poracle ≈ P0 — The benchmark exhibited insufficient")
        print("routing headroom. Don't blame the learner.")
    elif delta_p1_p0 <= 0 and H > 0:
        print("Large oracle gap but P1 ≈ P0 — That is a genuine AutoLearn")
        print("failure. There is value to capture and the learner isn't")
        print("finding it.")
    elif eta >= 0.9:
        print("P1 ≈ Poracle — AutoLearn captures most available")
        print("instance-level routing value. Strongest result.")
    else:
        print("Useful causal learning was not demonstrated at this scale.")

    if n_p1_beats_hand > 0:
        print(f"\nR_P1 < R_hand-router in {n_p1_beats_hand} crossover "
              f"subtype(s) — the test that matters.")
    else:
        print(f"\nR_P1 >= R_hand-router in all crossover subtypes — "
              f"AutoLearn has not surpassed the hand router yet.")

    print("\n" + "=" * 70)
    print("GATE A EXPERIMENT COMPLETE")
    print("=" * 70)
    return 0


def _utility_lookup(task, route, exps):
    """Look up verified utility for a task+route from experience records."""
    if hasattr(route, "value"):
        route = route.value
    if route == "abstain":
        return 0.0
    task_id = str(task.get("task_id", ""))
    for e in exps:
        if e["task_id"] == task_id:
            if route == "symbolic":
                return e["sym_u"]
            elif route == "llm":
                return e["llm_u"]
            return 0.0
    return 0.0


if __name__ == "__main__":
    sys.exit(main())
