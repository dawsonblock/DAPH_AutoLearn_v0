"""v0.3.10.2-alpha — real Qwen experiment (Section 51, Phase A-F).

This is the first scientifically meaningful real-model experiment.
It runs the complete real loop:

    Phase A: train data collection (capture h, execute both backends,
             verify, compute utility, build experiences)
    Phase B: fit policies (centroid, weighted centroid, logistic, MLP)
    Phase C: dev selection (select best policy by dev regret)
    Phase D: calibration (tune confidence/OOD thresholds)
    Phase E: steering dev study (dose-response +v/0/-v)
    Phase F: final evaluation (frozen pipeline, one pass)

The experiment uses Qwen2.5-0.5B-Instruct on integer arithmetic tasks
(Section 50 — exact verification, reliable symbolic executor, low
ambiguity, clear counterfactual backend utility).

Usage:
    python scripts/run_real_qwen_experiment.py
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
    from daph_learning.execution.real_backends import (
        CaptureConfig, LLMGenerationConfig,
        build_real_counterfactual_experience,
        capture_task_representation,
        execute_llm_backend,
        execute_symbolic_backend,
        make_arithmetic_tasks,
        verify_output,
    )
    from daph_learning.policy import (
        ExperimentConfig, fit_policy, predict_proba,
        action_confidence_ece, preference_brier_soft,
    )
    from daph_learning.policy.abstention import choose_route, choose_route_with_reason
    from daph_learning.policy.regret import mean_regret, paired_promotion_statistics
    from daph_learning.policy.ood import MahalanobisOOD
    from daph_learning.policy.types import FeatureRecord, Route
    from daph_learning.interventions.real_pipeline import (
        InterventionConfig, run_real_intervention,
    )

    print("=" * 70)
    print("v0.3.10.2-alpha REAL QWEN EXPERIMENT (Section 51, Phase A-F)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Device: {device}")
    print(f"Model: {model_id}")

    # --- Load model ---
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Loaded in {time.time()-t0:.1f}s ({n_layers} layers, dim {hidden_dim})")

    # --- Config ---
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
        random_seed=42,
        max_weight=2.0,
    )
    cap_cfg = CaptureConfig(layer=layer, location="last_token")
    gen_cfg = LLMGenerationConfig(max_new_tokens=16, do_sample=False)

    # --- Generate splits (smoke size for MPS speed) ---
    print("\n--- Split sizes (smoke) ---")
    train_tasks = make_arithmetic_tasks(n=50, seed=0, max_value=50)
    dev_tasks = make_arithmetic_tasks(n=25, seed=100, max_value=50)
    cal_tasks = make_arithmetic_tasks(n=25, seed=200, max_value=50)
    final_tasks = make_arithmetic_tasks(n=50, seed=300, max_value=50)
    print(f"  train={len(train_tasks)} dev={len(dev_tasks)} cal={len(cal_tasks)} final={len(final_tasks)}")

    # ================================================================
    # Phase A: train data collection
    # ================================================================
    print("\n--- Phase A: train data collection ---")
    def collect_data(tasks, label):
        """Execute both backends on each task and build experiences."""
        experiences = []
        records = []
        captures = []
        n_sym = 0
        n_llm = 0
        t0 = time.time()
        for i, task in enumerate(tasks):
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{len(tasks)}...")
            h = capture_task_representation(
                task, model, tok, config=cap_cfg, device=device)
            sym = execute_symbolic_backend(task)
            llm_out, llm_text = execute_llm_backend(
                task, model, tok, generation_config=gen_cfg, device=device)
            exp, _, llm_v = build_real_counterfactual_experience(
                task, h, sym, llm_out, llm_text, config=cfg)
            if sym.correct:
                n_sym += 1
            if llm_v.verified_correct is True:
                n_llm += 1
            experiences.append(exp)
            records.append(FeatureRecord(exp.task_id, h))
            captures.append(h)
        elapsed = time.time() - t0
        acts = np.array(captures, dtype=np.float32)
        print(f"  [{label}] done in {elapsed:.1f}s: sym={n_sym}/{len(tasks)} llm={n_llm}/{len(tasks)}")
        return experiences, records, acts

    train_exp, train_recs, train_acts = collect_data(train_tasks, "train")
    dev_exp, dev_recs, dev_acts = collect_data(dev_tasks, "dev")
    cal_exp, cal_recs, cal_acts = collect_data(cal_tasks, "cal")
    final_exp, final_recs, final_acts = collect_data(final_tasks, "final")

    # Utility helper: look up utility from pre-computed experiences.
    exp_by_id = {}
    for exp in train_exp + dev_exp + cal_exp + final_exp:
        exp_by_id[exp.task_id] = exp

    def utility_fn(task, route):
        if isinstance(route, Route):
            route = route.value
        if route == "abstain":
            return 0.0
        exp = exp_by_id.get(str(task.get("task_id", "")))
        if exp is None:
            return 0.0
        if route == "symbolic":
            return cfg.quality_weight * exp.symbolic.quality
        if route == "llm":
            return cfg.quality_weight * exp.llm.quality
        return 0.0

    def oracle_utility(task):
        exp = exp_by_id.get(str(task.get("task_id", "")))
        if exp is None:
            return 0.0
        return max(cfg.quality_weight * exp.symbolic.quality,
                   cfg.quality_weight * exp.llm.quality)

    # ================================================================
    # Phase B: fit policies
    # ================================================================
    print("\n--- Phase B: fit policies ---")
    train_du = np.array([e.delta_utility for e in train_exp], dtype=np.float32)
    train_w = np.array([e.sample_weight for e in train_exp], dtype=np.float32)
    dev_du = np.array([e.delta_utility for e in dev_exp], dtype=np.float32)

    policies = {}
    for ptype in ("centroid", "logistic", "mlp_experimental"):
        pcfg = ExperimentConfig(
            policy_type=ptype,
            target_mode="soft",
            target_temperature=0.5,
            weight_mode="clipped_gap",
            gap_threshold=0.01,
            confidence_threshold=0.5,
            selected_layer=layer,
            random_seed=42,
            max_weight=2.0,
        )
        t0 = time.time()
        model_p = fit_policy(
            pcfg, train_acts, train_du, train_w,
            dev_features=dev_acts, dev_delta_u=dev_du,
            dev_weights=np.ones(len(dev_exp), dtype=np.float32),
            dev_tasks=dev_tasks, utility_fn=utility_fn, seed=42)
        elapsed = time.time() - t0
        probs = predict_proba(model_p, dev_acts)
        routes = [choose_route(float(p), 0.5).value for p in probs]
        pred_u = np.array([utility_fn(t, r) for t, r in zip(dev_tasks, routes)], dtype=np.float64)
        ora_u = np.array([oracle_utility(t) for t in dev_tasks], dtype=np.float64)
        dev_reg = float(mean_regret(pred_u, ora_u))
        dev_util = float(pred_u.mean())
        print(f"  {ptype:20s}: dev_regret={dev_reg:.4f} dev_utility={dev_util:.4f} ({elapsed:.1f}s)")
        policies[ptype] = {
            "model": model_p,
            "cfg": pcfg,
            "dev_regret": dev_reg,
            "dev_utility": dev_util,
        }

    # Also fit unweighted logistic for the weighting test.
    pcfg_uw = ExperimentConfig(
        policy_type="logistic", target_mode="soft", target_temperature=0.5,
        weight_mode="uniform", gap_threshold=0.01, confidence_threshold=0.5,
        selected_layer=layer, random_seed=42, max_weight=2.0)
    model_uw = fit_policy(
        pcfg_uw, train_acts, train_du, train_w,
        dev_features=dev_acts, dev_delta_u=dev_du,
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=dev_tasks, utility_fn=utility_fn, seed=42)
    probs_uw = predict_proba(model_uw, dev_acts)
    routes_uw = [choose_route(float(p), 0.5).value for p in probs_uw]
    pred_u_uw = np.array([utility_fn(t, r) for t, r in zip(dev_tasks, routes_uw)], dtype=np.float64)
    ora_u = np.array([oracle_utility(t) for t in dev_tasks], dtype=np.float64)
    dev_reg_uw = float(mean_regret(pred_u_uw, ora_u))
    print(f"  {'logistic_unweighted':20s}: dev_regret={dev_reg_uw:.4f}")

    # ================================================================
    # Phase C: dev selection
    # ================================================================
    print("\n--- Phase C: dev selection ---")
    best_ptype = min(policies, key=lambda k: policies[k]["dev_regret"])
    best_policy = policies[best_ptype]
    print(f"  Best policy: {best_ptype} (dev_regret={best_policy['dev_regret']:.4f})")

    # Always-LLM baseline.
    routes_llm = ["llm"] * len(dev_tasks)
    pred_u_llm = np.array([utility_fn(t, "llm") for t in dev_tasks], dtype=np.float64)
    reg_llm = float(mean_regret(pred_u_llm, ora_u))
    print(f"  Always-LLM:   dev_regret={reg_llm:.4f}")
    # Always-symbolic baseline.
    routes_sym = ["symbolic"] * len(dev_tasks)
    pred_u_sym = np.array([utility_fn(t, "symbolic") for t in dev_tasks], dtype=np.float64)
    reg_sym = float(mean_regret(pred_u_sym, ora_u))
    print(f"  Always-sym:   dev_regret={reg_sym:.4f}")

    # ================================================================
    # Phase D: calibration
    # ================================================================
    print("\n--- Phase D: calibration ---")
    # Fit OOD on train features.
    ood = MahalanobisOOD(ridge=cfg.ood_ridge)
    ood.fit(train_acts)
    cal_scores = np.array([ood.score(h) for h in cal_acts])
    tau_ood = float(np.quantile(cal_scores, cfg.ood_quantile))
    print(f"  tau_ood = {tau_ood:.4f} (q={cfg.ood_quantile})")

    # Grid search confidence threshold using actual policy probs.
    cal_probs = predict_proba(best_policy["model"], cal_acts)
    best_threshold = 0.5
    best_cal_util = -1.0
    for threshold in np.arange(0.5, 0.95, 0.05):
        utils = []
        for i, task in enumerate(cal_tasks):
            p = float(cal_probs[i])
            decision = choose_route_with_reason(
                p, float(threshold), ood_score=float(cal_scores[i]),
                ood_threshold=tau_ood)
            u = utility_fn(task, decision.route)
            utils.append(u)
        mean_u = float(np.mean(utils))
        if mean_u > best_cal_util:
            best_cal_util = mean_u
            best_threshold = float(threshold)
    print(f"  tau_conf = {best_threshold:.4f} (cal_utility={best_cal_util:.4f})")

    # Calibration metrics.
    soft_targets = 1.0 / (1.0 + np.exp(-cal_probs / 0.5))
    brier = float(preference_brier_soft(cal_probs, soft_targets))
    ece = float(action_confidence_ece(cal_probs, soft_targets))
    print(f"  preference_brier_soft = {brier:.4f}")
    print(f"  action_confidence_ece = {ece:.4f}")

    # ================================================================
    # Phase F: final evaluation (one pass, frozen pipeline)
    # ================================================================
    print("\n--- Phase F: final evaluation ---")
    final_probs = predict_proba(best_policy["model"], final_acts)
    final_ora_u = np.array([oracle_utility(t) for t in final_tasks], dtype=np.float64)

    # Candidate policy (AutoLearn).
    final_routes = []
    final_reasons = []
    for i, task in enumerate(final_tasks):
        p = float(final_probs[i])
        ood_score = ood.score(final_acts[i])
        decision = choose_route_with_reason(
            p, best_threshold, ood_score=ood_score, ood_threshold=tau_ood)
        final_routes.append(decision.route.value)
        final_reasons.append(decision.reason)
    final_cand_u = np.array(
        [utility_fn(t, r) for t, r in zip(final_tasks, final_routes)],
        dtype=np.float64)
    final_cand_reg = float(mean_regret(final_cand_u, final_ora_u))

    # Incumbent: always-LLM.
    final_inc_u = np.array(
        [utility_fn(t, "llm") for t in final_tasks], dtype=np.float64)
    final_inc_reg = float(mean_regret(final_inc_u, final_ora_u))

    # Always-symbolic.
    final_sym_u = np.array(
        [utility_fn(t, "symbolic") for t in final_tasks], dtype=np.float64)
    final_sym_reg = float(mean_regret(final_sym_u, final_ora_u))

    # Routing accuracy.
    final_du = np.array([e.delta_utility for e in final_exp], dtype=np.float32)
    final_acc = float(np.mean([
        (final_routes[i] == "symbolic" and final_du[i] > 0)
        or (final_routes[i] == "llm" and final_du[i] < 0)
        for i in range(len(final_tasks))]))

    # Paired statistics.
    stats = paired_promotion_statistics(
        final_cand_u, final_inc_u, final_ora_u, seed=42)

    print(f"\n  FINAL RESULTS (evidence_level=REAL_MODEL_FINAL):")
    print(f"  Method             | Utility ↑ | Regret ↓ | Accuracy ↑")
    print(f"  -------------------|-----------|----------|----------")
    print(f"  Always-LLM         | {final_inc_u.mean():.4f}    | {final_inc_reg:.4f}   | N/A")
    print(f"  Always-symbolic    | {final_sym_u.mean():.4f}    | {final_sym_reg:.4f}   | N/A")
    print(f"  Candidate AutoLearn| {final_cand_u.mean():.4f}    | {final_cand_reg:.4f}   | {final_acc:.4f}")
    print(f"\n  mean_utility_delta (cand - inc): {stats['mean_utility_delta']:.4f}")
    print(f"  mean_regret_delta  (inc - cand): {stats.get('mean_regret_delta', 0):.4f}")
    print(f"  win_rate: {stats['candidate_win_rate']:.2f}  loss_rate: {stats['incumbent_win_rate']:.2f}")

    # ================================================================
    # Phase E: steering dev study (dose-response)
    # ================================================================
    print("\n--- Phase E: steering dev study ---")
    # Get the centroid vector for steering.
    from daph_learning.policy.centroid_policy import CentroidPolicy
    centroid_cfg = ExperimentConfig(
        policy_type="centroid", gap_threshold=0.01,
        confidence_threshold=0.5, random_seed=42, max_weight=2.0)
    centroid_model = fit_policy(
        centroid_cfg, train_acts, train_du, train_w)
    v = centroid_model.vector
    v_norm = float(np.linalg.norm(v))
    print(f"  Centroid vector norm: {v_norm:.4f}")

    # Steering: run dose-response on dev tasks.
    def policy_prob_fn(h):
        h = np.asarray(h, dtype=np.float64).flatten()
        if centroid_model.vector.shape[0] != h.shape[0]:
            return 0.5
        score = (h @ centroid_model.vector - centroid_model.threshold) / max(centroid_model.temperature, 1e-8)
        return float(1.0 / (1.0 + np.exp(-score)))

    steering_tasks = [
        {"task_id": f"steer_{i}", "prompt": dev_tasks[i]["prompt"],
         "family": dev_tasks[i]["family"], "expected": dev_tasks[i]["expected"]}
        for i in range(min(10, len(dev_tasks)))
    ]

    int_cfg = InterventionConfig(
        layer=layer, alpha_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
        max_vector_norm=1.0, clamp_norm=5.0)
    intervention_results = run_real_intervention(
        model, tok, steering_tasks, v,
        config=int_cfg, policy_prob_fn=policy_prob_fn,
        utility_fn=lambda task, route: 1.0 if route == Route.SYMBOLIC else 0.0,
        device=device)

    # Aggregate +v / 0 / -v.
    plus = [r for r in intervention_results if r.alpha == 1.0]
    zero = [r for r in intervention_results if r.alpha == 0.0]
    minus = [r for r in intervention_results if r.alpha == -1.0]
    if plus and zero and minus:
        mean_p_plus = float(np.mean([r.route_score_after for r in plus]))
        mean_p_zero = float(np.mean([r.route_score_after for r in zero]))
        mean_p_minus = float(np.mean([r.route_score_after for r in minus]))
        print(f"  E[P(S|+v)] = {mean_p_plus:.4f}")
        print(f"  E[P(S| 0)] = {mean_p_zero:.4f}")
        print(f"  E[P(S|-v)] = {mean_p_minus:.4f}")
        plus_gt = sum(1 for r in plus if r.route_score_after > r.route_score_before)
        minus_lt = sum(1 for r in minus if r.route_score_after < r.route_score_before)
        print(f"  +v > 0: {plus_gt}/{len(plus)}, -v < 0: {minus_lt}/{len(minus)}")
        print(f"  evidence_level: REAL_MODEL_LATENT_INTERVENTION")

    # ================================================================
    # Save experiment artifact
    # ================================================================
    artifact = {
        "release": "0.3.10.2-alpha",
        "evidence_level": "REAL_MODEL_FINAL",
        "model_id": model_id,
        "device": device,
        "layer": layer,
        "timestamp": time.time(),
        "splits": {
            "train": len(train_tasks),
            "dev": len(dev_tasks),
            "calibration": len(cal_tasks),
            "final": len(final_tasks),
        },
        "phase_b_results": {
            ptype: {
                "dev_regret": policies[ptype]["dev_regret"],
                "dev_utility": policies[ptype]["dev_utility"],
            }
            for ptype in policies
        },
        "phase_b_unweighted": {"dev_regret": dev_reg_uw},
        "phase_c_selection": {
            "best_policy": best_ptype,
            "always_llm_dev_regret": reg_llm,
            "always_sym_dev_regret": reg_sym,
        },
        "phase_d_calibration": {
            "tau_ood": tau_ood,
            "tau_conf": best_threshold,
            "preference_brier_soft": brier,
            "action_confidence_ece": ece,
        },
        "phase_f_final": {
            "candidate_utility": float(final_cand_u.mean()),
            "candidate_regret": final_cand_reg,
            "incumbent_utility": float(final_inc_u.mean()),
            "incumbent_regret": final_inc_reg,
            "always_sym_utility": float(final_sym_u.mean()),
            "always_sym_regret": final_sym_reg,
            "routing_accuracy": final_acc,
            "abstain_rate": float(np.mean([r == "abstain" for r in final_routes])),
            "symbolic_utilization": float(np.mean([r == "symbolic" for r in final_routes])),
            "llm_utilization": float(np.mean([r == "llm" for r in final_routes])),
            "paired_stats": stats,
        },
        "phase_e_steering": {
            "n_intervention_results": len(intervention_results),
            "mean_p_plus": mean_p_plus if plus else None,
            "mean_p_zero": mean_p_zero if zero else None,
            "mean_p_minus": mean_p_minus if minus else None,
            "plus_gt_zero": plus_gt if plus else None,
            "minus_lt_zero": minus_lt if minus else None,
            "evidence_level": "REAL_MODEL_LATENT_INTERVENTION",
        },
    }

    out_path = REPO / "artifacts" / "real_qwen_experiment_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nExperiment artifact saved to {out_path}")

    # --- Scientific success tiers (Section 48) ---
    print("\n--- Scientific success tiers (Section 48) ---")
    tier1 = final_cand_reg < reg_llm  # Tier 1: beats always-LLM
    tier2 = final_cand_reg < final_inc_reg  # Tier 2: counterfactual value
    print(f"  Tier 1 (beats always-LLM):     {'YES' if tier1 else 'NO'} (cand_reg={final_cand_reg:.4f} < llm_reg={reg_llm:.4f})")
    print(f"  Tier 2 (counterfactual value):  {'YES' if tier2 else 'NO'} (cand_reg={final_cand_reg:.4f} < inc_reg={final_inc_reg:.4f})")

    print("\n" + "=" * 70)
    print("REAL QWEN EXPERIMENT COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
