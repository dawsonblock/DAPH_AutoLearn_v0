"""v0.3.10.3.2-alpha — real Qwen experiment (Section 51, Phase A-F).

v0.3.10.3 — CRITICAL FIXES from external review:

1. Symbolic output is now verified against expected answer (not just
   "execution succeeded = correct").
2. Weighted-vs-unweighted ablation actually uses w=1 for unweighted.
3. Calibration targets use ΔU, not the model's own predictions.
4. Final set is NOT executed until after all configuration is frozen
   (Phase F runs AFTER Phase E, not before).
5. Word pools are partitioned across splits — no prompt leakage.
6. Steering utility uses verified backend utility, not synthetic
   route=SYMBOLIC → 1.0.
7. Routing accuracy is tie-aware (ties are correct for either route).
8. Confidence semantics fixed: unsupported backend has confidence=1.0.
9. max_vector_norm only shrinks, never expands.

Usage:
    python scripts/run_real_qwen_experiment.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def _source_tree_sha256() -> str:
    """Canonical source-tree hash (Section 2).

    Delegates to :func:`daph_learning.provenance.compute_source_tree_sha256`
    so that all components record the same hash.
    """
    from daph_learning.provenance import compute_source_tree_sha256
    return compute_source_tree_sha256(REPO)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from daph_learning.execution.real_backends import (
        CaptureConfig, LLMGenerationConfig,
        assert_splits_disjoint,
        build_real_counterfactual_experience,
        capture_task_representation,
        execute_llm_backend,
        execute_symbolic_backend,
        make_arithmetic_tasks,
        make_letter_counting_tasks,
        partition_word_pool,
    )
    from daph_learning.policy import (
        ExperimentConfig, fit_policy, predict_proba,
        action_confidence_ece, preference_brier_soft,
    )
    from daph_learning.policy.abstention import choose_route_with_reason
    from daph_learning.policy.regret import mean_regret, paired_promotion_statistics
    from daph_learning.policy.ood import MahalanobisOOD
    from daph_learning.policy.types import FeatureRecord, Route
    from daph_learning.interventions.real_pipeline import (
        InterventionConfig, run_real_intervention,
    )

    print("=" * 70)
    print("v0.3.10.3.2-alpha REAL QWEN EXPERIMENT (Section 51, Phase A-F)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Device: {device}")
    print(f"Model: {model_id}")

    source_hash = _source_tree_sha256()
    print(f"Source tree SHA256: {source_hash}")

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

    # --- Generate splits with DISJOINT word pools (P0-7) ---
    print("\n--- Split sizes (smoke, mixed tasks, DISJOINT word pools) ---")
    word_splits = partition_word_pool(n_splits=4, seed=42)
    train_words, dev_words, cal_words, final_words = word_splits
    print(f"  Word pool sizes: train={len(train_words)} dev={len(dev_words)} "
          f"cal={len(cal_words)} final={len(final_words)}")

    # Arithmetic uses different operands per seed (no overlap concern).
    # Counting uses disjoint word pools.
    train_tasks = (
        make_arithmetic_tasks(n=30, seed=0, max_value=100)
        + make_letter_counting_tasks(n=30, seed=1000, word_pool=train_words)
    )
    dev_tasks = (
        make_arithmetic_tasks(n=15, seed=100, max_value=100)
        + make_letter_counting_tasks(n=15, seed=1100, word_pool=dev_words)
    )
    cal_tasks = (
        make_arithmetic_tasks(n=15, seed=200, max_value=100)
        + make_letter_counting_tasks(n=15, seed=1200, word_pool=cal_words)
    )
    final_tasks = (
        make_arithmetic_tasks(n=30, seed=300, max_value=100)
        + make_letter_counting_tasks(n=30, seed=1300, word_pool=final_words)
    )

    # Shuffle each split.
    for tasks, s in [(train_tasks, 0), (dev_tasks, 1), (cal_tasks, 2), (final_tasks, 3)]:
        rng = np.random.default_rng(s + 9999)
        rng.shuffle(tasks)

    # VERIFY split disjointness for counting tasks.
    print(f"  train={len(train_tasks)} dev={len(dev_tasks)} cal={len(cal_tasks)} final={len(final_tasks)}")
    try:
        assert_splits_disjoint([train_tasks, dev_tasks, cal_tasks, final_tasks])
        print("  Split disjointness: VERIFIED (no prompt leakage)")
    except AssertionError as e:
        print(f"  Split disjointness: FAILED — {e}")
        return 1

    n_arith = sum(1 for t in train_tasks if "integer_arithmetic" in t.get("capability_ids", []))
    n_count = sum(1 for t in train_tasks if "letter_counting" in t.get("capability_ids", []))
    print(f"  train composition: {n_arith} arithmetic, {n_count} counting")

    # ================================================================
    # Phase A: train + dev + cal data collection
    # (NOT final — final is deferred until after freeze)
    # ================================================================
    print("\n--- Phase A: train/dev/cal data collection ---")

    def collect_data(tasks, label):
        """Execute both backends on each task and build experiences."""
        experiences = []
        records = []
        captures = []
        n_sym_correct = 0
        n_llm_correct = 0
        t0 = time.time()
        for i, task in enumerate(tasks):
            if (i + 1) % 10 == 0:
                print(f"  [{label}] {i+1}/{len(tasks)}...")
            h = capture_task_representation(
                task, model, tok, config=cap_cfg, device=device)
            sym_outcome, sym_text = execute_symbolic_backend(task)
            llm_out, llm_text = execute_llm_backend(
                task, model, tok, generation_config=gen_cfg, device=device)
            exp, sym_v, llm_v = build_real_counterfactual_experience(
                task, h, sym_outcome, llm_out, llm_text,
                config=cfg, symbolic_output_text=sym_text)
            if exp.symbolic.correct:
                n_sym_correct += 1
            if exp.llm.correct:
                n_llm_correct += 1
            experiences.append(exp)
            records.append(FeatureRecord(exp.task_id, h))
            captures.append(h)
        elapsed = time.time() - t0
        acts = np.array(captures, dtype=np.float32)
        print(f"  [{label}] done in {elapsed:.1f}s: "
              f"sym_correct={n_sym_correct}/{len(tasks)} "
              f"llm_correct={n_llm_correct}/{len(tasks)}")
        return experiences, records, acts

    train_exp, train_recs, train_acts = collect_data(train_tasks, "train")
    dev_exp, dev_recs, dev_acts = collect_data(dev_tasks, "dev")
    cal_exp, cal_recs, cal_acts = collect_data(cal_tasks, "cal")

    # Utility helpers.
    exp_by_id = {}
    for exp in train_exp + dev_exp + cal_exp:
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

    # Report Phase A routing signal.
    train_du = np.array([e.delta_utility for e in train_exp], dtype=np.float32)
    n_sym_better = int(np.sum(train_du > 0.01))
    n_llm_better = int(np.sum(train_du < -0.01))
    n_tie = int(np.sum(np.abs(train_du) <= 0.01))
    print(f"\n  Phase A routing signal: sym_better={n_sym_better} llm_better={n_llm_better} tie={n_tie}")
    print(f"  ΔU range: [{train_du.min():.3f}, {train_du.max():.3f}] mean={train_du.mean():.3f}")

    # ================================================================
    # Phase B: fit policies
    # ================================================================
    print("\n--- Phase B: fit policies ---")
    train_du = np.array([e.delta_utility for e in train_exp], dtype=np.float32)
    train_w = np.array([e.sample_weight for e in train_exp], dtype=np.float32)
    dev_du = np.array([e.delta_utility for e in dev_exp], dtype=np.float32)
    dev_ora_u = np.array([oracle_utility(t) for t in dev_tasks], dtype=np.float64)

    def eval_policy(model_p, tasks, acts, threshold=0.5, ood=None, tau_ood=float("inf")):
        probs = predict_proba(model_p, acts)
        routes = []
        for i, task in enumerate(tasks):
            p = float(probs[i])
            ood_score = ood.score(acts[i]) if ood else None
            decision = choose_route_with_reason(
                p, threshold, ood_score=ood_score, ood_threshold=tau_ood)
            routes.append(decision.route.value)
        utils = np.array([utility_fn(t, r) for t, r in zip(tasks, routes)], dtype=np.float64)
        ora = np.array([oracle_utility(t) for t in tasks], dtype=np.float64)
        reg = float(mean_regret(utils, ora))
        return routes, utils, reg

    # Fit all policy types.
    policies = {}
    for ptype in ("centroid", "logistic", "mlp_experimental"):
        pcfg = ExperimentConfig(
            policy_type=ptype, target_mode="soft", target_temperature=0.5,
            weight_mode="clipped_gap", gap_threshold=0.01,
            confidence_threshold=0.5, selected_layer=layer,
            random_seed=42, max_weight=2.0)
        t0 = time.time()
        model_p = fit_policy(
            pcfg, train_acts, train_du, train_w,
            dev_features=dev_acts, dev_delta_u=dev_du,
            dev_weights=np.ones(len(dev_exp), dtype=np.float32),
            dev_tasks=dev_tasks, utility_fn=utility_fn, seed=42)
        elapsed = time.time() - t0
        _, _, dev_reg = eval_policy(model_p, dev_tasks, dev_acts)
        print(f"  {ptype:20s}: dev_regret={dev_reg:.4f} ({elapsed:.1f}s)")
        policies[ptype] = {"model": model_p, "cfg": pcfg, "dev_regret": dev_reg}

    # P0-5 — TRUE unweighted ablation: use w=1, not the weighted train_w.
    train_w_uniform = np.ones(len(train_exp), dtype=np.float32)
    pcfg_uw = ExperimentConfig(
        policy_type="logistic", target_mode="soft", target_temperature=0.5,
        weight_mode="uniform", gap_threshold=0.01, confidence_threshold=0.5,
        selected_layer=layer, random_seed=42, max_weight=2.0)
    model_uw = fit_policy(
        pcfg_uw, train_acts, train_du, train_w_uniform,
        dev_features=dev_acts, dev_delta_u=dev_du,
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=dev_tasks, utility_fn=utility_fn, seed=42)
    _, _, dev_reg_uw = eval_policy(model_uw, dev_tasks, dev_acts)
    print(f"  {'logistic_unweighted':20s}: dev_regret={dev_reg_uw:.4f} (TRUE w=1)")

    # Also unweighted centroid.
    pcfg_uwc = ExperimentConfig(
        policy_type="centroid", target_mode="soft", target_temperature=0.5,
        weight_mode="uniform", gap_threshold=0.01, confidence_threshold=0.5,
        selected_layer=layer, random_seed=42, max_weight=2.0)
    model_uwc = fit_policy(
        pcfg_uwc, train_acts, train_du, train_w_uniform,
        dev_features=dev_acts, dev_delta_u=dev_du,
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=dev_tasks, utility_fn=utility_fn, seed=42)
    _, _, dev_reg_uwc = eval_policy(model_uwc, dev_tasks, dev_acts)
    print(f"  {'centroid_unweighted':20s}: dev_regret={dev_reg_uwc:.4f} (TRUE w=1)")

    # Hard-target logistic.
    pcfg_hard = ExperimentConfig(
        policy_type="logistic", target_mode="hard", target_temperature=0.5,
        weight_mode="clipped_gap", gap_threshold=0.01, confidence_threshold=0.5,
        selected_layer=layer, random_seed=42, max_weight=2.0)
    model_hard = fit_policy(
        pcfg_hard, train_acts, train_du, train_w,
        dev_features=dev_acts, dev_delta_u=dev_du,
        dev_weights=np.ones(len(dev_exp), dtype=np.float32),
        dev_tasks=dev_tasks, utility_fn=utility_fn, seed=42)
    _, _, dev_reg_hard = eval_policy(model_hard, dev_tasks, dev_acts)
    print(f"  {'logistic_hard':20s}: dev_regret={dev_reg_hard:.4f}")

    # ================================================================
    # Phase C: dev selection + baselines
    # ================================================================
    print("\n--- Phase C: dev selection ---")
    best_ptype = min(policies, key=lambda k: policies[k]["dev_regret"])
    best_policy = policies[best_ptype]
    print(f"  Best policy: {best_ptype} (dev_regret={best_policy['dev_regret']:.4f})")

    # Baselines on dev.
    dev_u_llm = np.array([utility_fn(t, "llm") for t in dev_tasks], dtype=np.float64)
    dev_reg_llm = float(mean_regret(dev_u_llm, dev_ora_u))
    dev_u_sym = np.array([utility_fn(t, "symbolic") for t in dev_tasks], dtype=np.float64)
    dev_reg_sym = float(mean_regret(dev_u_sym, dev_ora_u))

    # Hand-coded router: route by family.
    dev_routes_hand = []
    for t in dev_tasks:
        caps = t.get("capability_ids", [])
        if "integer_arithmetic" in caps:
            dev_routes_hand.append("symbolic")
        elif "letter_counting" in caps:
            dev_routes_hand.append("llm")
        else:
            dev_routes_hand.append("llm")
    dev_u_hand = np.array([utility_fn(t, r) for t, r in zip(dev_tasks, dev_routes_hand)], dtype=np.float64)
    dev_reg_hand = float(mean_regret(dev_u_hand, dev_ora_u))

    print(f"  Always-LLM:           dev_regret={dev_reg_llm:.4f}")
    print(f"  Always-symbolic:      dev_regret={dev_reg_sym:.4f}")
    print(f"  Hand router (family): dev_regret={dev_reg_hand:.4f}")
    print(f"  Unweighted logistic:  dev_regret={dev_reg_uw:.4f}")
    print(f"  Unweighted centroid:  dev_regret={dev_reg_uwc:.4f}")
    print(f"  Hard-target logistic: dev_regret={dev_reg_hard:.4f}")

    # ================================================================
    # Phase D: calibration
    # ================================================================
    print("\n--- Phase D: calibration ---")
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

    # P0-6 — Calibration metrics use ΔU for soft targets, NOT cal_probs.
    cal_du = np.array([e.delta_utility for e in cal_exp], dtype=np.float64)
    soft_targets = 1.0 / (1.0 + np.exp(-cal_du / cfg.target_temperature))
    brier = float(preference_brier_soft(cal_probs, soft_targets))
    ece = float(action_confidence_ece(cal_probs, soft_targets))
    print(f"  preference_brier_soft = {brier:.4f} (targets from ΔU)")
    print(f"  action_confidence_ece = {ece:.4f} (targets from ΔU)")

    # ================================================================
    # Phase E: steering dev study (BEFORE final — P0-8)
    # ================================================================
    print("\n--- Phase E: steering dev study ---")
    centroid_cfg = ExperimentConfig(
        policy_type="centroid", gap_threshold=0.01,
        confidence_threshold=0.5, random_seed=42, max_weight=2.0)
    centroid_model = fit_policy(
        centroid_cfg, train_acts, train_du, train_w)
    v = centroid_model.vector
    v_norm = float(np.linalg.norm(v))
    print(f"  Centroid vector norm: {v_norm:.4f}")
    if hasattr(centroid_model, "weight_fallback"):
        print(f"  Weight fallback: {centroid_model.weight_fallback}")

    # P0-11 — Steering utility uses VERIFIED utility, not synthetic.
    # For each dev task, the utility of a route is the verified quality
    # of the backend that would be selected.
    def steering_utility_fn(task, route):
        return utility_fn(task, route)

    def policy_prob_fn(h):
        h = np.asarray(h, dtype=np.float64).flatten()
        if centroid_model.vector.shape[0] != h.shape[0]:
            return 0.5
        score = (h @ centroid_model.vector - centroid_model.threshold) / max(centroid_model.temperature, 1e-8)
        return float(1.0 / (1.0 + np.exp(-score)))

    # Use the original dev tasks directly (with their original task_ids)
    # so utility_fn can look up the verified backend outcomes.
    steering_tasks = dev_tasks[:min(10, len(dev_tasks))]

    int_cfg = InterventionConfig(
        layer=layer, alpha_grid=(-1.0, -0.5, 0.0, 0.5, 1.0),
        max_vector_norm=1.0, clamp_norm=5.0)
    intervention_results = run_real_intervention(
        model, tok, steering_tasks, v,
        config=int_cfg, policy_prob_fn=policy_prob_fn,
        utility_fn=steering_utility_fn,
        device=device)

    # Aggregate dose-response.
    steering_summary = {}
    for alpha_val in (-1.0, -0.5, 0.0, 0.5, 1.0):
        group = [r for r in intervention_results if r.alpha == alpha_val]
        if group:
            mean_p = float(np.mean([r.route_score_after for r in group]))
            mean_u = float(np.mean([r.intervened_utility for r in group]))
            steering_summary[alpha_val] = {
                "mean_p": mean_p,
                "mean_utility": mean_u,
                "n": len(group),
            }
    if 1.0 in steering_summary and 0.0 in steering_summary and -1.0 in steering_summary:
        print(f"  E[P(S|+v)] = {steering_summary[1.0]['mean_p']:.4f}")
        print(f"  E[P(S| 0)] = {steering_summary[0.0]['mean_p']:.4f}")
        print(f"  E[P(S|-v)] = {steering_summary[-1.0]['mean_p']:.4f}")
        print(f"  E[U|+v]    = {steering_summary[1.0]['mean_utility']:.4f}")
        print(f"  E[U| 0]    = {steering_summary[0.0]['mean_utility']:.4f}")
        print(f"  E[U|-v]    = {steering_summary[-1.0]['mean_utility']:.4f}")
    print("  evidence_level: REAL_MODEL_LATENT_INTERVENTION")

    # ================================================================
    # FREEZE: all configuration is now frozen.
    # No more fitting, selection, or calibration after this point.
    # ================================================================
    print("\n--- FREEZE: all configuration frozen ---")
    print(f"  best_policy: {best_ptype}")
    print(f"  tau_conf: {best_threshold}")
    print(f"  tau_ood: {tau_ood}")
    print(f"  source_hash: {source_hash}")

    # ================================================================
    # Phase F: final evaluation (ONE pass, frozen pipeline — P0-8)
    # ================================================================
    print("\n--- Phase F: final evaluation (one pass, frozen) ---")
    final_exp, final_recs, final_acts = collect_data(final_tasks, "final")

    # Update exp_by_id with final experiences.
    for exp in final_exp:
        exp_by_id[exp.task_id] = exp

    final_ora_u = np.array([oracle_utility(t) for t in final_tasks], dtype=np.float64)

    # Candidate policy (AutoLearn) with calibration.
    final_probs = predict_proba(best_policy["model"], final_acts)
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

    # All baselines on final.
    final_u_llm = np.array([utility_fn(t, "llm") for t in final_tasks], dtype=np.float64)
    final_reg_llm = float(mean_regret(final_u_llm, final_ora_u))
    final_u_sym = np.array([utility_fn(t, "symbolic") for t in final_tasks], dtype=np.float64)
    final_reg_sym = float(mean_regret(final_u_sym, final_ora_u))

    # Hand router.
    final_routes_hand = []
    for t in final_tasks:
        caps = t.get("capability_ids", [])
        if "integer_arithmetic" in caps:
            final_routes_hand.append("symbolic")
        elif "letter_counting" in caps:
            final_routes_hand.append("llm")
        else:
            final_routes_hand.append("llm")
    final_u_hand = np.array([utility_fn(t, r) for t, r in zip(final_tasks, final_routes_hand)], dtype=np.float64)
    final_reg_hand = float(mean_regret(final_u_hand, final_ora_u))

    # Unweighted logistic and centroid (TRUE w=1).
    _, _, final_reg_uw = eval_policy(model_uw, final_tasks, final_acts, threshold=best_threshold, ood=ood, tau_ood=tau_ood)
    _, _, final_reg_uwc = eval_policy(model_uwc, final_tasks, final_acts, threshold=best_threshold, ood=ood, tau_ood=tau_ood)
    # Hard-target logistic.
    _, _, final_reg_hard = eval_policy(model_hard, final_tasks, final_acts, threshold=best_threshold, ood=ood, tau_ood=tau_ood)
    # Centroid and MLP.
    _, _, final_reg_centroid = eval_policy(policies["centroid"]["model"], final_tasks, final_acts, threshold=best_threshold, ood=ood, tau_ood=tau_ood)
    _, _, final_reg_mlp = eval_policy(policies["mlp_experimental"]["model"], final_tasks, final_acts, threshold=best_threshold, ood=ood, tau_ood=tau_ood)

    # P0-14 — Tie-aware routing accuracy.
    final_du = np.array([e.delta_utility for e in final_exp], dtype=np.float32)
    # Strict accuracy (old): only counts non-tie tasks.
    decisive_mask = np.abs(final_du) > 0.01
    if decisive_mask.sum() > 0:
        strict_acc = float(np.mean([
            (final_routes[i] == "symbolic" and final_du[i] > 0)
            or (final_routes[i] == "llm" and final_du[i] < 0)
            for i in range(len(final_tasks)) if decisive_mask[i]]))
    else:
        strict_acc = 0.0
    # Tie-aware accuracy: on ties, either route is correct.
    tie_aware_acc = float(np.mean([
        (final_routes[i] == "symbolic" and final_du[i] > 0)
        or (final_routes[i] == "llm" and final_du[i] < 0)
        or abs(final_du[i]) <= 0.01  # tie → any route is correct
        for i in range(len(final_tasks))]))

    # Per-family metrics.
    arith_mask = np.array(["integer_arithmetic" in t.get("capability_ids", []) for t in final_tasks])
    count_mask = ~arith_mask
    per_family = {}
    for name, mask in [("arithmetic", arith_mask), ("counting", count_mask)]:
        if mask.sum() > 0:
            per_family[name] = {
                "n": int(mask.sum()),
                "candidate_utility": float(final_cand_u[mask].mean()),
                "candidate_regret": float(mean_regret(final_cand_u[mask], final_ora_u[mask])),
                "llm_utility": float(final_u_llm[mask].mean()),
                "sym_utility": float(final_u_sym[mask].mean()),
            }

    # Paired statistics (candidate vs incumbent=always-LLM).
    stats = paired_promotion_statistics(
        final_cand_u, final_u_llm, final_ora_u, seed=42)

    from collections import Counter
    reason_counts = Counter(final_reasons)

    # Results table.
    print(f"\n  {'Method':<25s} {'Utility':>8s} {'Regret':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    print(f"  {'Always LLM':<25s} {final_u_llm.mean():8.4f} {final_reg_llm:8.4f}")
    print(f"  {'Always symbolic':<25s} {final_u_sym.mean():8.4f} {final_reg_sym:8.4f}")
    print(f"  {'Hand router':<25s} {final_u_hand.mean():8.4f} {final_reg_hand:8.4f}")
    print(f"  {'Unweighted centroid':<25s} {'N/A':>8s} {final_reg_uwc:8.4f}")
    print(f"  {'Weighted centroid':<25s} {'N/A':>8s} {final_reg_centroid:8.4f}")
    print(f"  {'Unweighted logistic':<25s} {'N/A':>8s} {final_reg_uw:8.4f}")
    print(f"  {'Weighted logistic':<25s} {final_cand_u.mean():8.4f} {final_cand_reg:8.4f}")
    print(f"  {'Logistic hard':<25s} {'N/A':>8s} {final_reg_hard:8.4f}")
    print(f"  {'MLP':<25s} {'N/A':>8s} {final_reg_mlp:8.4f}")
    print(f"\n  strict_routing_accuracy (decisive only): {strict_acc:.4f}")
    print(f"  tie_aware_routing_accuracy:              {tie_aware_acc:.4f}")
    print(f"  abstain_rate: {float(np.mean([r == 'abstain' for r in final_routes])):.4f}")
    print(f"  symbolic_fraction: {float(np.mean([r == 'symbolic' for r in final_routes])):.4f}")
    print(f"  llm_fraction: {float(np.mean([r == 'llm' for r in final_routes])):.4f}")
    print(f"  per_family: {json.dumps(per_family, indent=2)}")

    # ================================================================
    # Save experiment artifact
    # ================================================================
    artifact = {
        "release": "0.3.10.3.2-alpha",
        "evidence_level": "REAL_MODEL_FINAL",
        "model_id": model_id,
        "device": device,
        "layer": layer,
        "source_tree_sha256": source_hash,
        "timestamp": time.time(),
        "splits": {
            "train": len(train_tasks),
            "dev": len(dev_tasks),
            "calibration": len(cal_tasks),
            "final": len(final_tasks),
        },
        "split_disjointness": "VERIFIED — disjoint word pools per split",
        "task_composition": {
            "arithmetic": n_arith,
            "counting": n_count,
        },
        "phase_a_routing_signal": {
            "sym_better": n_sym_better,
            "llm_better": n_llm_better,
            "tie": n_tie,
            "delta_u_mean": float(train_du.mean()),
            "delta_u_min": float(train_du.min()),
            "delta_u_max": float(train_du.max()),
        },
        "phase_b_results": {
            ptype: {"dev_regret": policies[ptype]["dev_regret"]}
            for ptype in policies
        },
        "phase_b_unweighted_logistic": {"dev_regret": dev_reg_uw, "weight_mode": "uniform (TRUE w=1)"},
        "phase_b_unweighted_centroid": {"dev_regret": dev_reg_uwc, "weight_mode": "uniform (TRUE w=1)"},
        "phase_b_hard_target": {"dev_regret": dev_reg_hard},
        "phase_c_selection": {
            "best_policy": best_ptype,
            "always_llm_dev_regret": dev_reg_llm,
            "always_sym_dev_regret": dev_reg_sym,
            "hand_router_dev_regret": dev_reg_hand,
        },
        "phase_d_calibration": {
            "tau_ood": tau_ood,
            "tau_conf": best_threshold,
            "preference_brier_soft": brier,
            "action_confidence_ece": ece,
            "calibration_target_source": "delta_u (NOT cal_probs)",
        },
        "phase_e_steering": {
            "n_intervention_results": len(intervention_results),
            "dose_response": {str(k): v for k, v in steering_summary.items()},
            "centroid_vector_norm": v_norm,
            "weight_fallback": getattr(centroid_model, "weight_fallback", False),
            "evidence_level": "REAL_MODEL_LATENT_INTERVENTION",
            "utility_source": "verified_backend_utility",
        },
        "phase_f_final": {
            "candidate_utility": float(final_cand_u.mean()),
            "candidate_regret": final_cand_reg,
            "incumbent_utility": float(final_u_llm.mean()),
            "incumbent_regret": final_reg_llm,
            "always_sym_utility": float(final_u_sym.mean()),
            "always_sym_regret": final_reg_sym,
            "hand_router_utility": float(final_u_hand.mean()),
            "hand_router_regret": final_reg_hand,
            "unweighted_logistic_regret": final_reg_uw,
            "unweighted_centroid_regret": final_reg_uwc,
            "weighted_centroid_regret": final_reg_centroid,
            "logistic_hard_regret": final_reg_hard,
            "mlp_regret": final_reg_mlp,
            "strict_routing_accuracy": strict_acc,
            "tie_aware_routing_accuracy": tie_aware_acc,
            "abstain_rate": float(np.mean([r == "abstain" for r in final_routes])),
            "symbolic_fraction": float(np.mean([r == "symbolic" for r in final_routes])),
            "llm_fraction": float(np.mean([r == "llm" for r in final_routes])),
            "ood_rate": float(reason_counts.get("ood", 0) / len(final_reasons)),
            "abstention_reasons": dict(reason_counts),
            "per_family": per_family,
            "paired_stats": stats,
        },
    }

    out_path = REPO / "artifacts" / "real_qwen_experiment_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nExperiment artifact saved to {out_path}")

    # --- Scientific success tiers ---
    print("\n--- Scientific success tiers ---")
    tier2 = final_cand_reg < final_reg_llm
    tier3 = final_cand_reg < final_reg_hand
    tier4 = final_cand_reg < final_reg_uw
    print(f"  Tier 2 (beats always-LLM):     {'YES' if tier2 else 'NO'} (cand={final_cand_reg:.4f} < llm={final_reg_llm:.4f})")
    print(f"  Tier 3 (beats hand router):    {'YES' if tier3 else 'NO'} (cand={final_cand_reg:.4f} < hand={final_reg_hand:.4f})")
    print(f"  Tier 4 (beats unweighted):     {'YES' if tier4 else 'NO'} (cand={final_cand_reg:.4f} < unw={final_reg_uw:.4f})")

    print("\n" + "=" * 70)
    print("REAL QWEN EXPERIMENT COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
