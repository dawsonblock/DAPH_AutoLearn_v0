"""Generate machine-readable results for v0.3.10 (Section 48).

Runs the synthetic closed-loop environment and produces the baseline
matrix table with utility, regret, accuracy, Brier, ECE, abstention, and
neutral KL metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from daph_learning import __version__
from daph_learning.policy.config import ExperimentConfig
from daph_learning.policy.learner import build_counterfactual_experiences, train_policy_learner
from daph_learning.policy.types import Route
from daph_learning.policy.abstention import choose_route
from daph_learning.policy.calibration import brier_score, expected_calibration_error
from daph_learning.policy.regret import mean_regret, per_task_regret
from daph_learning.policy.centroid import weighted_contrastive_mean, unweighted_contrastive_mean
from daph_learning.policy.weighting import utility_weight, WeightConfig
from daph_learning.policy.confidence import OutcomeConfidence
from daph_learning.environment_synthetic import (
    make_synthetic_tasks, synthetic_execute_fn, synthetic_utility, synthetic_oracle_utility,
)


def _run_baseline_matrix(seed: int = 42) -> dict:
    """Run the baseline matrix (Section 25) on the synthetic environment."""
    train_tasks = make_synthetic_tasks(n_per_family=80, n_ties=10, dim=8, seed=seed)
    dev_tasks = make_synthetic_tasks(n_per_family=40, n_ties=10, dim=8, seed=seed + 1)

    cfg = ExperimentConfig(
        gap_threshold=0.1, abstention_band=0.1,
        target_temperature=0.5, confidence_threshold=0.55,
        random_seed=seed,
    )

    train_exp = build_counterfactual_experiences(train_tasks, execute_fn=synthetic_execute_fn, config=cfg)
    dev_exp = build_counterfactual_experiences(dev_tasks, execute_fn=synthetic_execute_fn, config=cfg)
    train_acts = np.array([t["activation"] for t in train_tasks], dtype=np.float32)
    dev_acts = np.array([t["activation"] for t in dev_tasks], dtype=np.float32)

    oracle_utils = np.array([synthetic_oracle_utility(t) for t in dev_tasks], dtype=np.float64)

    results = {}

    # A. Base LLM only (always chooses LLM).
    llm_utils = np.array([synthetic_utility(t, Route.LLM) for t in dev_tasks], dtype=np.float64)
    results["base_llm"] = {
        "utility": float(llm_utils.mean()),
        "regret": float(mean_regret(llm_utils, oracle_utils)),
        "accuracy": float(np.mean([synthetic_utility(t, Route.LLM) >= 0.5 for t in dev_tasks])),
    }

    # B. Static hand-coded router (symbolic if h[0] > 0, else LLM).
    hand_routes = ["symbolic" if t["activation"][0] > 0 else "llm" for t in dev_tasks]
    hand_utils = np.array([synthetic_utility(t, r) for t, r in zip(dev_tasks, hand_routes)], dtype=np.float64)
    results["hand_router"] = {
        "utility": float(hand_utils.mean()),
        "regret": float(mean_regret(hand_utils, oracle_utils)),
        "accuracy": float(np.mean([u >= 0.5 for u in hand_utils])),
    }

    # C. Unweighted centroid.
    pos_mask = [e.preferred_action == Route.SYMBOLIC for e in train_exp]
    neg_mask = [e.preferred_action == Route.LLM for e in train_exp]
    pos_acts = train_acts[pos_mask]
    neg_acts = train_acts[neg_mask]
    if len(pos_acts) > 0 and len(neg_acts) > 0:
        v_unw = unweighted_contrastive_mean(pos_acts, neg_acts, normalize=True)
        unw_routes = ["symbolic" if float(h @ v_unw) > 0 else "llm" for h in dev_acts]
        unw_utils = np.array([synthetic_utility(t, r) for t, r in zip(dev_tasks, unw_routes)], dtype=np.float64)
        results["unweighted_centroid"] = {
            "utility": float(unw_utils.mean()),
            "regret": float(mean_regret(unw_utils, oracle_utils)),
            "accuracy": float(np.mean([u >= 0.5 for u in unw_utils])),
        }

    # D. Utility-weighted centroid.
    pos_weights = np.array([e.sample_weight for e in train_exp if e.preferred_action == Route.SYMBOLIC], dtype=np.float32)
    neg_weights = np.array([e.sample_weight for e in train_exp if e.preferred_action == Route.LLM], dtype=np.float32)
    if len(pos_acts) > 0 and len(neg_acts) > 0:
        v_w = weighted_contrastive_mean(pos_acts, pos_weights, neg_acts, neg_weights, normalize=True)
        w_routes = ["symbolic" if float(h @ v_w) > 0 else "llm" for h in dev_acts]
        w_utils = np.array([synthetic_utility(t, r) for t, r in zip(dev_tasks, w_routes)], dtype=np.float64)
        results["weighted_centroid"] = {
            "utility": float(w_utils.mean()),
            "regret": float(mean_regret(w_utils, oracle_utils)),
            "accuracy": float(np.mean([u >= 0.5 for u in w_utils])),
        }

    # E/F. Weighted logistic router (soft + hard targets).
    def incumbent_route_fn(h):
        return "llm"

    def utility_fn(task, route):
        return synthetic_utility(task, route)

    # Soft targets.
    cfg_soft = ExperimentConfig(
        gap_threshold=0.1, abstention_band=0.1,
        target_temperature=0.5, confidence_threshold=0.55,
        random_seed=seed, target_mode="soft", weight_mode="clipped_gap",
    )
    result_soft = train_policy_learner(
        train_exp, train_acts, config=cfg_soft,
        dev_experiences=dev_exp, dev_activations=dev_acts,
        incumbent_route_fn=incumbent_route_fn, utility_fn=utility_fn,
        dev_tasks=dev_tasks,
    )
    m = result_soft.dev_metrics
    results["soft_logistic"] = {
        "utility": m.get("mean_candidate_utility", 0.0),
        "regret": m.get("mean_candidate_regret", 0.0),
        "accuracy": float(np.mean([u >= 0.5 for u in (result_soft.candidate_utilities if result_soft.candidate_utilities is not None else np.array([0]))])),
        "brier": m.get("brier_score", 0.0),
        "ece": m.get("ece", 0.0),
        "abstain": m.get("abstain_rate", 0.0),
    }

    # Hard targets.
    cfg_hard = ExperimentConfig(
        gap_threshold=0.1, abstention_band=0.1,
        target_temperature=0.01, confidence_threshold=0.55,
        random_seed=seed, target_mode="hard", weight_mode="clipped_gap",
    )
    result_hard = train_policy_learner(
        train_exp, train_acts, config=cfg_hard,
        dev_experiences=dev_exp, dev_activations=dev_acts,
        incumbent_route_fn=incumbent_route_fn, utility_fn=utility_fn,
        dev_tasks=dev_tasks,
    )
    m_h = result_hard.dev_metrics
    results["hard_logistic"] = {
        "utility": m_h.get("mean_candidate_utility", 0.0),
        "regret": m_h.get("mean_candidate_regret", 0.0),
        "accuracy": float(np.mean([u >= 0.5 for u in (result_hard.candidate_utilities if result_hard.candidate_utilities is not None else np.array([0]))])),
        "brier": m_h.get("brier_score", 0.0),
        "ece": m_h.get("ece", 0.0),
        "abstain": m_h.get("abstain_rate", 0.0),
    }

    # G. Random matched steering direction.
    rng = np.random.default_rng(seed)
    v_rand = rng.normal(0, 1, size=8).astype(np.float32)
    v_rand = v_rand / np.linalg.norm(v_rand)
    rand_routes = ["symbolic" if float(h @ v_rand) > 0 else "llm" for h in dev_acts]
    rand_utils = np.array([synthetic_utility(t, r) for t, r in zip(dev_tasks, rand_routes)], dtype=np.float64)
    results["random_steering"] = {
        "utility": float(rand_utils.mean()),
        "regret": float(mean_regret(rand_utils, oracle_utils)),
        "accuracy": float(np.mean([u >= 0.5 for u in rand_utils])),
    }

    # H. Incumbent (always LLM).
    results["incumbent"] = results["base_llm"].copy()

    # I. AutoLearn v0.3.10 (= soft logistic).
    results["autolearn_v0310"] = results["soft_logistic"].copy()

    return results


def main() -> int:
    results = _run_baseline_matrix()
    output = {
        "version": __version__,
        "experiment": "synthetic_closed_loop_baseline_matrix",
        "baseline_matrix": results,
    }
    out_path = REPO / "MACHINE_READABLE_RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results written to {out_path}")
    # Print table.
    print(f"\n{'Method':<25} {'Utility':>8} {'Regret':>8} {'Accuracy':>8} {'Brier':>8} {'ECE':>8} {'Abstain':>8}")
    print("-" * 85)
    for method, m in results.items():
        print(f"{method:<25} {m.get('utility',0):>8.4f} {m.get('regret',0):>8.4f} "
              f"{m.get('accuracy',0):>8.4f} {m.get('brier',0):>8.4f} {m.get('ece',0):>8.4f} "
              f"{m.get('abstain',0):>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
