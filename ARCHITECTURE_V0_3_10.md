# ARCHITECTURE_V0_3_10.md — DAPH AutoLearn v0.3.10-alpha

## Overview

v0.3.10 upgrades AutoLearn from a "weighted steering-vector extractor" into a
**counterfactual compute-selection learner**. The system learns:

```
state / activation
    → action policy
    → computation choice
    → verified outcome
    → counterfactual utility / regret
    → policy improvement
```

The weighted centroid remains as a **transparent baseline**. The main learner
is a **calibrated contextual policy model** (weighted soft-target logistic
router) operating over LLM hidden-state features.

## Architecture Progression (P0–P10)

```
P0  Fix actual candidate/incumbent evaluation          [v0.3.9, done]
 ↓
P1  Counterfactual utility + regret                    [v0.3.9 + v0.3.10]
 ↓
P2  Weighted centroid baseline                         [v0.3.10]
 ↓
P3  Weighted soft-target logistic router               [v0.3.10, CRITICAL]
 ↓
P4  Calibration + abstention                           [v0.3.10]
 ↓
P5  OOD detection                                      [v0.3.10]
 ↓
P6  +v / -v causal intervention experiments            [v0.3.10]
 ↓
P7  KL + capability-preservation promotion gates       [v0.3.10]
 ↓
P8  Prioritized replay                                 [v0.3.10]
 ↓
P9  Low-rank multi-vector controller                   [v0.3.10, experimental]
 ↓
P10 Partial-feedback / off-policy learning             [v0.3.10 interface only]
```

## New Packages

### `daph_learning/policy/` — The learned policy system

| Module | Purpose |
|--------|---------|
| `types.py` | `BackendOutcome`, `CounterfactualExperience`, `CapturedActivation`, `PolicyDecision`, `Route` enum |
| `confidence.py` | `OutcomeConfidence` with explicit provenance (verifier, measurement, stability, ood) |
| `weighting.py` | `utility_weight`, `WeightConfig`, `snr_weight`, gap-threshold tie truncation |
| `centroid.py` | `weighted_mean`, `weighted_contrastive_mean`, `unweighted_contrastive_mean` (BASELINE) |
| `logistic.py` | `WeightedLogisticRouter`, `soft_preference_target`, `weighted_policy_loss`, hard-label mode |
| `abstention.py` | `choose_route` with calibrated confidence threshold |
| `calibration.py` | Brier score, ECE, reliability bins, selective-risk curve |
| `ood.py` | `MahalanobisOOD` detector with regularized covariance |
| `features.py` | `PCAFeatureReducer` fitted on TRAIN only |
| `config.py` | `ExperimentConfig` (frozen, hashed) |
| `regret.py` | `per_task_regret`, `mean_regret`, `paired_bootstrap_mean_ci`, `paired_promotion_statistics` |
| `learner.py` | `train_policy_learner` — integrated pipeline tying all components together |

### `daph_learning/interventions/` — Causal intervention experiments

| Module | Purpose |
|--------|---------|
| `__init__.py` | `run_intervention_experiment`, `direction_reversal_test`, `dose_response_summary` |
| `kl_gate.py` | `PromotionConstraints`, `evaluate_kl_capability_gate`, `mean_kl_neutral` |

### `daph_learning/replay/` — Prioritized experience replay

`PrioritizedReplayBuffer`, `ReplayExperience`, `replay_priority`.

### `daph_learning/bandit/` — Contextual bandit logging

`PolicyDecision`, `log_policy_decision`, `inverse_propensity_weight`,
`doubly_robust_utility`.

### `daph_learning/lowrank/` — Low-rank multi-vector controller (experimental)

`LowRankSteeringController`, `orthogonality_loss`, `interference_matrix`.

### `daph_learning/environment_synthetic.py` — Synthetic closed-loop environment

`make_synthetic_tasks`, `synthetic_execute_fn`, `synthetic_utility`.

### `daph_learning/latent_verifier.py` — Latent verifier experiment

`LatentVerifier` using trajectory centroids.

## Data Flow

```
1. Tasks arrive (train / dev / calibration / final)
2. Execute BOTH backends counterfactually (execute_both_backends)
3. Verify outcomes independently
4. Compute U(S), U(L), ΔU (compute_backend_utility)
5. Compute sample weight w_i = clip(|ΔU_i|·c_i) with gap truncation
6. Optionally reduce features (PCA on TRAIN only)
7. Fit OOD detector (Mahalanobis on TRAIN only)
8. Train weighted logistic router: p = σ(w^T h + b) against soft targets q = σ(ΔU/τ)
9. Evaluate on dev: candidate vs incumbent, compute regret
10. Calibrate abstention threshold on calibration split
11. Run causal intervention experiments (+v/0/-v dose-response)
12. KL/capability promotion gate
13. Final test evaluation (once, frozen)
```

## Split Discipline (Section 34)

| Split | May fit | May select |
|-------|---------|------------|
| TRAIN | centroids, logistic weights, PCA, covariance, feature selectors | — |
| DEV | — | algorithm, regularization, layer, temperature, steering params |
| CALIBRATION | — | probability calibration, abstention threshold, OOD threshold |
| FINAL TEST | — | used once for final frozen evaluation |

## Action Space

Primary (v0.3.10): `SYMBOLIC`, `LLM`, `ABSTAIN`

Future (v0.4): `HYBRID`, `VERIFY`, `RETRIEVE`, `THINK_MORE`, `TOOL`, ...

The `Route` enum and APIs are designed to grow.

## Key Design Principles

1. **Weighted centroid is a baseline**, not the final algorithm.
2. **The learned router is the policy.**
3. **Counterfactual utility is supervision.**
4. **Regret is the primary objective.**
5. **Intervention experiments are causal evidence.**
6. **Abstention and OOD detection are part of the policy.**
7. **Promotion is a statistical decision.**
8. **No oracle may secretly choose the candidate action.**
9. **No test set may influence training or tuning.**
