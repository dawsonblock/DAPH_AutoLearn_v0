# REAL MODEL QUALIFICATION — v0.3.10.3.2-alpha

## Overview

This document describes how to run the real-model qualification experiment for
DAPH AutoLearn v0.3.10.3.2-alpha. The qualification compares the learned routing
policy against a baseline matrix (A–K) on a small local model to determine
whether the learned policy improves computation selection (reduces held-out
**regret**) over the incumbent. Regret — not routing accuracy — is the primary
metric.

> **v0.3.10.3.2 changes vs. v0.3.10.3.1.** The crossover benchmark now has
> within-subtype crossover (all 6 subtypes have both symbolic-preferred and
> LLM-preferred instances). A semantic parser allows the symbolic backend to
> compete on NL tasks. CLI paths (evaluate, calibrate, intervene) are completed
> with real backends. A freeze manifest binds final evaluation to the frozen
> state. The 32-gate registry is fully executed. Source:
> `src/daph_learning/cli/commands/policy.py`, `src/daph_learning/data/crossover_benchmark.py`,
> `src/daph_learning/data/semantic_parser.py`, `src/daph_learning/policy/stage.py`.

## Prerequisites

```bash
pip install torch>=2.1 transformers>=4.45 accelerate>=0.21 safetensors>=0.4
```

The model must be downloaded separately; the qualification scripts do not make
network/model availability a unit-test requirement.

## Preferred Model

```
Qwen/Qwen2.5-1.5B-Instruct
```

Alternative (if fully compatible): `Qwen3-1.7B`. The real-model residual-stream
hook resolves Qwen2 / Llama / GPT-2 architectures
(`src/daph_learning/interventions/real_pipeline.py`).

## Dataset Splits

Two split configurations are defined (Section 37):

### Smoke split (initial verification)

| Split        | Size | Purpose                                      |
|--------------|------|----------------------------------------------|
| train        | 100  | Counterfactual experience + candidate update |
| development  | 50   | Held-out candidate-vs-incumbent evaluation   |
| calibration  | 50   | Alpha/layer/OOD-threshold selection          |
| final test   | 100  | Final evaluation (inaccessible during tuning)|

### Research split (primary study)

| Split        | Size  | Purpose                                      |
|--------------|-------|----------------------------------------------|
| train        | 2,000 | Counterfactual experience + candidate update |
| development  | 500   | Held-out candidate-vs-incumbent evaluation   |
| calibration  | 500   | Alpha/layer/OOD-threshold selection          |
| final test   | 1,000 | Final evaluation (inaccessible during tuning)|

Splits must be family-disjoint (no prompt/family leakage across splits). The
final-test set must remain inaccessible during training, candidate generation,
hyperparameter tuning, alpha selection, layer selection, and promotion
decisions (final-test access ledger — Section 39).

## Baseline Matrix (A–K)

All methods are evaluated on identical splits and identical metrics (Section
37). The candidate is the full v0.3.10.1 system.

| Label | Description |
|-------|-------------|
| A | Always-LLM (base model, no routing) |
| B | Always-symbolic |
| C | Hand router (human-designed routing heuristic) |
| D | Unweighted centroid (each example contributes equally) |
| E | Weighted centroid (utility-weighted contrastive mean) |
| F | Soft logistic (soft targets `q = σ(ΔU/τ)`) |
| G | Hard logistic (hard targets, ties masked out) |
| H | MLP router (`mlp_experimental`, diagnostic) |
| I | Random direction (matched norm/layer/alpha, ≥ 500 directions) |
| J | Incumbent (current best prior AutoLearn version) |
| K | Candidate (full v0.3.10.1 learned policy) |

## Primary Metric

**Mean regret on the final-test set:**

```
Regret_i = max_a U_i(a) − U_i(π(x_i))
R̄       = (1/N) Σ_i Regret_i
```

The candidate (K) must reduce mean regret relative to the incumbent (J) and
beat the median random direction (I).

## Secondary Metrics

- Mean downstream utility
- Task accuracy
- Routing accuracy
- Abstention rate (by reason: `low_confidence`, `ood`, `policy_tie`)
- Latency
- Symbolic / LLM utilization
- Per-family performance
- Capture coverage
- Steering clamp frequency
- `preference_brier_soft`, `action_confidence_ece` (corrected calibration)

## CLI Commands

The policy learner CLI is `daph-autolearn-policy` with four modes: `train`,
`evaluate`, `intervene`, `calibrate` (source:
`src/daph_learning/cli/commands/policy.py`).

### 1. Prepare datasets

```bash
# Smoke split (100/50/50/100)
daph-build-protocol-dataset --output data/smoke/ \
    --train 100 --dev 50 --cal 50 --test 100

# Research split (2000/500/500/1000)
daph-build-protocol-dataset --output data/research/ \
    --train 2000 --dev 500 --cal 500 --test 1000
```

### 2. Smoke run — train the candidate policy (Condition K)

Use the smoke split for a fast end-to-end verification that the pipeline runs,
both backends execute, features are captured and joined by `task_id`, and the
policy trains and evaluates on dev.

```bash
# Synthetic closed-loop smoke (no model required)
daph-autolearn-policy train --synthetic \
    --policy logistic \
    --soft-targets \
    --target-temperature 1.0 \
    --weight-mode gap \
    --gap-threshold 0.0 \
    --layer 24 \
    --alpha 1.0 \
    --confidence-threshold 0.70 \
    --ood-threshold inf \
    --seed 0 \
    --output artifacts/smoke_policy_train_result.json

# Real-model smoke (Qwen2.5-1.5B-Instruct, 100/50/50/100)
daph-autolearn-policy train \
    --train-tasks data/smoke/train.jsonl \
    --dev-tasks data/smoke/dev.jsonl \
    --policy logistic \
    --soft-targets \
    --target-temperature 1.0 \
    --weight-mode gap \
    --gap-threshold 0.0 \
    --layer 24 \
    --alpha 1.0 \
    --confidence-threshold 0.70 \
    --ood-threshold inf \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --seed 0 \
    --output artifacts/smoke_policy_train_result.json
```

> **Note on CLI flags.** The `daph-autolearn-policy` CLI in
> `src/daph_learning/cli/commands/policy.py` exposes `--soft-targets` /
> `--no-soft-targets` and `--weight-mode gap|snr` for backwards compatibility.
> The underlying `ExperimentConfig` (Section 1/2) uses the corrected
> `target_mode: "soft"|"hard"` and the 4-mode `weight_mode`
> (`uniform|absolute_gap|clipped_gap|snr`). The clean-break enum values are
> the canonical forms; the legacy CLI aliases map to `target_mode="soft"` and
> `weight_mode="clipped_gap"` respectively.

### 3. Research run — train each baseline (Conditions D–H, K)

Run the research split (2000/500/500/1000) for each policy configuration.

```bash
# D — Unweighted centroid
daph-autolearn-policy train \
    --train-tasks data/research/train.jsonl \
    --dev-tasks data/research/dev.jsonl \
    --policy centroid \
    --weight-mode gap \
    --gap-threshold 0.0 \
    --layer 24 --alpha 1.0 --confidence-threshold 0.70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
    --output artifacts/research_D_unweighted_centroid.json

# E — Weighted centroid
daph-autolearn-policy train \
    --train-tasks data/research/train.jsonl \
    --dev-tasks data/research/dev.jsonl \
    --policy centroid \
    --weight-mode gap --gap-threshold 0.0 \
    --layer 24 --alpha 1.0 --confidence-threshold 0.70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
    --output artifacts/research_E_weighted_centroid.json

# F — Soft logistic (primary learner)
daph-autolearn-policy train \
    --train-tasks data/research/train.jsonl \
    --dev-tasks data/research/dev.jsonl \
    --policy logistic \
    --soft-targets --target-temperature 1.0 \
    --weight-mode gap --gap-threshold 0.0 \
    --layer 24 --alpha 1.0 --confidence-threshold 0.70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
    --output artifacts/research_F_soft_logistic.json

# G — Hard logistic (ties masked out)
daph-autolearn-policy train \
    --train-tasks data/research/train.jsonl \
    --dev-tasks data/research/dev.jsonl \
    --policy logistic \
    --no-soft-targets --gap-threshold 0.1 \
    --weight-mode gap \
    --layer 24 --alpha 1.0 --confidence-threshold 0.70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
    --output artifacts/research_G_hard_logistic.json

# H — MLP router (diagnostic, experimental)
daph-autolearn-policy train \
    --train-tasks data/research/train.jsonl \
    --dev-tasks data/research/dev.jsonl \
    --policy mlp_experimental \
    --soft-targets --target-temperature 1.0 \
    --weight-mode gap --gap-threshold 0.0 \
    --layer 24 --alpha 1.0 --confidence-threshold 0.70 \
    --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
    --output artifacts/research_H_mlp.json

# K — Candidate (full system, soft logistic, calibrated OOD)
daph-autolearn-policy train \
    --train-tasks data/research/train.jsonl \
    --dev-tasks data/research/dev.jsonl \
    --policy logistic \
    --soft-targets --target-temperature 1.0 \
    --weight-mode clipped_gap --gap-threshold 0.0 \
    --layer 24 --alpha 1.0 --confidence-threshold 0.70 \
    --ood-threshold 10.0 \
    --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
    --output artifacts/research_K_candidate.json
```

> Conditions A (always-LLM) and B (always-symbolic) do not require training —
> they route every task to the same backend. Condition C (hand router) uses a
> human-designed heuristic. Condition J (incumbent) loads the prior best
> AutoLearn artifact.

### 4. Intervention study (Sections 40–41)

Run the dose-response intervention on the dev set with the candidate steering
vector, then repeat the frozen best setting once on the final-test set.

```bash
# Dose-response on dev: alpha ∈ {-A, -A/2, 0, A/2, A}
daph-autolearn-policy intervene \
    --vector-file artifacts/candidate_vector.npy \
    --layer 24 --alpha 1.0 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output artifacts/intervention_dose_response.json
```

The real-model residual-stream pipeline installs a hook at the target layer,
captures the baseline hidden state, and runs `h' = h + alpha * v` for each
alpha, recording route/utility/KL/clamp telemetry as `RealInterventionResult`
with `evidence_level = "real_model_causal"`
(`src/daph_learning/interventions/real_pipeline.py`).

### 5. Calibrate abstention / OOD thresholds

```bash
# Tune τ_conf and τ_OOD on the calibration split (quantile-based τ_OOD)
daph-autolearn-policy calibrate \
    --calibration-tasks data/research/cal.jsonl \
    --policy logistic \
    --confidence-threshold 0.70 \
    --ood-threshold 10.0 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output artifacts/calibration_result.json
```

OOD threshold is calibrated (quantile-based `ood_quantile = 0.99`), not
infinite by default in qualified runs
(`src/daph_learning/policy/config.py`).

### 6. Random-direction control (Condition I)

```bash
daph-random-control \
    --train-tasks data/research/train.jsonl \
    --val-tasks data/research/dev.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --layer 24 \
    --alpha 1.0 \
    --n-random 500 \
    --output artifacts/random_control_result.json
```

Use `N >= 500` where compute permits. Report the empirical p-value:
`p = (1 + count(random >= learned)) / (N + 1)`.

### 7. Final evaluation (Condition K on final-test)

```bash
# Evaluate the promoted policy on the final-test set (ONE pass)
daph-autolearn-policy evaluate \
    --policy-file artifacts/candidate_policy.pt \
    --test-tasks data/research/test.jsonl \
    --policy logistic \
    --confidence-threshold 0.70 \
    --ood-threshold 10.0 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output artifacts/final_test_evaluation.json
```

### 8. Legacy AutoLearn regression comparison (Condition J incumbent)

```bash
# Incumbent / legacy engine for regression comparison
daph-autolearn \
    --engine legacy \
    --train-tasks data/research/train.jsonl \
    --val-tasks data/research/dev.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --layer 24 --alpha 1.0 --n-iterations 10 \
    --capture-token-scope anchor --prompt-format chat \
    --output artifacts/legacy_result.json \
    --vector-output artifacts/legacy_vector.npz
```

The legacy engine is deprecated and emits a `DeprecationWarning`. The corrected
counterfactual engine (`--engine counterfactual`) is the default.

## Important Notes

- **Do NOT optimize against final-test results.** The final-test set must
  remain inaccessible during training, candidate generation, hyperparameter
  tuning, alpha selection, layer selection, and promotion decisions. Maintain
  a final-test access ledger (Section 39).
- **Regret is the primary metric.** A policy can have high routing accuracy but
  make its mistakes on tasks where the utility penalty is enormous. The
  learning objective and evaluation metric must preserve this distinction
  (`src/daph_learning/policy/regret.py`).
- **Candidate vs. incumbent uses actual policy decisions** (Section 22). The
  oracle `max_a U(a)` is used ONLY for scoring regret, never for choosing the
  candidate action.
- For the random-direction control, use `N >= 500` where compute permits.
  Report the empirical p-value: `p = (1 + count(random >= learned)) / (N + 1)`.
- Every reported result must be tagged with an evidence level:
  `UNIT` | `SYNTHETIC` | `REAL_MODEL_DEV` | `REAL_MODEL_FINAL` (Section 41).
- The qualification script does not make network/model availability a
  unit-test requirement. The model must be downloaded separately.
- Multi-seed: run every cell across at least 5 seeds; report mean ± std and
  per-seed values. A result is significant only if the mean effect exceeds the
  standard deviation by the configured factor and the sign is consistent
  across all seeds.
