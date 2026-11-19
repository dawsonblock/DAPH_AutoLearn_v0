# REAL_EVALUATION_PIPELINE — v0.3.10.3.2-alpha

## Real Evaluation Pipeline Specification

**Release:** v0.3.10.3.2-alpha
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`

---

## 1. Overview

The real evaluation pipeline executes the full causal chain:

```
task x
  → pre-execution hidden state h(x)
  → learned policy π(a | h)
  → selected backend
  → actual execution
  → actual verifier
  → canonical utility
  → regret
  → learning / promotion
```

No fitting is allowed during final evaluation.

---

## 2. CLI Commands

### 2.1 Train

```bash
daph-autolearn-policy train \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --train-tasks tasks/train.json \
    --val-tasks tasks/dev.json \
    --output artifacts/policy.json
```

Flow:
1. Load model + tokenizer (frozen revision)
2. For each train task:
   - Capture h(x) once
   - Execute symbolic backend
   - Execute LLM backend
   - Verify both
   - Compute canonical utility
   - Compute ΔU
   - Compute training weight
   - Store immutable experience
3. Fit candidate policies (centroid, logistic, MLP)
4. Save policy artifact

### 2.2 Evaluate

```bash
daph-autolearn-policy evaluate \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --policy-file artifacts/policy.json \
    --test-tasks tasks/final.json \
    --output artifacts/eval.json
```

Flow:
1. Load frozen model/tokenizer
2. Load frozen policy artifact
3. Load frozen feature transform, PCA, OOD model
4. For each test task:
   - Capture real pre-execution activation
   - Apply frozen feature transform
   - OOD check
   - Policy probabilities
   - Route / abstain
   - Backend execution
   - Verifier
   - Canonical utility
   - Regret
5. Save evaluation results

**No fitting allowed.** No PCA.fit, OOD.fit, calibrator.fit, policy.fit,
threshold selection, alpha search, or layer selection.

### 2.3 Calibrate

```bash
daph-autolearn-policy calibrate \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --policy-file artifacts/policy.json \
    --train-tasks tasks/train.json \
    --val-tasks tasks/cal.json \
    --output artifacts/calibration.json
```

Flow:
1. Capture real calibration hidden states
2. Apply frozen TRAIN feature transform
3. Generate policy probabilities
4. Execute both backends
5. Compute ΔU
6. Compute calibration target: q_i = sigmoid(ΔU_i / τ)
7. Fit probability calibration
8. Select confidence threshold
9. Select OOD threshold
10. Save CalibrationArtifact

**No zero features. No placeholder probabilities.**

### 2.4 Intervene

```bash
daph-autolearn-policy intervene \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --policy-file artifacts/policy.json \
    --vector artifacts/steering_vector.json \
    --tasks tasks/dev.json \
    --layer 14 \
    --alpha-grid=-1,-0.5,0,0.5,1 \
    --output artifacts/steering_results.json
```

Flow:
1. Load frozen model + policy + steering vector
2. For each alpha in the grid:
   - For each task:
     - Apply intervention: h + αv
     - Obtain routing probability
     - Select route
     - Execute selected backend
     - Verify result
     - Calculate canonical utility
     - Calculate regret
3. Report ΔU(α), flip rates, beneficial/harmful fractions

**Uses canonical `backend_utility`** — not synthetic utility.

---

## 3. Stage Machine

```
TRAIN → DEV → CALIBRATION → FROZEN → FINAL
```

### 3.1 Freeze Manifest (Section 35)

At FROZEN stage, write:

```json
{
  "experiment_id": "autolearn-v031032-alpha-smoke",
  "source_tree_sha256": "eec93338...",
  "policy_sha256": "...",
  "calibration_sha256": "...",
  "feature_transform_sha256": "...",
  "utility_config_sha256": "...",
  "train_dataset_sha256": "...",
  "dev_dataset_sha256": "...",
  "calibration_dataset_sha256": "...",
  "final_dataset_sha256": "...",
  "model_revision": "Qwen/Qwen2.5-1.5B-Instruct",
  "tokenizer_revision": "Qwen/Qwen2.5-1.5B-Instruct",
  "selected_layer": 14,
  "capture_location": "last_prompt_token",
  "selected_alpha": 0.0,
  "frozen_at": 1234567890.0,
  "release_version": "0.3.10.3.2-alpha"
}
```

### 3.2 Final Access Ledger (Section 36)

- Default final access limit: 1
- After final access: experiment state = SEALED
- No more mutation allowed

### 3.3 Zero Fitting on Final (Section 24)

Prohibited during final:
- PCA.fit
- OOD.fit
- calibrator.fit
- policy.fit
- threshold selection
- alpha search
- layer selection

---

## 4. OOD Model Fitting (Section 26)

- OOD distribution fitted ONLY on TRAIN
- Calibration may select threshold but must not refit covariance/PCA
- FINAL: read-only scoring

---

## 5. Fail-Closed Behavior (Sections 23, 33)

If activation is unavailable: raise / fail closed. Do NOT fabricate
hidden state with `np.zeros(8)`.

If experience is missing: raise `MissingExperienceError`. Do NOT
return `0.0` as utility.

If KL is unavailable in qualified mode: FAIL CLOSED. No null-as-pass.
