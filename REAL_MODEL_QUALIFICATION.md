# REAL MODEL QUALIFICATION — V0.3.9

## Overview

This document describes how to run the small real-model qualification
experiment for DAPH AutoLearn v0.3.9. The qualification compares six
conditions on a small local model to determine whether the learned
AutoLearn controller improves computation selection.

## Prerequisites

```bash
pip install torch>=2.1 transformers>=4.45 accelerate>=0.21 safetensors>=0.4
```

## Preferred Model

```
Qwen/Qwen2.5-1.5B-Instruct
```

Alternative (if fully compatible): `Qwen3-1.7B`

## Dataset Splits

| Split        | Size  | Purpose                                      |
|--------------|-------|----------------------------------------------|
| train        | 2,000 | Counterfactual experience + candidate update |
| development  | 500   | Held-out candidate-vs-incumbent evaluation   |
| calibration  | 500   | Alpha/layer selection (not used in training) |
| final test   | 1,000 | Final evaluation (inaccessible during tuning)|

Smaller smoke-test subsets (e.g. 50/20/20/50) can be used for initial
verification.

## Conditions Compared

| Label | Description |
|-------|-------------|
| A | Base LLM (always route to LLM) |
| B | Hand-coded router (capability heuristic) |
| C | Fixed steering (non-learned vector) |
| D | Legacy AutoLearn (`--engine legacy`) |
| E | Corrected counterfactual AutoLearn (`--engine counterfactual`) |
| F | Matched random steering (same norm/layer/alpha) |

## Primary Metric

Mean downstream utility on the final-test set.

## Secondary Metrics

- Task accuracy
- Routing accuracy
- Abstention rate
- Latency
- Symbolic utilization
- LLM utilization
- Per-family performance
- Capture coverage
- Steering clamp frequency

## Commands

### 1. Prepare datasets

```bash
# Build protocol datasets with disjoint splits
daph-build-protocol-dataset --output data/ --train 2000 --dev 500 --cal 500 --test 1000
```

### 2. Run corrected counterfactual AutoLearn (Condition E)

```bash
daph-autolearn \
    --engine counterfactual \
    --train-tasks data/train.jsonl \
    --val-tasks data/dev.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --layer 24 \
    --alpha 1.0 \
    --n-iterations 10 \
    --eta 0.5 \
    --abstention-band 0.1 \
    --capture-token-scope anchor \
    --prompt-format chat \
    --output artifacts/counterfactual_result.json \
    --vector-output artifacts/counterfactual_vector.npz
```

### 3. Run legacy AutoLearn (Condition D, for regression comparison)

```bash
daph-autolearn \
    --engine legacy \
    --train-tasks data/train.jsonl \
    --val-tasks data/dev.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --layer 24 \
    --alpha 1.0 \
    --n-iterations 10 \
    --capture-token-scope anchor \
    --prompt-format chat \
    --output artifacts/legacy_result.json \
    --vector-output artifacts/legacy_vector.npz
```

### 4. Run random-direction control (Condition F)

```bash
daph-random-control \
    --train-tasks data/train.jsonl \
    --val-tasks data/dev.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --layer 24 \
    --alpha 1.0 \
    --n-random 500 \
    --output artifacts/random_control_result.json
```

### 5. Evaluate on final-test set

```bash
# Evaluate the promoted vector on the final-test set
daph-evaluate-routes \
    --tasks data/test.jsonl \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --steering artifacts/counterfactual_vector.npz \
    --layer 24 \
    --alpha 1.0 \
    --output artifacts/final_test_evaluation.json
```

## Important Notes

- **Do NOT optimize against final-test results.** The final-test set must
  remain inaccessible during training, candidate generation, hyperparameter
  tuning, alpha selection, layer selection, and promotion decisions.
- The corrected counterfactual engine (`--engine counterfactual`) is the
  default. The legacy engine (`--engine legacy`) is deprecated and emits a
  `DeprecationWarning`.
- For the random-direction control, use N >= 500 where compute permits.
  Report the empirical p-value: `p = (1 + count(random >= learned)) / (N + 1)`.
- The qualification script does not make network/model availability a
  unit-test requirement. The model must be downloaded separately.
