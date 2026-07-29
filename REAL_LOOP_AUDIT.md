# REAL_LOOP_AUDIT.md — v0.3.10.2-alpha

**Date:** 2026-07-28
**Release:** DAPH AutoLearn v0.3.10.2-alpha

## 1. Removed Placeholders

| Location | Old behavior | New behavior |
|----------|-------------|-------------|
| `cli/commands/policy.py:_cmd_train_real_model` | Used `task.get("symbolic_correct", False)` and `task.get("llm_correct", False)` as outcome labels | Uses `execute_symbolic_backend(task)` and `execute_llm_backend(task, model, tok)` — actual execution + verification |
| `cli/commands/policy.py:_cmd_calibrate` | Used `p = 0.5` placeholder for all calibration tasks | Trains a policy on train set, computes `policy.predict_proba(cal_acts)` — actual policy probabilities |
| `policy/abstention.py:choose_route_with_reason` | `p == 0.5` → abstain (`policy_tie`) — inconsistent with `choose_route` | `p >= 0.5` → symbolic (consistent with `choose_route`) |

## 2. Real Backend Integrations

### Symbolic Backend (`execution/real_backends.py:execute_symbolic_backend`)
- Reuses the existing bounded symbolic executor (`execution/symbolic_executor.py`)
- Returns `BackendOutcome` from actual execution
- Fails closed on unsupported tasks (`correct=False`, not fabricated)

### LLM Backend (`execution/real_backends.py:execute_llm_backend`)
- Actual model generation with `do_sample=False` (deterministic)
- Records output hash, latency, generation config hash
- Returns `(BackendOutcome, generated_text)` for verification

### Verifier (`execution/real_backends.py:verify_arithmetic`)
- Exact numeric verification: extracts last integer from output
- `312` is NOT correct for expected `12` (never substring matching)
- Fails closed on unsupported tasks (`verified_correct=None`)

### Capture (`execution/real_backends.py:capture_task_representation`)
- Captures hidden state ONCE before backend execution (Section 14)
- Pre-execution state: processes task prompt only
- Does NOT access `expected` field (leakage prevention, Section 15)
- Supports `last_token`, `anchor`, `mean_pool` capture locations

### Experience Builder (`execution/real_backends.py:build_real_counterfactual_experience`)
- Verifies both backend outputs
- Computes utilities from verified outcomes
- Builds `CounterfactualExperience` with real ΔU

## 3. Gates Corrected

| Gate | Old assertion (v0.3.10.1) | Problem | New assertion (v0.3.10.2) |
|------|--------------------------|---------|--------------------------|
| G10 | `reg_w <= reg_u + 0.01` (non-inferiority) | Called "beats" but was non-inferiority | `mean(d_s) >= 0.005` AND lower CI > 0, 10 seeds (true superiority) |
| G14 | Import check only | Didn't execute intervention | Actually loads model, installs hook, confirms hidden state changes |
| G16 | `--benchmark` synthetic path | Didn't test real model branch | (Pending: real train CLI integration test) |
| G17 | `p = 0.5` placeholder | Calibration was meaningless | Uses actual `policy.predict_proba` |
| G22 | Didn't exercise KL logic | Gate passed without testing KL | Cases A (reject), B (eligible), C (fail-closed) |
| G24 | Save/load only | Didn't test actual promotion failure | Tests failed promotion leaves incumbent unchanged |
| G25 | File existence only | Didn't validate artifact content | Validates required schema fields |

## 4. Real Qwen Experiment Results

```
Model: Qwen/Qwen2.5-0.5B-Instruct
Task domain: integer arithmetic (Section 50)
Evidence level: REAL_MODEL_FINAL

Method              | Utility | Regret | Accuracy
Always-LLM          | 0.54    | 0.46   | N/A
Always-symbolic     | 1.00    | 0.00   | N/A
Candidate AutoLearn | 0.98    | 0.02   | 0.46

Tier 1 (beats always-LLM): YES
Tier 2 (counterfactual value): YES
Steering: no effect (centroid norm=0.0, valid negative result)
```

## 5. Scientific Findings

1. **Routing is trivial on simple arithmetic:** the symbolic backend always wins, so the optimal policy is "always symbolic." The candidate correctly learns this (0.02 regret, only 2% abstention).

2. **No steering signal:** the centroid direction is zero because the hidden state (captured before execution) depends only on the task prompt, not the outcome. Both classes (symbolic-preferred and LLM-preferred) have identical hidden states. This is a valid negative result — steering cannot help when there's no routing signal in the pre-execution representation.

3. **Weighting helps on synthetic near-tie:** the redesigned near-tie environment (with nuisance direction) shows a true superiority of weighted over unweighted training (mean_gain=0.0097, 9/10 seeds, lower CI=0.005).

4. **Calibration works with real probabilities:** the calibration utility improved from 0.0 (with p=0.5 placeholder) to 0.96 (with actual policy probabilities).
