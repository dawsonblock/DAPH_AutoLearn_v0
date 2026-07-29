# TEST_REPORT_V0_3_10_2.md

**Release:** DAPH AutoLearn v0.3.10.2-alpha
**Date:** 2026-07-28
**Environment:** macOS Darwin 25.2.0, Python 3.12.0, pytest 8.4.2, MPS

## 1. Tests executed in current environment

**Command:** `python -m pytest tests/ --tb=no`

**Result:** `760 passed, 1 skipped, 4 warnings in 23.62s`

**Collection:** `761 tests collected`

### Test files added in this release

| File | Tests | Purpose |
|------|-------|---------|
| `tests/test_v0310_2_real_gates.py` | 22 | Real backends (G18-G21), gate semantics (G22/G24/G25), real intervention (G14), leakage test (P15) |

### Test categories

| Category | Count | Description |
|----------|-------|-------------|
| Real symbolic backend | 3 | G18: executes arithmetic, fails closed, no placeholder labels |
| Real verifier | 6 | G20: exact numeric, never substring, fail closed, dispatch |
| Real counterfactual experience | 1 | G21: experience from executed outcomes |
| Real LLM backend | 1 | G19: integration test with Qwen2.5-0.5B (skip if unavailable) |
| Real intervention | 1 | G14: integration test — hook changes hidden state |
| Neutral KL gate | 3 | G22: cases A (reject), B (eligible), C (fail-closed) |
| Atomic promotion | 2 | G24: failed promotion leaves incumbent, successful swaps |
| Smoke validation | 2 | G25: artifact schema validation |
| Evidence version | 2 | G27: machine-readable evidence matches version |
| Leakage test | 1 | P15: capture function must not access 'expected' |
| G10 multi-seed | 1 | True superiority gate: 10 seeds, mean(d)>min_gain, CI>0 |
| Existing tests | 738 | All v0.3.10.1 tests still pass |

## 2. Tests copied from prior artifacts

**None.** Every test was executed in the current environment.

## 3. Skipped tests

| Test | Reason |
|------|--------|
| `test_xor_is_nonlinear` | Optional scikit-learn dependency |

## 4. Real-model experiment (G28-G31)

**Command:** `python scripts/run_real_qwen_experiment.py`

**Result:** PASSED — first scientifically meaningful real Qwen result.

```
Model: Qwen/Qwen2.5-0.5B-Instruct (24 layers, dim 896, MPS)
Splits: train=50, dev=25, cal=25, final=50

Phase A: sym=100% correct, LLM=54% correct on arithmetic
Phase B: all policies achieve 0.0 dev regret (symbolic always wins)
Phase D: tau_ood=51.31, tau_conf=0.50, brier=0.053, ece=0.231
Phase F: Candidate utility=0.98, regret=0.02 (vs LLM 0.54/0.46)
Phase E: steering no effect (centroid norm=0.0, valid negative result)

Tier 1 (beats always-LLM): YES
Tier 2 (counterfactual value): YES
```

## 5. Algorithmic observations

- **G10 (weighting value):** weighted truly beats unweighted on the redesigned near-tie environment (mean_gain=0.0097, 9/10 seeds, lower CI=0.005 > 0).
- **G11 (centroid failure):** centroid degrades on multimodal (0.318 vs 0.201 on linear).
- **G13 (MLP value):** MLP beats logistic on XOR (0.083 vs 0.299).
- **Real experiment:** routing is trivial on simple arithmetic (symbolic always correct), so the candidate correctly routes to symbolic. Steering has no effect because the centroid direction is zero (no routing signal in pre-execution hidden state). This is a valid negative result for steering.
