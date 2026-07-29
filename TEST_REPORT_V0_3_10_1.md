# TEST_REPORT_V0_3_10_1.md

**Release:** DAPH AutoLearn v0.3.10.1-alpha
**Date:** 2026-07-28
**Environment:** macOS Darwin 25.2.0, Python 3.12.0, pytest 8.4.2
**GPU:** Apple Silicon MPS (no CUDA)
**Models available locally:** Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct

## 1. Tests executed in current environment

**Command:** `python -m pytest tests/ --tb=no`

**Result:**
```
737 passed, 1 skipped, 4 warnings in 18.23s
```

**Collection:** `738 tests collected` (1 skipped due to optional dependency).

### Test files added in this release

| File | Tests | Purpose |
|------|-------|---------|
| `tests/test_v0310_1_p0_gates.py` | 49 | P0 correctness repairs (G1-G8): TargetMode, WeightMode, policy_type, fail-closed, task-ID alignment, weighted_mean validation, calibration math, dev-regret early stopping |
| `tests/test_v0310_1_benchmark.py` | 19 | Redesigned synthetic benchmark + scientific tests (G9-G14): random direction control, weighting value, centroid failure, linear router value, nonlinear router value, random steering |
| `tests/test_v0310_1_gates.py` | 17 | Comparative gate, capability gate, real intervention pipeline, atomic promotion, OOD calibration (G14, G16, G20, G22, G23) |

### Test categories

| Category | Count | Description |
|----------|-------|-------------|
| P0 correctness | 49 | G1-G8: the specific bugs fixed in this release |
| Benchmark + scientific | 19 | G9-G14: the redesigned benchmark falsifies assumptions |
| Gate + pipeline | 17 | G14-G25: comparative gates, real pipeline, atomic promotion |
| Existing v0.3.10 gates | 85 | G2-G16 from the prior release, still passing after clean break |
| Existing v0.3.9 and earlier | 556 | All prior tests, updated for version consistency |
| Skipped | 1 | Optional scikit-learn dependency (XOR linear separability check) |

## 2. Tests copied from prior artifacts

**None.** Every test in this report was executed in the current environment on 2026-07-28. No stored historical pytest output is presented as current execution.

## 3. Skipped optional dependency tests

| Test | Reason |
|------|--------|
| `test_v0310_1_benchmark.py::TestEnvironmentsAreWellFormed::test_xor_is_nonlinear` | `scikit-learn` not installed; the XOR linear-separability check requires `LogisticRegression` from sklearn. The test is skipped, not failed. The XOR environment itself is exercised by G13 (MLP beats logistic on XOR). |

## 4. Environment-related failures

**None.** All 737 collected tests pass. The 1 skip is an optional-dependency skip, not an environment failure.

## 5. Algorithmic failures

**None in the test suite.** The release gates (Section 47) intentionally include tests that verify the system can *falsify* its own assumptions:

- **G11 (centroid fails on multimodal):** the centroid policy achieves 0.318 regret on the multimodal environment (vs 0.201 on linear). This is an *expected* algorithmic failure — the test confirms the system can detect when centroid geometry is insufficient.
- **G13 (MLP beats logistic on XOR):** logistic achieves 0.299 regret on XOR vs MLP's 0.083. This is an *expected* algorithmic failure of the linear router — the test confirms the system can detect when nonlinear routing is needed.

These are not bugs; they are the scientific falsification capabilities required by Section 51.

## 6. Real-model smoke test (G25)

**Command:** `python scripts/smoke_real_model.py`

**Result:** PASSED

```
Device: mps
Model: Qwen/Qwen2.5-0.5B-Instruct (24 layers, dim 896)
Layer used: 10

[1/4] Captured 20 hidden states, shape (20, 896)
[2/4] Trained centroid policy (vector norm 2.4653, n_symbolic=10, n_llm=10)
[3/4] 25 intervention results recorded (5 tasks x 5 alphas)
[4/4] Aggregate +v / 0 / -v directional effect:
  E[P(S|+v)] = 0.7383 > E[P(S|0)] = 0.6427 > E[P(S|-v)] = 0.5350
  +v > 0: 5/5, -v < 0: 5/5 (perfect directional consistency)

G25 SMOKE TEST PASSED
```

**Evidence level:** `REAL_MODEL_DEV` (synthetic task labels, real Qwen hidden states, real residual-stream intervention).

## 7. Release gates (G1-G25)

**Command:** `python scripts/generate_v0310_1_gates.py`

**Result:** ALL 25 GATES PASSED

See `release_gates.json` for per-gate measurements.

## 8. Warnings

| Warning | Source | Impact |
|---------|--------|--------|
| `DeprecationWarning: builtin type SwigPyPacked has no __module__` | macOS Python 3.12 | None — cosmetic |
| `DeprecationWarning: builtin type SwigPyObject has no __module__` | macOS Python 3.12 | None — cosmetic |
| `DeprecationWarning: BPE.__init__ will not create from files anymore` | transformers GPT-2 tokenizer | None — cosmetic, from a test that loads GPT-2 |
| `torch_dtype is deprecated` | transformers 5.13 | None — cosmetic |

## 9. Honesty statement

This report describes only tests actually executed on 2026-07-28 in the environment listed above. No prior pytest output is repurposed as current. The 1 skipped test is labeled with its reason. The 2 "algorithmic failures" (G11, G13) are *intentional* falsification tests, not bugs — they verify the system can detect when its methods fail.
