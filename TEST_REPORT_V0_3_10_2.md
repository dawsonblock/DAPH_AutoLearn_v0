# TEST_REPORT v0.3.10.2-alpha

## Test Execution Summary

Date: 2026-07-28
Source tree SHA256: 98c2f13ee85fa165
All 25 release gates: PASS

## Test Suite Results

### v0.3.10.2 Real Gates (tests/test_v0310_2_real_gates.py)

39 tests, all passing:

- **G18 Real symbolic backend**: 3 tests — arithmetic execution,
  fail-closed on unsupported, no placeholder labels.
- **G20 Real verifier**: 6 tests — correct/incorrect arithmetic, no
  number, no expected field, never substring match, dispatch.
- **G21 Real counterfactual experience**: 1 test — build from real
  outcomes.
- **G22 Neutral KL gate**: 3 tests — Case A (reject), Case B (eligible),
  Case C (fail closed).
- **G24 Atomic promotion**: 2 tests — failed promotion leaves
  incumbent, successful promotion swaps.
- **G25 Smoke artifact validation**: 2 tests — required fields, evidence
  level.
- **G27 Evidence version match**: 2 tests — release gates structure,
  experiment results.
- **G19 Real LLM backend**: 1 integration test — generates text with
  Qwen2.5-0.5B-Instruct.
- **G14 Real intervention**: 1 integration test — changes hidden state
  with alpha!=0.
- **P15 No leakage**: 1 test — capture is pre-execution.
- **G26 Mixed tasks**: 4 tests — both families, symbolic fails on
  counting, symbolic succeeds on arithmetic, routing signal.
- **G27 Exact string verifier**: 6 tests — exact match, answer phrase,
  trailing punctuation, wrong answer, no expected, dispatch.
- **G28 CalibrationArtifact**: 2 tests — construction, frozen.
- **G29 Source tree hash**: 2 tests — deterministic, hex.
- **G30 Evidence labels**: 2 tests — refined levels exist, default
  label.
- **G31 Real experiment artifact**: 1 test — mixed-task signal,
  per-family, all baselines.

### v0.3.10.1 Benchmark Tests (tests/test_v0310_1_benchmark.py)

All tests passing, including:
- **G10 multi-seed**: 10 seeds, true superiority with CI.
- **G11 centroid failure**: multimodal degrades centroid.
- **G12 linear router**: logistic learns linear signal.
- **G13 MLP on XOR**: MLP beats logistic on nonlinear.
- **Random direction control**: random vectors don't trivially solve.
- **Environment well-formedness**: all 4 environments produce valid
  tasks.

### v0.3.10.1 Gate Tests (tests/test_v0310_1_gates.py, test_v0310_1_p0_gates.py)

All tests passing.

### Full Test Suite

```
python -m pytest tests/ -x --tb=short
```

All tests pass (including integration tests with Qwen2.5-0.5B-Instruct
on MPS).

## Release Gates (release_gates.json)

All 25 gates pass:

| Gate | Name | Evidence Level | Status |
|------|------|---------------|--------|
| G1 | soft/hard target modes | UNIT | PASS |
| G2 | weight modes | UNIT | PASS |
| G3 | policy selection | UNIT | PASS |
| G4 | task-ID alignment | UNIT | PASS |
| G5 | fail closed on missing utility | UNIT | PASS |
| G6 | negative weights rejected | UNIT | PASS |
| G7 | calibration math corrected | UNIT | PASS |
| G8 | dev-regret early stopping | UNIT | PASS |
| G9 | benchmark not trivially solvable | SYNTHETIC | PASS |
| G10 | weighted beats unweighted (multi-seed) | SYNTHETIC | PASS |
| G11 | centroid fails on multimodal | SYNTHETIC | PASS |
| G12 | logistic strong on linear | SYNTHETIC | PASS |
| G13 | MLP beats logistic on XOR | SYNTHETIC | PASS |
| G14 | comparative gate | SYNTHETIC | PASS |
| G15 | mechanical causal sanity | UNIT | PASS |
| G16 | real intervention pipeline | REAL_MODEL_LATENT | PASS |
| G17 | real train CLI | UNIT | PASS |
| G18 | real evaluate CLI | UNIT | PASS |
| G19 | real calibrate CLI | UNIT | PASS |
| G20 | OOD threshold calibrated | SYNTHETIC | PASS |
| G21 | candidate vs incumbent | UNIT | PASS |
| G22 | capability gate | UNIT | PASS |
| G23 | atomic rollback | UNIT | PASS |
| G24 | version/config provenance | UNIT | PASS |
| G25 | real-model smoke run | REAL_MODEL_FINAL | PASS |

## Real Experiment Results

See `artifacts/real_qwen_experiment_result.json` for full details.

Model: Qwen/Qwen2.5-0.5B-Instruct
Device: MPS
Layer: 10
Splits: train=60, dev=30, cal=30, final=60 (mixed arithmetic + counting)

### Key Findings

1. **Routing signal exists**: 10 tasks where symbolic is better, 8
   where LLM is better, 42 ties. ΔU ∈ [-1.0, 1.0].

2. **Candidate achieves zero regret**: The weighted centroid policy
   achieves 0.0 regret on the final test set, matching the hand-coded
   router and beating always-LLM (0.22).

3. **Steering has real effect**: The centroid vector has norm 2.87
   (non-zero because mixed tasks create a meaningful direction).
   Dose-response: E[P(S|+v)]=0.69, E[P(S|0)]=0.53, E[P(S|-v)]=0.37.

4. **Weighting helps**: Weighted centroid (0.0 regret) beats
   unweighted logistic (0.2 regret) on the final test set.

### Scientific Success Tiers

- Tier 2 (beats always-LLM): **YES**
- Tier 3 (beats hand router): NO (both 0.0 — hand router is perfect
  on this easy task set)
- Tier 4 (beats unweighted): **YES**
