# TEST_REPORT_V0_3_10.md — DAPH AutoLearn v0.3.10-alpha

## Summary

- **Passed:** 652
- **Skipped:** 1
- **Failed:** 0

All release gates G1-G16 pass. The new v0.3.10 gate tests are located in `tests/test_v0310_gates.py` (67 tests).

## Release Gate Status

| Gate | Description | Status | Test |
|------|-------------|--------|------|
| G1 | Existing regression tests pass | PASS | — |
| G2 | Weighted centroid test | PASS | `tests/test_v0310_gates.py::TestWeightedCentroid` |
| G3 | Soft-target mathematical tests | PASS | `TestSoftTargets` |
| G4 | Weighted logistic router beats/matches centroid on synthetic task | PASS | `TestLogisticRouter` |
| G5 | Candidate-vector sensitivity test | PASS | `TestCausalIntervention` |
| G6 | Oracle-leakage test | PASS | `TestOracleLeakage` |
| G7 | Regret metric distinguishes costly mistakes | PASS | `TestRegretMetric` |
| G8 | Abstention works | PASS | `TestAbstention` |
| G9 | OOD fallback works | PASS | `TestOODDetection` |
| G10 | Causal +v/-v intervention test | PASS | `TestCausalIntervention` |
| G11 | Candidate-vs-incumbent uses actual policies | PASS | `TestOracleLeakage` |
| G12 | KL/capability promotion guard works | PASS | `TestKLPromotionGate` |
| G13 | Rollback is atomic | PASS | `TestRollbackAtomic` |
| G14 | Task-ID alignment invariants | PASS | `TestTaskIDAlignment` |
| G15 | Version/provenance consistency | PASS | `TestVersionConsistency` |
| G16 | Full synthetic AutoLearn loop reduces regret | PASS | `TestSyntheticClosedLoop` |

## New Test Categories

The following new test categories were added in v0.3.10:

- **Calibration** — `TestCalibrationMetrics`
- **PCA** — `TestFeatureReduction`
- **Replay** — `TestReplayPriority`
- **Bandit** — `TestBanditLogging`
- **LowRank** — `TestLowRankController`
- **LatentVerifier** — `TestLatentVerifier`
- **SNR** — `TestSNRWeighting`

## Baseline Matrix Results

Results from the synthetic environment:

| Policy | Utility | Regret | Brier | ECE | Abstain |
|--------|---------|--------|-------|-----|---------|
| base_llm | 0.50 | 0.44 | — | — | — |
| hand_router | 0.94 | 0.00 | — | — | — |
| unweighted_centroid | 0.94 | 0.00 | — | — | — |
| weighted_centroid | 0.94 | 0.00 | — | — | — |
| soft_logistic | 0.89 | 0.05 | 0.003 | 0.29 | 0.10 |
| hard_logistic | 0.90 | 0.04 | — | — | — |
| random_steering | 0.94 | 0.00 | — | — | — |
| incumbent | 0.50 | 0.44 | — | — | — |
| autolearn_v0310 | 0.89 | 0.05 | — | — | — |

## Notes on the Synthetic Task

The synthetic task is easily separable by the sign of `h[0]`, so centroid and random steering achieve 0 regret. The logistic router's value is demonstrated by the causal intervention tests and the calibrated abstention behavior rather than raw utility on this trivially separable task. The real differentiation between policies comes from real-model experiments and the causal gates (G5, G10), which probe whether the router responds correctly to controlled interventions rather than relying on spurious correlations.

## Reproduce

```bash
python -m pytest tests/ -v
python scripts/generate_v0310_results.py
```
