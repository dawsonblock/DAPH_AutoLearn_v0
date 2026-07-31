# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-07-31T00:52:01
**Criteria hash:** `1e95de8b3d12642c901b7a9ae2c90d5fc2533a24a3fbc352083ceb66b282785a`
**Overall status:** FAIL

## Final Verdict: FAIL

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | -0.0256 | 0.0000 | gt | NO |
| require_lcb_vs_p0_above | -0.1256 | -1.0000 | gt | YES |
| require_lcb_vs_sham_above | -0.0681 | -1.0000 | gt | YES |
| minimum_oracle_gap_capture | -0.0724 | 0.0000 | gte | NO |
| minimum_positive_group_fraction | 0.5397 | 0.0000 | gt | YES |
| maximum_worst_subtype_regression | 0.3716 | 1.0000 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: -0.025635274777077464
- 95% CI (group_weighted): [-0.12563527477707748, 0.07436472522292253]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.5119047619047619
- P1 LLM fraction: 0.4880952380952381
- P1 abstain fraction: 0.0
- Oracle symbolic fraction: 0.6706349206349206
- Oracle LLM fraction: 0.32936507936507936
- P1–oracle action agreement: 0.48412698412698413
- P1–always-symbolic agreement: 0.5119047619047619

## Sham Control

- P1 utility: -0.025635274777077464
- Mean sham utility: -0.06811462698203485
- P1 minus sham (mean): 0.04247935220495739
- P1 minus sham 95% CI: [-0.06811462698203485, -0.06811462698203485]
- P1 percentile vs sham: 100.0%
- Sham seeds: 3
- Training spec hash: `821595ba5a15ed5841bb3a5065b8d069b9c20df61bc7b7ed6f0340aeacf798ec`

## Dataset

- N groups: 63
- N tasks: 252
- Crossover subtypes: 6
- Decisive fraction: 0.638095238095238

## Baselines


## Final Access

- Access count: 1
- Source hash: `38278f432dfb042733245876c98e2598ca7248695b54ad049b46f9c7b76be247`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.