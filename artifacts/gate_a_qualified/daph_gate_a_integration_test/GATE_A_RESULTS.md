# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-07-31T14:06:31
**Criteria hash:** `1e95de8b3d12642c901b7a9ae2c90d5fc2533a24a3fbc352083ceb66b282785a`
**Overall status:** FAIL

## Final Verdict: FAIL

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.0000 | 0.0000 | gt | NO |
| require_lcb_vs_p0_above | 0.0000 | -1.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.0000 | -1.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.0000 | 0.0000 | gte | YES |
| minimum_positive_group_fraction | 0.0000 | 0.0000 | gt | NO |
| maximum_worst_subtype_regression | 0.0000 | 1.0000 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.0
- 95% CI (group_weighted): [0.0, 0.0]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.6349206349206349
- P1 LLM fraction: 0.36507936507936506
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.0
- Oracle LLM fraction: 0.0
- P1–oracle action agreement: 0.0
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.0
- Mean sham utility: 0.0
- P1 minus sham (mean): 0.0
- P1 minus sham 95% CI: [0.0, 0.0]
- P1 percentile vs sham: 0.0%
- Sham seeds: 3
- Training spec hash: `N/A`

## Dataset

- N groups: 63
- N tasks: 252
- Crossover subtypes: 0
- Decisive fraction: N/A

## Baselines


## Final Access

- Access count: 1
- Source hash: `ef0932c962eadbad5a239b059e7330cb0db9ea5e2f230b603ac28728a57c493f`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.