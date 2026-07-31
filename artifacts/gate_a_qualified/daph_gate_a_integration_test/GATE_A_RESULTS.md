# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-07-31T17:29:34
**Criteria hash:** `1e95de8b3d12642c901b7a9ae2c90d5fc2533a24a3fbc352083ceb66b282785a`
**Overall status:** PASS

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.2440 | 0.0000 | gt | YES |
| require_lcb_vs_p0_above | 0.1964 | -1.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.0536 | -1.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.9318 | 0.0000 | gte | YES |
| minimum_positive_group_fraction | 0.6548 | 0.0000 | gt | YES |
| maximum_worst_subtype_regression | 0.0000 | 1.0000 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.24404761904761904
- 95% CI (group_weighted): [0.19642857142857142, 0.29464285714285715]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.6577380952380952
- P1 LLM fraction: 0.34226190476190477
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.2619047619047619
- Oracle LLM fraction: 0.10714285714285714
- P1–oracle action agreement: 0.35119047619047616
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.7827380952380952
- Mean sham utility: 0.6934523809523809
- P1 minus sham (mean): 0.0892857142857143
- P1 minus sham 95% CI: [0.05357142857142849, 0.125]
- P1 percentile vs sham: 100.0%
- Sham seeds: 3
- Training spec hash: `N/A`

## Dataset

- N groups: 84
- N tasks: 336
- Crossover subtypes: 0
- Decisive fraction: N/A

## Baselines


## Final Access

- Access count: 1
- Source hash: `c14874d03187aba91b3fbd1fe818fcaffd34548d8d7f74c223b13168b0b5df5f`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.