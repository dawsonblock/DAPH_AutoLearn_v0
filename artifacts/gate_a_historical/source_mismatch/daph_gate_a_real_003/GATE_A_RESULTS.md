# Gate A Results — daph_gate_a_real_003

**Generated:** 2026-07-31T22:21:27
**Criteria hash:** `23deadc0d96801e49284e86bdd3843476f81e521083fccff127132171c70a164`
**Overall status:** PASS

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.3735 | 0.0200 | gt | YES |
| require_lcb_vs_p0_above | 0.3125 | 0.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.2098 | 0.0000 | gt | YES |
| minimum_oracle_gap_capture | 1.0000 | 0.5000 | gte | YES |
| minimum_positive_group_fraction | 0.7857 | 0.6000 | gt | YES |
| maximum_worst_subtype_regression | 0.0000 | 0.0300 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.37351190476190477
- 95% CI (group_weighted): [0.3125, 0.43601190476190477]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.6830357142857143
- P1 LLM fraction: 0.3169642857142857
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.37351190476190477
- Oracle LLM fraction: 0.24404761904761904
- P1–oracle action agreement: 0.6175595238095238
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.9464285714285714
- Mean sham utility: 0.7023809523809523
- P1 minus sham (mean): 0.24404761904761907
- P1 minus sham 95% CI: [0.2098214285714286, 0.27827380952380953]
- P1 percentile vs sham: 100.0%
- Sham seeds: 20
- Training spec hash: `N/A`

## Dataset

- N groups: 84
- N tasks: 672
- Crossover subtypes: 2
- Decisive fraction: N/A

## Baselines


## Final Access

- Access count: 1
- Source hash: `51ad856027a33c21e4a477fbc2c7cf57ccd67c292602e42339f09c5fab933809`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.