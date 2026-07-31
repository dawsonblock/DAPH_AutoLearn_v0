# Gate A Results — daph_gate_a_real_002

**Generated:** 2026-07-31T03:31:31
**Criteria hash:** `8d1ce37900fdfb4852454b36cc3efb2b51c6d4b6f1e8d02a6e983b315e66227c`
**Overall status:** PASS

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.4274 | 0.0200 | gt | YES |
| require_lcb_vs_p0_above | 0.3274 | 0.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.3527 | 0.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.6225 | 0.5000 | gte | YES |
| minimum_positive_group_fraction | 0.9206 | 0.6000 | gt | YES |
| maximum_worst_subtype_regression | 0.0000 | 0.0300 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.42744826671899666
- 95% CI (group_weighted): [0.32744826671899663, 0.5274482667189967]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.7003968253968254
- P1 LLM fraction: 0.2996031746031746
- P1 abstain fraction: 0.0
- Oracle symbolic fraction: 0.6845238095238095
- Oracle LLM fraction: 0.31547619047619047
- P1–oracle action agreement: 0.9484126984126984
- P1–always-symbolic agreement: 0.7003968253968254

## Sham Control

- P1 utility: 0.42744826671899666
- Mean sham utility: 0.35522598036497066
- P1 minus sham (mean): 0.072222286354026
- P1 minus sham 95% CI: [0.3527448622435649, 0.3586007976779611]
- P1 percentile vs sham: 100.0%
- Sham seeds: 20
- Training spec hash: `ce89851a73dd46102522fb5d16b62272eae9107a5107ad3f80019aa1e9d4c52e`

## Dataset

- N groups: 63
- N tasks: 504
- Crossover subtypes: 6
- Decisive fraction: 1.0

## Baselines


## Final Access

- Access count: 1
- Source hash: `bd073965319371bc76d65e851991dc1f332a79c23b9aa554b44f976723b57393`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.