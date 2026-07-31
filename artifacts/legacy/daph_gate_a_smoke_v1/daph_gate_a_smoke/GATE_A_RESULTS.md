# Gate A Results — daph_gate_a_smoke

**Generated:** 2026-07-30T19:03:24
**Criteria hash:** `614ab07d44b7f8a377a0dfe2b09acf34a02c220496520582820f73e3325c1d7c`

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Direction | Passed |
|------|--------|-----------|-----------|--------|
| minimum_point_gain_vs_p0 | 0.5016 | 0.0000 | above | YES |
| require_lcb_vs_p0_above | 0.4016 | -1.0000 | above | YES |
| require_lcb_vs_sham_above | 0.5016 | -1.0000 | above | YES |
| minimum_oracle_gap_capture | 0.5000 | 0.0000 | above | YES |
| minimum_positive_group_fraction | 1.0000 | 0.0000 | above | YES |
| maximum_worst_subtype_regression | 0.0000 | 1.0000 | at_most | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | at_most | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.5015756143944131
- 95% CI (group_weighted): [0.40157561439441314, 0.6015756143944131]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Sham Control

- P1 utility: 0.5015756143944131
- Mean sham utility: 0.5015756143944131
- P1 minus sham (mean): 0.0
- P1 minus sham 95% CI: [0.5015756143944131, 0.5015756143944131]
- P1 percentile vs sham: 0.0%
- Sham seeds: 5
- Training spec hash: `8168e254942971cf3129da9d80f297e10bca74df61e191474701ada55bfa8443`

## Dataset

- N groups: 9
- N tasks: 36
- Crossover subtypes: 5
- Decisive fraction: 0.8333333333333334

## Baselines


## Final Access

- Access count: 1
- Source hash: `a17eae174b7b665eab487a3d3000e5cf56a1c29ed6e72c60a2e993556c78556e`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.