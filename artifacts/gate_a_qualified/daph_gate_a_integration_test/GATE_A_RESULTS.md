# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-07-31T03:35:31
**Criteria hash:** `1e95de8b3d12642c901b7a9ae2c90d5fc2533a24a3fbc352083ceb66b282785a`
**Overall status:** PASS

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.0627 | 0.0000 | gt | YES |
| require_lcb_vs_p0_above | -0.0373 | -1.0000 | gt | YES |
| require_lcb_vs_sham_above | -0.0277 | -1.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.1769 | 0.0000 | gte | YES |
| minimum_positive_group_fraction | 0.5397 | 0.0000 | gt | YES |
| maximum_worst_subtype_regression | 0.0923 | 1.0000 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.06267058167710864
- 95% CI (group_weighted): [-0.03732941832289137, 0.16267058167710863]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.6349206349206349
- P1 LLM fraction: 0.36507936507936506
- P1 abstain fraction: 0.0
- Oracle symbolic fraction: 0.6706349206349206
- Oracle LLM fraction: 0.32936507936507936
- P1–oracle action agreement: 0.9563492063492064
- P1–always-symbolic agreement: 0.6349206349206349

## Sham Control

- P1 utility: 0.06267058167710864
- Mean sham utility: -0.027077170309603067
- P1 minus sham (mean): 0.0897477519867117
- P1 minus sham 95% CI: [-0.027731801982796324, -0.025998619911601167]
- P1 percentile vs sham: 100.0%
- Sham seeds: 3
- Training spec hash: `53daf959e10d2322597e3e966c06b3dfb25fdfa549cb32022a583e3119f9b49b`

## Dataset

- N groups: 63
- N tasks: 252
- Crossover subtypes: 6
- Decisive fraction: 0.638095238095238

## Baselines


## Final Access

- Access count: 1
- Source hash: `bd073965319371bc76d65e851991dc1f332a79c23b9aa554b44f976723b57393`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.