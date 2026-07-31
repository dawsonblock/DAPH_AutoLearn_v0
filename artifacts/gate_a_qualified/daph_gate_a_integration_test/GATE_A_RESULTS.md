# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-07-30T19:47:14
**Criteria hash:** `1e95de8b3d12642c901b7a9ae2c90d5fc2533a24a3fbc352083ceb66b282785a`
**Overall status:** NOT_EVALUABLE

## Final Verdict: NOT_EVALUABLE

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| precondition_crossover | N/A | N/A | N/A | NOT_EVALUABLE: no backend crossover: oracle routes 100.0% symbolic, 0.0% LLM |
| minimum_point_gain_vs_p0 | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |
| require_lcb_vs_p0_above | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |
| require_lcb_vs_sham_above | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |
| minimum_oracle_gap_capture | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |
| minimum_positive_group_fraction | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |
| maximum_worst_subtype_regression | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |
| maximum_final_access_count | N/A | N/A | N/A | NOT_EVALUABLE: precondition failure |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.5054853445171684
- 95% CI (group_weighted): [0.40548534451716844, 0.6054853445171684]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 1.0
- P1 LLM fraction: 0.0
- P1 abstain fraction: 0.0
- Oracle symbolic fraction: 1.0
- Oracle LLM fraction: 0.0
- P1–oracle action agreement: 1.0
- P1–always-symbolic agreement: 1.0

## Sham Control

- P1 utility: 0.5054853445171684
- Mean sham utility: 0.5054853445171684
- P1 minus sham (mean): 0.0
- P1 minus sham 95% CI: [0.5054853445171684, 0.5054853445171684]
- P1 percentile vs sham: 0.0%
- Sham seeds: 3
- Training spec hash: `ca3a676bc909e65c87c9ff26ebc5abe454057ab8c5e5ac6b2105c635147d1342`

## Dataset

- N groups: 63
- N tasks: 252
- Crossover subtypes: 6
- Decisive fraction: 0.4797619047619048

## Baselines


## Final Access

- Access count: 1
- Source hash: `f76dbaf80b9b1eca20dd47286395e680237ae893a41f674ae7037154723cec38`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.