# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-07-31T01:51:37
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
| maximum_worst_subtype_regression | 0.3716 | 1.0000 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.06267058702488193
- 95% CI (group_weighted): [-0.037329412975118076, 0.16267058702488194]
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

- P1 utility: 0.06267058702488193
- Mean sham utility: -0.02707717095712182
- P1 minus sham (mean): 0.08974775798200375
- P1 minus sham 95% CI: [-0.027731802631323174, -0.025998620559128202]
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
- Source hash: `73706bbc115291a1966b3caf02c0ebd98ea23b21671e0aa8f18cba2ecb062e23`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.