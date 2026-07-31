# Gate A Results — daph_gate_a_real_002

**Generated:** 2026-07-31T00:50:36
**Criteria hash:** `8d1ce37900fdfb4852454b36cc3efb2b51c6d4b6f1e8d02a6e983b315e66227c`
**Overall status:** FAIL

## Final Verdict: FAIL

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.2677 | 0.0200 | gt | YES |
| require_lcb_vs_p0_above | 0.1677 | 0.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.2467 | 0.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.4228 | 0.5000 | gte | NO |
| minimum_positive_group_fraction | 0.7778 | 0.6000 | gt | YES |
| maximum_worst_subtype_regression | 0.2308 | 0.0300 | lte | NO |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.26772378593779006
- 95% CI (group_weighted): [0.16772378593779005, 0.3677237859377901]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.5476190476190477
- P1 LLM fraction: 0.4523809523809524
- P1 abstain fraction: 0.0
- Oracle symbolic fraction: 0.6845238095238095
- Oracle LLM fraction: 0.31547619047619047
- P1–oracle action agreement: 0.8115079365079365
- P1–always-symbolic agreement: 0.5476190476190477

## Sham Control

- P1 utility: 0.26772378593779006
- Mean sham utility: 0.24668489277412148
- P1 minus sham (mean): 0.02103889316366858
- P1 minus sham 95% CI: [0.24668489277412145, 0.24668489277412145]
- P1 percentile vs sham: 100.0%
- Sham seeds: 20
- Training spec hash: `d4cf477cc2d038cc50872b689398392653290896d293569cd4fbcca133854af4`

## Dataset

- N groups: 63
- N tasks: 504
- Crossover subtypes: 6
- Decisive fraction: 0.9321428571428572

## Baselines


## Final Access

- Access count: 1
- Source hash: `38278f432dfb042733245876c98e2598ca7248695b54ad049b46f9c7b76be247`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.