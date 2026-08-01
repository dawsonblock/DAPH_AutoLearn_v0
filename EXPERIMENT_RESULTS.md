# Experiment Results — daph_gate_a_real_003

**Generated from:** `artifacts/current/pointer.json`
**Generated at:** 2026-08-01T01:08:03.553605+00:00
**Experiment ID:** daph_gate_a_real_003
**Qualification status:** **PASS**
**Evidence level:** EXPERIMENTALLY_QUALIFIED
**Model:** Qwen/Qwen2.5-7B-Instruct (revision: `a09a35458c702b33eeacc393d103063234e8bc28`)

## Gate Decision

**Overall:** PASS

| Gate | Comparator | Threshold | Actual | Passed |
|------|-----------|-----------|--------|--------|
| minimum_point_gain_vs_p0 | gt | 0.0200 | 0.3735 | PASS |
| require_lcb_vs_p0_above | gt | 0.0000 | 0.3125 | PASS |
| require_lcb_vs_sham_above | gt | 0.0000 | 0.2098 | PASS |
| minimum_oracle_gap_capture | gte | 0.5000 | 1.0000 | PASS |
| minimum_positive_group_fraction | gt | 0.6000 | 0.7857 | PASS |
| maximum_worst_subtype_regression | lte | 0.0300 | 0.0000 | PASS |
| maximum_final_access_count | lte | 1.0000 | 1.0000 | PASS |

## Primary Endpoint (P1 − P0)

- **Estimand:** group_weighted
- **Point estimate:** 0.3735
- **95% CI:** [0.3125, 0.4360]
- **Bootstrap iterations:** 20000

## Sham Comparison (P1 − Sham)

- **P1 utility:** 0.9464
- **Mean sham utility:** 0.7024
- **P1 − sham mean:** 0.2440
- **P1 − sham 95% CI:** [0.2098, 0.2783]
- **P1 percentile vs sham:** 100.0%
- **Sham seeds:** 20

## Route Distribution

- P1 symbolic fraction: 68.3%
- P1 LLM fraction: 31.7%
- Oracle symbolic fraction: 37.4%
- Oracle LLM fraction: 24.4%
- P1-oracle action agreement: 61.8%

## Summary Metrics

- **Oracle gap capture:** 1.0000
- **P1 utility:** 0.9464
- **P0 utility:** 0.5729
- **Oracle utility:** 0.9464
- **Positive group fraction:** 78.6%
- **Worst subtype regression:** 0.0000

---

*This file is auto-generated from `artifacts/current/pointer.json`. Do not edit manually — run `python scripts/generate_experiment_results.py` to regenerate.*
