# Experiment Results — daph_gate_a_real_005_requal

**Generated from:** `artifacts/current/pointer.json`
**Generated at:** 2026-08-01T01:35:48.809786+00:00
**Experiment ID:** daph_gate_a_real_005_requal
**Qualification status:** **PASS**
**Evidence level:** EXPERIMENTALLY_QUALIFIED
**Model:** Qwen/Qwen2.5-1.5B-Instruct (revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)

## Gate Decision

**Overall:** PASS

| Gate | Comparator | Threshold | Actual | Passed |
|------|-----------|-----------|--------|--------|
| minimum_point_gain_vs_p0 | gt | 0.0200 | 0.5372 | PASS |
| require_lcb_vs_p0_above | gt | 0.0000 | 0.4866 | PASS |
| require_lcb_vs_sham_above | gt | 0.0000 | 0.0863 | PASS |
| minimum_oracle_gap_capture | gte | 0.5000 | 0.9972 | PASS |
| minimum_positive_group_fraction | gt | 0.6000 | 0.9762 | PASS |
| maximum_worst_subtype_regression | lte | 0.0300 | 0.0000 | PASS |
| maximum_final_access_count | lte | 1.0000 | 1.0000 | PASS |

## Primary Endpoint (P1 − P0)

- **Estimand:** group_weighted
- **Point estimate:** 0.5372
- **95% CI:** [0.4866, 0.5863]
- **Bootstrap iterations:** 20000

## Sham Comparison (P1 − Sham)

- **P1 utility:** 0.8125
- **Mean sham utility:** 0.6597
- **P1 − sham mean:** 0.1528
- **P1 − sham 95% CI:** [0.0863, 0.3289]
- **P1 percentile vs sham:** 100.0%
- **Sham seeds:** 20

## Route Distribution

- P1 symbolic fraction: 70.1%
- P1 LLM fraction: 29.9%
- Oracle symbolic fraction: 53.9%
- Oracle LLM fraction: 11.2%
- P1-oracle action agreement: 64.9%

## Summary Metrics

- **Oracle gap capture:** 0.9972
- **P1 utility:** 0.8125
- **P0 utility:** 0.2753
- **Oracle utility:** 0.8140
- **Positive group fraction:** 97.6%
- **Worst subtype regression:** 0.0000

---

*This file is auto-generated from `artifacts/current/pointer.json`. Do not edit manually — run `python scripts/generate_experiment_results.py` to regenerate.*
