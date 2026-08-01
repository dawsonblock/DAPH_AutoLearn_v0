# Gate A Results — daph_gate_a_real_005_requal

**Generated:** 2026-08-01T01:33:22
**Criteria hash:** `9c76a54463c6e3b9b6fa1b7cc6463c4ff852d0cbedffc72a8954f53ff1cd42e3`
**Overall status:** PASS

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.5372 | 0.0200 | gt | YES |
| require_lcb_vs_p0_above | 0.4866 | 0.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.0863 | 0.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.9972 | 0.5000 | gte | YES |
| minimum_positive_group_fraction | 0.9762 | 0.6000 | gt | YES |
| maximum_worst_subtype_regression | 0.0000 | 0.0300 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.5372023809523809
- 95% CI (group_weighted): [0.48660714285714285, 0.5863095238095238]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.7008928571428571
- P1 LLM fraction: 0.29910714285714285
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.5386904761904762
- Oracle LLM fraction: 0.11160714285714286
- P1–oracle action agreement: 0.6488095238095238
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.8125
- Mean sham utility: 0.6597470238095238
- P1 minus sham (mean): 0.15275297619047612
- P1 minus sham 95% CI: [0.08630952380952384, 0.3288690476190476]
- P1 percentile vs sham: 100.0%
- Sham seeds: 20
- Training spec hash: `N/A`

## Dataset

- N groups: 84
- N tasks: 672
- Crossover subtypes: 1
- Decisive fraction: N/A

## Baselines

- always_llm: utility=0.275298
- always_symbolic: utility=0.702381
- oracle: utility=0.813988
- p1_policy: utility=0.8125
- subtype_majority: utility=0.702381

## Final Access

- Access count: 1
- Source hash: `8902e990c65dfc9acdaddc9dedede48c0726b0069ad90227b870b27ec761c2eb`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.