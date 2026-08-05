# Gate A Results — daph_gate_a_real_006_repair

**Generated:** 2026-08-01T02:37:15
**Criteria hash:** `91902f3ea1febe902d83f145c81c1069406993d934366511bf3fe52ca12792fd`
**Overall status:** FAIL

## Final Verdict: FAIL

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.1116 | 0.0200 | gt | YES |
| require_lcb_vs_p0_above | 0.0863 | 0.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.0878 | 0.0000 | gt | YES |
| minimum_oracle_gap_capture | 1.0000 | 0.5000 | gte | YES |
| minimum_positive_group_fraction | 0.5714 | 0.6000 | gt | NO |
| maximum_worst_subtype_regression | 0.0000 | 0.0300 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.11160714285714286
- 95% CI (group_weighted): [0.08630952380952381, 0.13839285714285715]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.7023809523809523
- P1 LLM fraction: 0.2976190476190476
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.5386904761904762
- Oracle LLM fraction: 0.11160714285714286
- P1–oracle action agreement: 0.6502976190476191
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.8139880952380952
- Mean sham utility: 0.7023809523809523
- P1 minus sham (mean): 0.1116071428571429
- P1 minus sham 95% CI: [0.08779761904761896, 0.1383928571428571]
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
- best_fixed: utility=0.702381
- oracle: utility=0.813988
- hidden_plus_surface: utility=0.813988
- subtype_only: utility=0.702381

## Trained Baselines

- surface_only: utility=0.8139880952380952
- hidden_only: utility=0.8139880952380952
- tfidf: utility=0.8139880952380952
- heuristic: utility=0.7901785714285714
- shuffled_hidden: utility=0.8139880952380952
- random_projection: utility=0.7023809523809523
- hidden_norm_only: utility=0.7023809523809523

## Hidden-State Contribution Ablation

- P_COMBINED - P_SURFACE: estimate=0.0, LCB95=0.0, UCB95=0.0
- P_HIDDEN - P_TFIDF: estimate=0.0, LCB95=0.0, UCB95=0.0
- Hidden-state claim supported: False
- Minimum effect threshold: 0.02

## Limitations

1. **Benchmark specificity:** This experiment evaluates routing on a
   structured-math benchmark. Generalization to other domains is not
   established.
2. **Model specificity:** Results are for the specified model revision
   only. Different models may produce different hidden-state representations.
3. **No OOD evaluation in this run:** OOD results are not reported in
   this bundle. The policy may be benchmark-specific.
4. **Hidden-state contribution:** The hidden-state ablation result
   (claim_supported field above) determines whether hidden states
   added measurable routing value beyond prompt-only baselines.
5. **Single final access:** The final stage was run exactly once.
   No hyperparameter tuning was performed after final access.
6. **Sham control:** Sham uses label permutation within bins, not
   feature permutation. Both variants should be tested in future work.

## Final Access

- Access count: 1
- Source hash: `141cbece50b986e9a945302fa99c0967f5c2a486e8fbae72167e792a7f19d8f5`
- Primary policy: hidden_plus_surface
- Primary comparator: best_fixed
- Best fixed policy: always_symbolic

---
This report was generated from machine-readable artifacts. No numbers were manually typed.