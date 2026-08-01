# Gate A Results — daph_gate_a_real_007_harder

**Generated:** 2026-08-01T03:23:48
**Criteria hash:** `6276a492b6e847b86405b88974b36795ba4f5069c215d11272e894b59cc780dd`
**Overall status:** FAIL

## Final Verdict: FAIL

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.1295 | 0.0200 | gt | YES |
| require_lcb_vs_p0_above | 0.0744 | 0.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.0744 | 0.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.7565 | 0.5000 | gte | YES |
| minimum_positive_group_fraction | 0.5000 | 0.6000 | gt | NO |
| maximum_worst_subtype_regression | 0.0000 | 0.0300 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.12946428571428573
- 95% CI (group_weighted): [0.0744047619047619, 0.18303571428571427]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.41964285714285715
- P1 LLM fraction: 0.5803571428571429
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.31845238095238093
- Oracle LLM fraction: 0.17113095238095238
- P1–oracle action agreement: 0.4479166666666667
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.65625
- Mean sham utility: 0.527529761904762
- P1 minus sham (mean): 0.12872023809523814
- P1 minus sham 95% CI: [0.07440476190476186, 0.1830357142857143]
- P1 percentile vs sham: 100.0%
- Sham seeds: 20
- Training spec hash: `N/A`

## Dataset

- N groups: 84
- N tasks: 672
- Crossover subtypes: 3
- Decisive fraction: N/A

## Baselines

- always_llm: utility=0.379464
- always_symbolic: utility=0.526786
- best_fixed: utility=0.526786
- oracle: utility=0.697917
- hidden_plus_surface: utility=0.65625
- subtype_only: utility=0.589286

## Trained Baselines

- surface_only: utility=0.43452380952380953
- hidden_only: utility=0.6264880952380952
- tfidf: utility=0.6934523809523809
- heuristic: utility=0.5267857142857143
- shuffled_hidden: utility=0.41964285714285715
- random_projection: utility=0.3794642857142857
- hidden_norm_only: utility=0.3794642857142857

## Hidden-State Contribution Ablation

- P_COMBINED - P_SURFACE: estimate=0.22172619047619047, LCB95=0.17261904761904762, UCB95=0.27232142857142855
- P_HIDDEN - P_TFIDF: estimate=-0.06696428571428571, LCB95=-0.09821428571428571, UCB95=-0.03869047619047619
- Hidden-state claim supported: True
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
- Source hash: `a48a3555177015102d82042cf0b16bd775733187af5483d2cd78a4c96a579c01`
- Primary policy: hidden_plus_surface
- Primary comparator: best_fixed
- Best fixed policy: always_symbolic

---
This report was generated from machine-readable artifacts. No numbers were manually typed.