# Gate A Results — daph_gate_a_integration_test

**Generated:** 2026-08-02T16:27:49
**Criteria hash:** `1e95de8b3d12642c901b7a9ae2c90d5fc2533a24a3fbc352083ceb66b282785a`
**Overall status:** PASS

## Final Verdict: PASS

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Comparator | Passed |
|------|--------|-----------|------------|--------|
| minimum_point_gain_vs_p0 | 0.0893 | 0.0000 | gt | YES |
| require_lcb_vs_p0_above | 0.0536 | -1.0000 | gt | YES |
| require_lcb_vs_sham_above | 0.0863 | -1.0000 | gt | YES |
| minimum_oracle_gap_capture | 0.8333 | 0.0000 | gte | YES |
| minimum_positive_group_fraction | 0.3810 | 0.0000 | gt | YES |
| maximum_worst_subtype_regression | 0.0250 | 1.0000 | lte | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | lte | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.08928571428571429
- 95% CI (group_weighted): [0.05357142857142857, 0.125]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Route Distribution

- P1 symbolic fraction: 0.6577380952380952
- P1 LLM fraction: 0.34226190476190477
- P1 abstain fraction: N/A
- Oracle symbolic fraction: 0.2619047619047619
- Oracle LLM fraction: 0.10714285714285714
- P1–oracle action agreement: 0.35119047619047616
- P1–always-symbolic agreement: N/A

## Sham Control

- P1 utility: 0.7827380952380952
- Mean sham utility: 0.6180555555555555
- P1 minus sham (mean): 0.16468253968253965
- P1 minus sham 95% CI: [0.08630952380952384, 0.23809523809523803]
- P1 percentile vs sham: 100.0%
- Sham seeds: 3
- Training spec hash: `N/A`

## Dataset

- N groups: 84
- N tasks: 336
- Crossover subtypes: 0
- Decisive fraction: N/A

## Baselines

- always_llm: utility=0.53869
- always_symbolic: utility=0.693452
- best_fixed: utility=0.693452
- oracle: utility=0.800595
- hidden_plus_surface: utility=0.782738
- subtype_only: utility=0.699405

## Trained Baselines

- surface_only: utility=0.8005952380952381
- hidden_only: utility=0.6041666666666666
- tfidf: utility=0.7976190476190477
- charngram: utility=0.7946428571428571
- hidden_plus_tfidf: utility=0.7976190476190477
- heuristic: utility=0.7946428571428571
- shuffled_hidden: utility=0.7767857142857143
- orthogonal_rotation: utility=0.7827380952380952
- reduced_projection: utility=0.7976190476190477
- gaussian_noise: utility=0.7767857142857143
- coord_permutation: utility=0.7857142857142857
- hidden_norm_only: utility=0.6934523809523809

## Hidden-State Contribution Ablation

- P_COMBINED - P_SURFACE: estimate=-0.017857142857142856, LCB95=-0.03869047619047619, UCB95=-0.002976190476190476
- P_HIDDEN - P_TFIDF: estimate=-0.19345238095238096, LCB95=-0.23214285714285715, UCB95=-0.15476190476190477
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
- Source hash: `5681bca6b614e0324067ccca4f727b37f1a75eee3aa59e3dde8abf7446535051`
- Primary policy: hidden_plus_surface
- Primary comparator: best_fixed
- Best fixed policy: always_symbolic

---
This report was generated from machine-readable artifacts. No numbers were manually typed.