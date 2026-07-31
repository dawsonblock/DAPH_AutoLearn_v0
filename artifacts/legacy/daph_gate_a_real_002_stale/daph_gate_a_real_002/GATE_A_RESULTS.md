# Gate A Results — daph_gate_a_real_002

**Generated:** 2026-07-30T18:49:34
**Criteria hash:** `8d1ce37900fdfb4852454b36cc3efb2b51c6d4b6f1e8d02a6e983b315e66227c`

## Final Verdict: FAIL

## Gate-by-Gate Verdicts

| Gate | Actual | Threshold | Direction | Passed |
|------|--------|-----------|-----------|--------|
| minimum_point_gain_vs_p0 | 0.4321 | 0.0200 | above | YES |
| require_lcb_vs_p0_above | 0.3321 | 0.0000 | above | YES |
| require_lcb_vs_sham_above | 0.4321 | 0.0000 | above | YES |
| minimum_oracle_gap_capture | 0.5000 | 0.5000 | above | NO |
| minimum_positive_group_fraction | 1.0000 | 0.6000 | above | YES |
| maximum_worst_subtype_regression | 0.0000 | 0.0300 | at_most | YES |
| maximum_final_access_count | 1.0000 | 1.0000 | at_most | YES |

## Primary Endpoint

- Estimand: group_weighted
- Point estimate: 0.43214445208851904
- 95% CI (group_weighted): [0.33214445208851906, 0.532144452088519]
- CI label: group_weighted
- Utility protocol: gate_a_accuracy_primary

## Sham Control

- P1 utility: 0.43214445208851904
- Mean sham utility: 0.432144452088519
- P1 minus sham (mean): 5.551115123125783e-17
- P1 minus sham 95% CI: [0.43214445208851904, 0.43214445208851904]
- P1 percentile vs sham: 0.0%
- Sham seeds: 20
- Training spec hash: `04452cf27deffa0914f48799f0947a82a1ac8566f994589b3ebf05fa0dc56b29`

## Dataset

- N groups: 9
- N tasks: 72
- Crossover subtypes: 5
- Decisive fraction: 0.8208333333333333

## Baselines


## Final Access

- Access count: 1
- Source hash: `65864625c49252717cbd99cf07b39855e3e132ad8a231062e51459589cc970ff`

---
This report was generated from machine-readable artifacts. No numbers were manually typed.