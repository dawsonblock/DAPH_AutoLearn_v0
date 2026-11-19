# Experiment Results — daph_gate_a_real_007_harder

**Generated from:** `artifacts/current/pointer.json`
**Generated at:** 2026-08-01T03:24:05.517116+00:00
**Experiment ID:** daph_gate_a_real_007_harder
**Qualification status:** **FAIL**
**Evidence level:** EXPERIMENTALLY_FAILED
**Model:** Qwen/Qwen2.5-7B-Instruct (revision: `a0937280e4cf806f80c7fef2b1adb7bc71aa306`)

## Gate Decision

**Overall:** FAIL

## Primary Endpoint (P1 − P0)

- **Estimand:** group_weighted
- **Point estimate:** 0.1295
- **95% CI:** [0.0744, 0.1830]
- **Bootstrap iterations:** 20000

## Sham Comparison (P1 − Sham)

- **P1 utility:** 0.6562
- **Mean sham utility:** 0.5275
- **P1 − sham mean:** 0.1287
- **P1 − sham 95% CI:** [0.0744, 0.1830]
- **P1 percentile vs sham:** 100.0%
- **Sham seeds:** 20

## Route Distribution

- P1 symbolic fraction: 42.0%
- P1 LLM fraction: 58.0%
- Oracle symbolic fraction: 31.8%
- Oracle LLM fraction: 17.1%
- P1-oracle action agreement: 44.8%

## Summary Metrics

- **Oracle gap capture:** 0.7565
- **P1 utility:** 0.6562
- **P0 utility:** 0.5268
- **Oracle utility:** 0.6979
- **Positive group fraction:** 50.0%
- **Worst subtype regression:** 0.0000

---

*This file is auto-generated from `artifacts/current/pointer.json`. Do not edit manually — run `python scripts/generate_experiment_results.py` to regenerate.*
