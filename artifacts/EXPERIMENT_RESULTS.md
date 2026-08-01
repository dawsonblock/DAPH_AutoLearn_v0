# Experiment Results — daph_gate_a_real_006_repair

**Generated from:** `artifacts/current/pointer.json`
**Generated at:** 2026-08-01T02:37:31.928452+00:00
**Experiment ID:** daph_gate_a_real_006_repair
**Qualification status:** **FAIL**
**Evidence level:** EXPERIMENTALLY_FAILED
**Model:** Qwen/Qwen2.5-1.5B-Instruct (revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`)

## Gate Decision

**Overall:** FAIL

## Primary Endpoint (P1 − P0)

- **Estimand:** group_weighted
- **Point estimate:** 0.1116
- **95% CI:** [0.0863, 0.1384]
- **Bootstrap iterations:** 20000

## Sham Comparison (P1 − Sham)

- **P1 utility:** 0.8140
- **Mean sham utility:** 0.7024
- **P1 − sham mean:** 0.1116
- **P1 − sham 95% CI:** [0.0878, 0.1384]
- **P1 percentile vs sham:** 100.0%
- **Sham seeds:** 20

## Route Distribution

- P1 symbolic fraction: 70.2%
- P1 LLM fraction: 29.8%
- Oracle symbolic fraction: 53.9%
- Oracle LLM fraction: 11.2%
- P1-oracle action agreement: 65.0%

## Summary Metrics

- **Oracle gap capture:** 1.0000
- **P1 utility:** 0.8140
- **P0 utility:** 0.7024
- **Oracle utility:** 0.8140
- **Positive group fraction:** 57.1%
- **Worst subtype regression:** 0.0000

---

*This file is auto-generated from `artifacts/current/pointer.json`. Do not edit manually — run `python scripts/generate_experiment_results.py` to regenerate.*
