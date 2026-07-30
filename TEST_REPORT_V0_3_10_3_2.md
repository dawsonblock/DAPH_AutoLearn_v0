# TEST_REPORT_V0_3_10_3_2 — v0.3.10.3.2-alpha

## Structured Test Report

**Release:** v0.3.10.3.2-alpha
**Date:** 2026-07-29
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`
**Config SHA-256:** `efb98b1e12c931e52b58ac8c98676af5f65897464b9305c56fd87c8ecfddd72e`

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Collected tests | 963 |
| Passed | 959 |
| Failed | 0 |
| Skipped | 4 |
| Errors | 0 |
| xfail | 0 |
| xpass | 0 |
| Test framework | pytest 8.4.2 |
| Python version | 3.12.0 |
| Platform | darwin |
| Collection SHA-256 | `7797a792eb1911b5374de2c3afaa7d45d6cad0aa38d18ab03a175e29dd5eb540` |

---

## 2. Skipped Tests

The 4 skipped tests are real-model integration tests that require a
GPU and the Qwen2.5 model:

1. `test_real_model_integration.py::test_real_model_loads`
2. `test_real_model_integration.py::test_real_model_generates`
3. `test_real_model_qualification.py::test_real_model_routes`
4. `test_v0310_2_real_gates.py::TestRealLLMBackend::test_llm_generates_text`

These are G29/G30 gates (SKIP, not PASS). They must be executed on
the actual Qwen setup for external qualification.

---

## 3. Test Categories

### 3.1 Version Consistency (G01)
- `test_all_version_surfaces_match.py` (11 tests)
- `test_version_claims_discipline.py` (9 tests)

### 3.2 Canonical Source Hash (G02, G26)
- `test_canonical_source_hash.py` (9 tests)
- `test_current_artifact_tree_contains_no_stale_source_hash.py` (3 tests)

### 3.3 Benchmark Correctness (G06, G07, G09, G12)
- `test_benchmark_arithmetic_correctness.py` (9 tests)
- `test_no_duplicate_prompts_per_split.py` (4 tests)
- `test_template_disjointness.py` (4 tests)
- `test_within_subtype_crossover.py` (5 tests)
- `test_crossover_benchmark_v0310_3_1.py` (multiple tests)

### 3.4 CLI Completion (G14, G32)
- `test_cli_path_completion.py` (5 tests)
- `test_cli_entrypoints.py` (13 tests)

### 3.5 Steering Utility (G15-G18, G21-G22)
- `test_steering_utility_evidence.py` (8 tests)
- `test_steering_utility_v0310_3_1.py` (multiple tests)

### 3.6 Stage Machine + Freeze (G10-G11)
- `test_stage_freeze_final_guards.py` (12 tests)
- `test_stage_access_v0310_3_1.py` (10 tests)

### 3.7 Verifier + Ablations (G08, G20, G24)
- `test_verifier_modes_ablations.py` (10 tests)

### 3.8 Baseline Matrix (G13)
- `test_baseline_matrix.py` (5 tests)

### 3.9 32-Gate Registry (G48)
- `test_32_gate_execution.py` (9 tests)

---

## 4. Gate Execution Summary

| Gate Count | Status |
|------------|--------|
| 30 | PASS |
| 2 | SKIP (G29, G30 — real model) |
| 0 | FAIL |

**All non-model gates pass. The 2 SKIP gates require Qwen GPU
execution.**

---

## 5. JUnit XML

The structured JUnit XML report is available at:
`artifacts/current/junit.xml`

This was generated using `pytest --junitxml` for reliable parsing
(Section 5: no terminal string parsing).
