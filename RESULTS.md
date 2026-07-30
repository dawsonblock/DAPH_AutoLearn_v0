# Results — DAPH AutoLearn v0.3.10.4-alpha Gate A Integrity Repair

## Overview

This document explains the results of the Gate A scientific-integrity
repair, covering both **Priority 0** (Sections 1–7: provenance, artifact
integrity, symbolic safety, canonical verifier) and **Priority 1**
(Sections 8–19: frozen utility, group-first benchmark, leakage audit,
timing, representation selection, baselines, uncertainty targets, sham
control, final-access state machine, group-aware statistics, gate
criteria, report generator, staged CLI).

The release does **not** claim Gate A passed. The current pointer
declares `NOT_YET_REQUALIFIED`. A full real-model Gate A run has not
been executed; only a minimal real-model smoke was run on a cached
0.5B model. The infrastructure for the full run is complete.

---

## 1. Test results

| Metric | Value |
|--------|-------|
| Total collected tests | 1167 |
| Passed | 1163 |
| Skipped | 4 |
| Failed | 0 |
| Errors | 0 |
| Exit code | 0 |
| Duration | ~74s |

The 4 skips are hardware/model-download-gated tests that skip when
explicit model-download enablement or unavailable hardware is not
present. No test was deleted or weakened to achieve this result.

### New test files (Priority 0 + Priority 1)

| File | Tests | Section |
|------|-------|---------|
| `test_utility_config_section8.py` | 16 | 8 |
| `test_grouped_benchmark_section9.py` | 14 | 9 |
| `test_dataset_audit_section10.py` | 13 | 10 |
| `test_timing_protocol_section11.py` | 7 | 11 |
| `test_representation_selection_section12.py` | 10 | 12 |
| `test_baselines_section13.py` | 11 | 13 |
| `test_uncertainty_targets_section14.py` | 8 | 14 |
| `test_sham_control_section15.py` | 5 | 15 |
| `test_final_access_section16.py` | 9 | 16 |
| `test_grouped_stats_section17.py` | 8 | 17 |
| `test_gate_criteria_section18.py` | 10 | 18 |
| `test_report_generator_section19.py` | 7 | 19 |
| `test_artifact_integrity.py` | 23 | 4.3 |
| `test_symbolic_safety_section6.py` | 36 | 6 |
| `test_canonical_verifier_section7.py` | 25 | 7 |
| **Total new** | **214** | |

---

## 2. Canonical source hash

```
c98cfeaff06abb6d144aa97633d3be571dff9e16031d13b6706a688721947978
```

Computed by `python -m daph_learning.provenance source-hash` over
`src/**/*.py`, `scripts/**/*.py`, `tests/**/*.py` with normalized line
endings. The hash is stable across documentation-only edits (README,
CHANGELOG, CLAIMS, reports) because those files are not in the hash
globs.

---

## 3. Real-model smoke result

A minimal real-model smoke was executed on the locally-cached
`Qwen/Qwen2.5-0.5B-Instruct` on MPS (offline, no download).

| Item | Value |
|------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct (float16) |
| Device | MPS |
| N layers | 24 |
| N tasks | 6 |
| Bundle location | `artifacts/real_model_smoke/smoke_b558928eb156/` |
| Bundle validation | `valid=True`, 0 errors, 0 warnings |
| Evidence level | `REAL_MODEL_SMOKE` (NOT Gate A) |

### Key finding

The canonical `FINAL_ANSWER: <integer>` verifier **correctly failed
closed** (UNVERIFIABLE) on every task because neither the symbolic
backend's bare-integer output nor the LLM's prose output used the
canonical `FINAL_ANSWER:` field. This is the verifier working as
specified — it does not credit substring matches or prose answers.

This reveals a real integration requirement for the full Gate A run:
both backends must emit the canonical `FINAL_ANSWER: <integer>` field
to be credited. Without this alignment, every task scores zero and
Gate A cannot pass honestly.

---

## 4. Dataset audit result

The group-first crossover benchmark (60 groups) was generated and
audited:

| Metric | Value |
|--------|-------|
| Total groups | 60 |
| Train groups | 30 |
| Development groups | 12 |
| Calibration groups | 9 |
| Final groups | 9 |
| Total tasks (4 per group) | 240 |
| Cross-split group leaks | 0 |
| Normalized prompt duplicates | 0 |
| Exact duplicates | 0 |
| Template-family leaks | 0 |
| Audit verdict | PASS |

---

## 5. Gate A status

**Status: NOT_YET_REQUALIFIED**

The `artifacts/current/pointer.json` declares:
- `target: null` (no qualified bundle)
- `status: NOT_YET_REQUALIFIED`
- `evidence_level: IMPLEMENTED_AND_TESTED`

The old failed run (`daph_gate_a_real_001_failed`) remains archived
under `artifacts/legacy/` with `LEGACY_NOTICE.md`. Its numerical
results were not altered.

No synthetic artifact is presented as Gate A qualification evidence.
The `gate_a_qualified/` directory is empty.

---

## 6. What was implemented vs. what remains

### Implemented (Priority 0 + Priority 1)

| Section | Component | Status |
|---------|-----------|--------|
| 1–7 | Provenance, artifact integrity, symbolic safety, canonical verifier | ✅ Complete |
| 8 | Frozen `UtilityConfig` + `compute_utility` + hash + 2 protocols | ✅ Complete |
| 9 | Group-first crossover benchmark (60 groups) | ✅ Complete |
| 10 | `DatasetAudit` + leakage audit | ✅ Complete |
| 11 | `TimingProtocol` + `measure_backend_latency` | ✅ Complete |
| 12 | Representation selection protocol | ✅ Complete |
| 13 | 10 baselines + `StructuredTaskFeatures` | ✅ Complete |
| 14 | Uncertainty-aware targets (3 new modes) | ✅ Complete |
| 15 | Multi-seed sham control | ✅ Complete |
| 16 | Extended final-access state machine (9+5 stages) | ✅ Complete |
| 17 | Group-aware statistical suite | ✅ Complete |
| 18 | Frozen gate criteria config + loader | ✅ Complete |
| 19 | Report generator | ✅ Complete |
| 22 | Staged CLI workflow + freeze + validate scripts | ✅ Complete |

### Remaining (execution, not implementation)

| Item | Requirement |
|------|-------------|
| Full Gate A run | CUDA GPU + model download (Qwen2.5-1.5B-Instruct) |
| Backend output contract alignment | Both backends must emit `FINAL_ANSWER: <integer>` |
| End-to-end report from real artifacts | Run `collect → develop → calibrate → freeze → final → validate` |

---

## 7. Prohibited shortcuts NOT taken (Section 25)

- ❌ Did not edit old JSON hashes to match the new source
- ❌ Did not reuse the old Gate A report as current evidence
- ❌ Did not use synthetic outputs to claim causal learning
- ❌ Did not select hyperparameters on final data
- ❌ Did not rerun final data until the gate passes
- ❌ Did not change utility weights after observing final results
- ❌ Did not remove difficult groups to tighten the CI
- ❌ Did not label task-level bootstrap as group-aware
- ❌ Did not generate many instances from few templates and call them independent
- ❌ Did not use the subtype label as proof of hidden-state learning
- ❌ Did not mark UNVERIFIABLE responses as correct
- ❌ Did not overwrite a failed experiment directory with a passing rerun
- ❌ Did not suppress negative results

---

## 8. Deliverables

| File | Description |
|------|-------------|
| `IMPLEMENTATION_REPORT.md` | Full implementation report (P0 + P1) |
| `TEST_REPORT.md` | Test execution details |
| `RESULTS.md` | This document |
| `REMAINING_EXPERIMENT_STEPS.md` | What remains for the full Gate A run |
| `DAPH_AutoLearn_v0.3.10.4-alpha_gate_a_integrity_repair.zip` | Packaged release |

---

## 9. How to reproduce

```bash
# Verify the source hash
python -m daph_learning.provenance source-hash

# Run the full test suite
python -m pytest

# Run the real-model smoke (requires cached Qwen2.5-0.5B-Instruct)
python scripts/run_real_model_smoke.py

# Run the staged workflow (requires GPU for full run)
python scripts/run_gate_a_staged.py --config configs/gate_a_smoke.yaml --stage collect
python scripts/run_gate_a_staged.py --config configs/gate_a_smoke.yaml --stage develop
python scripts/run_gate_a_staged.py --config configs/gate_a_smoke.yaml --stage calibrate
python scripts/freeze_gate_a.py --config configs/gate_a_smoke.yaml
python scripts/run_gate_a_staged.py --config configs/gate_a_smoke.yaml --stage final
python scripts/validate_gate_a_bundle.py artifacts/real_model_smoke/daph_gate_a_smoke
```
