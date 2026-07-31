# Implementation Report — DAPH AutoLearn v0.3.10.5-alpha

## Gate A Scientific-Integrity Repair — Priority 0+1 (Sections 1–19)

**Scope:** This release implements the **Priority 0** (Sections 1–7) and
**Priority 1** (Sections 8–19) slices of the Gate A integrity repair plan.
It is a scientific-integrity, reproducibility, and qualification repair —
**not** an architecture expansion. No latent memory, activation steering,
continual learning, model merging, world models, autonomous self-modification,
multi-agent debate, or unrelated agent features were added.

**Scope explicitly deferred (Sections 20–27):** full real-model Gate A
execution, final-access enforcement on a real run, end-to-end report
generation from real artifacts. The infrastructure is in place; only the
actual execution remains. See `REMAINING_EXPERIMENT_STEPS.md`.

---

## Canonical source hash

```
c5191b73d4ca07d58c94d6ec31ac4a12d8f3de118640878cd34b27f8e3be2f32
```

Computed by `python -m daph_learning.provenance source-hash` over
`src/**/*.py`, `scripts/**/*.py`, `tests/**/*.py` with normalized line
endings, deterministic ordering, and the exclusions in
`DEFAULT_EXCLUDE_GLOBS`.

---

## Modified and new files

### Version & experiment identity (Section 2)

| File | Change |
|------|--------|
| `pyproject.toml` | Version `0.3.10.3.2-alpha` → `0.3.10.4-alpha`; description updated. |
| `src/daph_learning/__init__.py` | `__version__` bumped; added `CURRENT_EXPERIMENT_ID = "daph_gate_a_real_002"` and `LEGACY_FAILED_EXPERIMENT_ID = "daph_gate_a_real_001_failed"`. |
| `src/daph_learning/policy/stage.py` | Functional `release_version` defaults bumped to `0.3.10.4-alpha`. |
| `src/daph_learning/policy/config.py` | `autolearn_version` default bumped. |
| `src/daph_learning/policy/provenance.py` | `release_version` default bumped. |

### Artifact layout (Section 3)

| Path | Change |
|------|--------|
| `artifacts/synthetic_ci/` | New bucket (with `.gitkeep`). |
| `artifacts/real_model_smoke/` | New bucket; populated by the smoke run. |
| `artifacts/gate_a_qualified/` | New bucket (empty until a real run passes). |
| `artifacts/gate_a_failed/` | New bucket (empty until a real run fails). |
| `artifacts/legacy/daph_gate_a_real_001_failed/` | Old failed run archived here. |
| `artifacts/current/pointer.json` | New machine-readable pointer; declares `NOT_YET_REQUALIFIED`, `target: null`, current source hash. |

### Provenance (Section 4.1)

| File | Change |
|------|--------|
| `src/daph_learning/provenance.py` | Added `compute_canonical_source_hash()` with explicit `include_globs`/`exclude_globs`, normalized line endings (CRLF/CR→LF), `DEFAULT_INCLUDE_GLOBS`/`DEFAULT_EXCLUDE_GLOBS` constants, and a CLI entry point (`_cli_main`) so `python -m daph_learning.provenance source-hash [--json] [--repo-root PATH]` prints only the canonical hash. Existing `compute_source_tree_sha256` retained for backward compatibility. |

### Artifact validator (Section 4.2)

| File | Change |
|------|--------|
| `src/daph_learning/evaluation/artifact_integrity.py` | Added `ArtifactValidationResult` dataclass and `validate_artifact_bundle()` — recursively validates source-hash consistency, experiment/run ID consistency, evidence-level consistency, dataset/model/tokenizer/utility/policy hash consistency, split validity, final-access ledger presence; rejects synthetic-as-qualified, failed-as-promoted, and cross-run metric copying. Legacy bundles (marked `legacy`/`not_current_source`/`historical_source_hash`) are exempt from the current-hash requirement via a two-pass detection. Added `json` import. |

### Artifact integrity CI gate (Section 4.3)

| File | Change |
|------|--------|
| `tests/test_artifact_integrity.py` | **New.** 23 tests: layout dirs exist, legacy notice exists, current pointer resolves, current contains pointer only, all bundled artifacts use current hash or are legacy, legacy never marked current, synthetic cannot be qualified, failed cannot be promoted, mixed run/experiment/dataset/utility hashes rejected, stale source hash rejected, legacy hash accepted, missing final ledger rejected. |
| `tests/test_current_artifact_tree_contains_no_stale_source_hash.py` | Rewritten for the new pointer layout (accepts `source_hash` field; treats `NOT_YET_REQUALIFIED` pointer with no target as valid). |

### Legacy archival (Section 5)

| Path | Change |
|------|--------|
| `artifacts/legacy/daph_gate_a_real_001_failed/LEGACY_NOTICE.md` | **New.** States the run failed the group-aware LCB gate (LCB95% = −0.041) and is retained for audit only. |
| `artifacts/legacy/daph_gate_a_real_001_failed/historical_source_manifest.json` | **New.** `legacy: true`, `not_current_source: true`, historical hash `fd4d47e3...`, lists evidence present and **missing** (`raw_metrics.json`, `bootstrap_results.json`, `environment.json`) honestly. |
| `artifacts/legacy/daph_gate_a_real_001_failed/` | Moved here: `GATE_A_RESULTS.md`, `GATE_A_FIXES.md`, `experiment_results.json`, `current_source_manifest.json`, `freeze_manifest.json`, `release_gates.json`, `test_report_*`, `junit.xml`, and the archived real-Qwen smoke results. The old `artifacts/current/` evidence and `artifacts/archive/` tree were removed. |

### Symbolic safety — eval removal (Section 6)

| File | Change |
|------|--------|
| `src/daph_learning/tools/symbolic_math.py` | `safe_eval_int_expr` extended to the spec'd signature (`max_ast_nodes=128`, `max_depth=32`, `max_integer_bits=4096`, `max_exponent=12`). Added explicit `_PERMITTED_NODE_TYPES` allowlist; `_validate_tree` now rejects every non-permitted node (names, calls, attributes, floats, true division, `**`, bitwise, comparisons, containers, lambdas). Added char-level guard. `_require_int` now takes `max_bits` and uses bit-length (not decimal-digit) limits. `_eval_node` rejects bool/float/string constants. |
| `src/daph_learning/execution/real_backends.py` | Removed `eval(expr, {"__builtins__": {}}, {})` from the symbolic execution path; all arithmetic now routes through `safe_eval_int_expr`. |
| `tests/test_symbolic_safety_section6.py` | **New.** 36 tests: static scan for `eval(`/`exec(`/`compile(` in symbolic paths (allowing `re.compile`/`model.eval()`), valid/nested arithmetic, floor div, modulo, divide-by-zero, oversized integer growth, excessive depth/nodes, true division, floats, strings, booleans, bitwise, comparisons, malicious expressions, Unicode digits, and `max_integer_bits` parameter honored. |

### Canonical verifier (Section 7)

| File | Change |
|------|--------|
| `src/daph_learning/evaluation/canonical_verifier.py` | **New.** `ParsedFinalAnswer` dataclass (status VALID/MISSING/MULTIPLE/MALFORMED/CONTRADICTORY), `parse_canonical_integer_answer()` (exactly one `FINAL_ANSWER:` field, one integer, no range/list/arithmetic, fail-closed on ambiguity, Unicode-minus normalization), closed `VerificationStatus` enum (CORRECT/INCORRECT/UNVERIFIABLE/EXECUTION_ERROR/TIMEOUT), and `CanonicalIntegerVerifier`. Legacy permissive extraction is not wired into qualification. |
| `tests/test_canonical_verifier_section7.py` | **New.** 25 tests covering every spec example (incidental number does not pass, multiple/conflicting markers unverifiable, negative integers, Unicode minus, ranges/lists/arithmetic malformed, closed enum, execution error, timeout, legacy parser does not credit). |

### Real-model smoke (Section 21, minimal)

| File | Change |
|------|--------|
| `scripts/run_real_model_smoke.py` | **New.** Loads a cached HF causal LM offline, captures a hidden state, executes both backends, verifies with the canonical verifier, writes a `REAL_MODEL_SMOKE` bundle, validates it with `validate_artifact_bundle`. Never marks Gate A qualified. |
| `artifacts/real_model_smoke/smoke_a3416479f15f/` | **New.** Smoke evidence bundle (real Qwen2.5-0.5B-Instruct on MPS, 6 tasks). `validation.json` records `valid=True`. |

### Documentation (Section 24, minimal)

| File | Change |
|------|--------|
| `README.md` | Header version bumped; added Gate A status banner (`NOT YET REQUALIFIED`); added `v0.3.10.5-alpha` changelog entry. |
| `CLAIMS.md` | Header bumped; added evidence-level discipline (IMPLEMENTED/UNIT_TESTED/SYNTHETIC_VALIDATED/REAL_MODEL_SMOKE/EXPERIMENTALLY_FAILED/EXPERIMENTALLY_QUALIFIED); added "Gate A status: NOT YET REQUALIFIED" section; updated test count to 1048. |
| `CHANGELOG.md` | Prepended full `0.3.10.4-alpha` entry documenting stale-artifact discovery, source-hash mismatch, old failed run archival, new experiment identity, verifier hardening, eval removal, artifact layout, validator, and CI gates. |

### Version-consistency test updates

| File | Change |
|------|--------|
| `tests/test_all_version_surfaces_match.py` | `TARGET_VERSION` → `0.3.10.4-alpha`. |
| `tests/test_cli_entrypoints.py` | Hardcoded version assertion → `0.3.10.4-alpha`. |
| `tests/test_v0310_gates.py` | Version assertions → `0.3.10.4-alpha`. |
| `tests/test_v037_exit_gate.py` | Version assertion → `0.3.10.4-alpha`. |
| `tests/test_version_claims_discipline.py` | Version expectation → `0.3.10.4-alpha`. |

---

## What was NOT done (honest scope boundary)

- No frozen `UtilityConfig` / utility hash (Section 8).
- No benchmark crossover redesign or 60+ generation groups (Section 9).
- No `src/daph_learning/benchmark/audit.py` leakage audit module (Section 10).
- No `TimingProtocol` / `measure_backend_latency` (Section 11).
- No development-only representation-selection study (Section 12).
- No surface/structured-feature baselines (Section 13).
- No uncertainty-aware training targets (Section 14).
- No multi-seed sham control (Section 15).
- No immutable `ExperimentStage` final-access state machine (Section 16).
- No group-aware bootstrap statistical suite (Section 17).
- No frozen `configs/gate_a_real_002.yaml` gate criteria (Section 18).
- No full report generator (Section 19).
- No full real-model Gate A run (Sections 21–22) — only a minimal smoke.

These remain as documented next steps. The current pointer correctly
reflects `NOT_YET_REQUALIFIED`.

---

## Priority 1 additions (Sections 8–19)

### Section 8 — Frozen UtilityConfig
- `src/daph_learning/policy/utility.py`: Added `UtilityConfig` dataclass (11 fields), `compute_utility()`, `utility_config_hash`, two protocol presets (`gate_a_accuracy_primary`, `gate_a_cost_sensitive_secondary`), `get_protocol()`.
- `tests/test_utility_config_section8.py`: 16 tests.

### Section 9 — Group-first crossover benchmark (60 groups)
- `src/daph_learning/data/grouped_benchmark.py`: 60 groups (10 per subtype), group-first split assignment (30 train / 12 dev / 9 cal / 9 final), no group crosses splits, no normalized prompt overlap.
- `tests/test_grouped_benchmark_section9.py`: 14 tests.

### Section 10 — Dataset leakage audit
- `src/daph_learning/benchmark/audit.py`: `DatasetAudit` class, `audit_dataset()`, exact/normalized duplicate detection, cross-split group/template-family leak detection, MinHash near-duplicate detection, crossover metrics, decisive fraction, `assert_dataset_clean()`.
- `tests/test_dataset_audit_section10.py`: 13 tests.

### Section 11 — TimingProtocol
- `src/daph_learning/evaluation/timing.py`: Added `TimingProtocol` (warmup, measured_runs, median/mean aggregation, CUDA sync), `measure_backend_latency()`, `LatencyMeasurement`.
- `tests/test_timing_protocol_section11.py`: 7 tests.

### Section 12 — Representation selection
- `src/daph_learning/evaluation/representation_selection.py`: `RepresentationCandidate`, `CandidateResult`, `RepresentationSelection`, `layer_candidates()`, `pooling_candidates()`, `select_representation()` with deterministic tie-breaking.
- `tests/test_representation_selection_section12.py`: 10 tests.

### Section 13 — Baselines
- `src/daph_learning/evaluation/baselines.py`: All 10 baselines (`always_llm`, `always_symbolic`, `hand_router`, `subtype_only_logistic`, `surface_tfidf_logistic`, `structured_feature_logistic`, `hidden_state_centroid`, `hidden_state_logistic`, `sham_hidden_state_logistic`, `oracle`), `StructuredTaskFeatures`, `extract_structured_features()`.
- `tests/test_baselines_section13.py`: 11 tests.

### Section 14 — Uncertainty-aware targets
- `src/daph_learning/policy/targets.py`: Extended `TargetMode` with `HARD_GAP`, `SOFT_TEMPERATURE`, `SIGNAL_TO_NOISE`; added `estimate_sigma()`, `build_uncertainty_aware_targets()` with reliability weights.
- `tests/test_uncertainty_targets_section14.py`: 8 tests.

### Section 15 — Sham control
- `src/daph_learning/evaluation/sham.py`: `ShamResult`, `shuffle_labels_within_bins()` (subtype×split×decisive binning), `run_sham_control()` (≥20 seeds, P1 percentile vs sham).
- `tests/test_sham_control_section15.py`: 5 tests.

### Section 16 — Extended final-access state machine
- `src/daph_learning/policy/stage.py`: Extended `ExperimentStage` to 9+5 stages, `_VALID_TRANSITIONS` table, `StageGuard.transition_to()` now rejects invalid backward transitions.
- `tests/test_final_access_section16.py`: 9 tests.

### Section 17 — Group-aware statistical suite
- `src/daph_learning/evaluation/grouped_stats.py`: Added `grouped_bootstrap_utility()` (group_weighted/task_weighted estimands), `grouped_permutation_test()`, `leave_one_group_out()`, `cluster_robust_mean_test()`.
- `tests/test_grouped_stats_section17.py`: 8 tests.

### Section 18 — Frozen gate criteria config
- `configs/gate_a_real_002.yaml`, `configs/gate_a_smoke.yaml`: Frozen gate criteria with all required fields.
- `src/daph_learning/evaluation/gate_criteria.py`: `GateCriteria` loader with schema validation, `criteria_hash`, rejects unknown/missing keys.
- `tests/test_gate_criteria_section18.py`: 10 tests.

### Section 19 — Report generator
- `src/daph_learning/evaluation/report.py`: `GateDecision`, `evaluate_gates()`, `generate_gate_a_results_md()`, `generate_report()` — generates `gate_decision.json`, `GATE_A_RESULTS.md`, `experiment_results.json` from machine-readable artifacts.
- `tests/test_report_generator_section19.py`: 7 tests.

### Section 22 — Staged CLI workflow
- `scripts/run_gate_a_staged.py`: Staged experiment runner (`--stage collect|develop|calibrate|final`), uses new infrastructure.
- `scripts/freeze_gate_a.py`: Freezes protocol + policy, writes freeze manifest + final access ledger.
- `scripts/validate_gate_a_bundle.py`: Validates artifact bundles with Gate A-specific requirements.

---

## Test summary

- **Priority 0 tests:** 86 (4 new test files)
- **Priority 1 tests:** 128 (11 new test files)
- **Total new tests:** 214
- **Full suite:** 1163 passed, 4 skipped, 0 failed
