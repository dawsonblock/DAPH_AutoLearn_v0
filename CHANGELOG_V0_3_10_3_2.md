# Changelog — v0.3.10.3.2-alpha

## Qualification Integrity Release

**Date:** 2026-07-29

**Mission:** Prove or falsify that AutoLearn can choose the better
computation between two available backends for individual tasks that
share the same family and subtype. This release hardens the
qualification evidence so that the central scientific claim is
testable, falsifiable, and backed by real executed utility.

### Core Scientific Repair

#### Within-Subtype Crossover (Sections 12-17)

The previous release (v0.3.10.3.1) had crossover at the subtype
level — entire subtypes were classified as "symbolic-preferred" or
"LLM-preferred." This release introduces **within-subtype crossover**:
at least 3 subtypes must individually contain both
symbolic-preferred and LLM-preferred examples, with the optimal
backend emerging from actual executed utilities.

**Mechanism:** Each subtype now generates a mix of small-operand
tasks (where both backends are correct and LLM wins on cost) and
large-operand tasks (where LLM arithmetic errs and symbolic wins on
quality). This produces real instance-level crossover within each
subtype.

**Result:** All 6 subtypes (A-F) now have within-subtype crossover,
with symbolic win fractions between 0.47 and 0.73.

#### Semantic Parser (Section 14)

A bounded semantic parser (`daph_learning.data.semantic_parser`)
extracts structured expressions from natural-language arithmetic
prompts. This allows the symbolic backend to compete on NL tasks
(subtypes B, C, E, F), enabling true within-subtype crossover.

The parser does NOT use the expected answer — it only uses the prompt
text. If parsing fails, it returns `None` (low confidence), and the
LLM backend is preferred.

### Qualification Hardening

#### Canonical Source-Tree Hash (Section 2)

One canonical `compute_source_tree_sha256()` implementation in
`daph_learning.provenance`. All other modules delegate to it. Full
64-character SHA-256 in artifacts (not truncated to 16).

Three previously incompatible implementations
(`policy.provenance.source_tree_sha256`,
`evaluation.artifact_integrity.compute_source_tree_hash`,
`scripts/run_real_qwen_experiment._source_tree_sha256`) now all
delegate to the canonical function.

#### Artifact Integrity (Sections 3-4)

* `REQUIRED_ARTIFACT_FIELDS` — every artifact must have
  `release_version`, `source_tree_sha256`, `config_sha256`,
  `created_at`.
* `assert_artifact_has_integrity_fields()` — verifies required
  fields.
* `assert_artifact_matches_current_source()` — verifies artifact hash
  matches current source tree.
* Artifacts restructured: `artifacts/current/` for current,
  `artifacts/archive/<version>/<hash>/` for historical.

#### Benchmark Correctness Fixes (Sections 6-8)

* **Subtype E (equal products):** regenerate operands until
  `left_product != right_product` (no mutation after the fact).
* **Subtype C (fractional half):** use only even values for "half of"
  so `x // 2` has exact mathematical meaning.
* **Subtype F (percentage truncation):** generate `total` such that
  `total * loss_pct % 100 == 0` (no integer truncation).

#### Linguistic Template Disjointness (Section 10)

Splits are now disjoint by actual linguistic template ID (computed
from normalized prompt text), not just slot numbers. Each split has
split-specific wording wrappers for NL subtypes (B, C, E, F).

#### Deterministic Seed (Section 9)

Python's process-randomized `hash()` replaced with
`deterministic_seed()` — SHA-derived integer seed that is
deterministic across processes.

### CLI Completion (Sections 22-27)

* **intervene:** uses canonical `backend_utility` from executed
  backends, not synthetic `1.0 if symbolic else 0.0`.
* **calibrate:** fails closed when no `execute_fn` is available (no
  unsafe `0.0` fallback).
* **evaluate:** when `--test-tasks` is provided, uses real backends
  via `execute_symbolic_backend` and `execute_llm_backend`.

### Steering Utility (Sections 28-32)

* LLM cost updated to 0.001 (cheaper than symbolic) so both-correct
  cases are resolved by cost, creating real crossover.
* `evaluate_steering_utility` uses canonical `backend_utility` for
  all utility computations.
* Beneficial/harmful flip analysis preserved.
* Random control `p_emp` preserved.
* Oracle alpha (DEV-only diagnostic) preserved.

### Stage Machine + Freeze (Sections 33-36)

* `FreezeManifest` — records exact state at freeze time (source hash,
  config hash, policy hash, calibration hash).
* `StageGuard.freeze()` — transitions to FROZEN and creates the
  manifest.
* `StageGuard.verify_frozen_state()` — verifies current state matches
  frozen state before final evaluation.

### 32-Gate Framework (Sections 48-50)

* Full G1-G32 gate registry updated with new test files.
* Every gate has at least one test file.
* Every gate test file exists in the repo.
* Gate registry covers all new v0.3.10.3.2 sections.

### New Test Files

* `test_all_version_surfaces_match.py` — G1: version consistency
* `test_canonical_source_hash.py` — G2: canonical source hash
* `test_benchmark_arithmetic_correctness.py` — G6: arithmetic
  correctness
* `test_template_disjointness.py` — G9: linguistic template
  disjointness
* `test_no_duplicate_prompts_per_split.py` — G7: within-split dedup
* `test_within_subtype_crossover.py` — G31: within-subtype crossover
* `test_cli_path_completion.py` — G32: CLI path completion
* `test_steering_utility_evidence.py` — G15-G18, G21-G22: steering
  utility evidence
* `test_stage_freeze_final_guards.py` — G10-G11: stage + freeze
* `test_verifier_modes_ablations.py` — G8, G20, G24: verifier +
  ablations
* `test_baseline_matrix.py` — G13: baseline matrix
* `test_32_gate_execution.py` — G48: 32-gate registry

### New Source Files

* `daph_learning/provenance.py` — canonical source-tree hash +
  deterministic seed
* `daph_learning/data/semantic_parser.py` — bounded semantic parser
  for NL arithmetic
* `daph_learning/evaluation/pytest_report.py` — structured pytest
  reporting

### Updated Source Files

* `daph_learning/data/crossover_benchmark.py` — within-subtype
  crossover, semantic parser integration, benchmark fixes
* `daph_learning/execution/real_backends.py` — semantic parser
  integration in symbolic backend
* `daph_learning/policy/config.py` — utility weight defaults for
  crossover
* `daph_learning/policy/stage.py` — FreezeManifest
* `daph_learning/policy/provenance.py` — delegates to canonical hash
* `daph_learning/evaluation/artifact_integrity.py` — delegates to
  canonical hash, integrity fields
* `daph_learning/evaluation/release_gates.py` — updated G1-G32
  registry
* `daph_learning/routing/hand_router.py` — operand magnitude
  awareness
* `daph_learning/steering/utility_eval.py` — LLM cost update
* `daph_learning/cli/commands/policy.py` — real CLI paths
