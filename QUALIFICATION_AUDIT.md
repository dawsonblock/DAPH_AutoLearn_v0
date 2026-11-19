# QUALIFICATION_AUDIT — v0.3.10.3.2-alpha

## Qualification Integrity Audit

**Release:** v0.3.10.3.2-alpha
**Date:** 2026-07-29
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`
**Config SHA-256:** `efb98b1e12c931e52b58ac8c98676af5f65897464b9305c56fd87c8ecfddd72e`

---

## 1. Audit Summary

This release performs a focused qualification-hardening pass on the
v0.3.10.3.1-alpha source. The central scientific question is:

> Can the model's internal state predict which of two genuinely
> available computations will produce greater utility for this
> individual problem?

### Key Findings

| Finding | Status |
|---------|--------|
| Version unified to 0.3.10.3.2-alpha | PASS |
| One canonical source-tree hash (64-char SHA-256) | PASS |
| Current artifacts match current source tree | PASS |
| Benchmark arithmetic correctness (no truncation) | PASS |
| Within-subtype crossover (6/6 subtypes) | PASS |
| Linguistic template split disjointness | PASS |
| No within-split prompt duplicates | PASS |
| CLI paths completed (evaluate, calibrate, intervene) | PASS |
| Stage machine + freeze manifest | PASS |
| 32-gate registry executed (30 PASS, 2 SKIP) | PASS |
| Real model integration (G29/G30) | SKIP (requires GPU) |

### Test Results

- **959 tests passed**
- **4 tests skipped** (real model integration — requires Qwen GPU)
- **0 tests failed**
- **963 test node IDs collected**
- **Collection SHA-256:** `7797a792eb1911b5...`

---

## 2. Defects Identified and Fixed

### 2.1 Source-Tree Hash Ambiguity (Section 2)

**Defect:** Three incompatible source-tree hash implementations
existed:
- `policy.provenance.source_tree_sha256` (truncated to 16 chars)
- `evaluation.artifact_integrity.compute_source_tree_hash` (different
  file selection)
- `scripts.run_real_qwen_experiment._source_tree_sha256` (different
  exclusion rules)

**Fix:** Created `daph_learning.provenance.compute_source_tree_sha256()`
as the ONE canonical implementation. All other modules now delegate to
it. The canonical hash is the full 64-character SHA-256.

### 2.2 Benchmark Arithmetic Bugs (Sections 6-8)

**Defect E:** Subtype E could generate `left_product == right_product`
and then mutate the expected value without changing the prompt.

**Fix:** Regenerate operands until `left_product != right_product`.

**Defect C:** Subtype C "half of 75" computed `75 // 2 = 37` (wrong).

**Fix:** Generate only even values so `x // 2` is exact.

**Defect F:** Subtype F percentage problems used integer truncation
without stating so.

**Fix:** Generate `total` such that `total * p % 100 == 0`.

### 2.3 Subtype Classification Shortcut (Section 12)

**Defect:** The crossover benchmark was still subtype classification:
- A, D → symbolic
- B, C, E, F → LLM

**Fix:** Within-subtype crossover redesign. Each subtype now generates
a mix of small-operand tasks (LLM wins on cost) and large-operand
tasks (symbolic wins on quality). All 6 subtypes now have
`0.47 ≤ symbolic_win_fraction ≤ 0.73`.

### 2.4 CLI Path Incompleteness (Sections 22-27)

**Defect:**
- `intervene` used synthetic utility (`1.0 if symbolic else 0.0`)
- `calibrate` had unsafe `0.0` fallback when no `execute_fn`
- `evaluate` didn't use real backends for test tasks

**Fix:**
- `intervene` uses canonical `backend_utility` from executed backends
- `calibrate` fails closed when no `execute_fn` is available
- `evaluate` uses `execute_symbolic_backend` and `execute_llm_backend`

### 2.5 Stale Artifacts (Sections 3-4)

**Defect:** Root-level artifacts had source hash
`3a171769d64e68c1` (16-char, from v0.3.10.3-alpha), not matching the
current source tree.

**Fix:** All artifacts regenerated with the current 64-char source
hash. Previous artifacts moved to `artifacts/archive/`.

---

## 3. 32-Gate Execution Results

| Gate | Section | Description | Status |
|------|---------|-------------|--------|
| G01 | 1 | Version unified across all surfaces | PASS |
| G02 | 2 | Canonical source-tree hash (64-char SHA-256) | PASS |
| G03 | 3 | One canonical utility function everywhere | PASS |
| G04 | 4-5 | BackendOutcome distinguishes verified_correct from correct | PASS |
| G05 | 4-5 | Confidence != quality semantics | PASS |
| G06 | 6-8 | Benchmark arithmetic correctness (no truncation) | PASS |
| G07 | 11 | No within-split prompt duplicates | PASS |
| G08 | 9 | Grouped bootstrap resamples groups not records | PASS |
| G09 | 10 | Linguistic template split disjointness | PASS |
| G10 | 14-15 | Final split inaccessible before FROZEN | PASS |
| G11 | 15 | Final access ledger records every access | PASS |
| G12 | 11 | No optimal backend encoded in task metadata | PASS |
| G13 | 13 | Strong hand router baseline exists | PASS |
| G14 | 20 | Fail closed on missing utility/verifier/policy | PASS |
| G15 | 21-25 | Steering evaluated by ΔU(α) not P(symbolic) | PASS |
| G16 | 23 | Beneficial/harmful flip classification | PASS |
| G17 | 24 | Oracle alpha probe (DEV only diagnostic) | PASS |
| G18 | 25 | Random control p_emp for steering gain | PASS |
| G19 | 26 | Neutral KL release gate (3 cases) | PASS |
| G20 | 27 | Verifier naming: exact vs constrained_answer | PASS |
| G21 | 28 | Capture representation ablation (DEV only) | PASS |
| G22 | 29 | Decisive policy class comparison | PASS |
| G23 | 30 | Tie-aware metrics (win/tie/loss) | PASS |
| G24 | 31 | ESS reported for weighted estimators | PASS |
| G25 | 32 | Artifact directory discipline | PASS |
| G26 | 33 | Source tree hash enforcement (canonical) | PASS |
| G27 | 34 | Test collection hash recorded | PASS |
| G28 | 35 | Re-run test report | PASS |
| G29 | 37 | Real LLM backend integration (Qwen) | SKIP |
| G30 | 38 | Real symbolic executor integration | SKIP |
| G31 | 12-17 | Within-subtype crossover (≥ 3 subtypes) | PASS |
| G32 | 22-27 | CLI paths completed (real backends) | PASS |

**Summary:** 30 PASS, 2 SKIP, 0 FAIL

---

## 4. Scientific Success Tiers (Section 51)

| Tier | Description | Status |
|------|-------------|--------|
| 0 | Mechanism executes | PASS (synthetic) |
| 1 | Hidden state predicts backend suitability above chance | PENDING (real model) |
| 2 | Learned policy beats constant routes inside crossover subtype | PENDING (real model) |
| 3 | Candidate beats incumbent | PENDING (real model) |
| 4 | Candidate beats strong hand router | PENDING (real model) |
| 5 | Utility weighting beats uniform weighting | PENDING (real model) |
| 6 | Steering improves verified utility | PENDING (real model) |
| 7 | Learned steering beats matched random vectors | PENDING (real model) |
| 8 | Meaningful fraction of oracle routing gap captured | PENDING (real model) |

**Note:** Tiers 1-8 require real Qwen model execution (G29/G30).
The synthetic evidence confirms the mechanism works; real model
qualification is the next step.

---

## 5. Out of Scope (Section 57)

The following were explicitly NOT implemented in this release:
- GDN2 changes
- COCONUT
- Full SAE training
- Model merging (TIES, DARE)
- PPO / GRPO
- Multi-agent debate
- New memory systems
- Multi-vector steering as default
- Learned alpha controller

These are conditional on this qualification pass demonstrating that
the current architecture can test the central hypothesis.
