# DAPH AutoLearn v0.3.10.3.1-alpha — Qualification Repair Release Report

**Release date:** 2026-07-29  
**Mission:** Make the evidence trustworthy. Focused repair, not architecture expansion.

---

## 1. Summary

This release is a qualification-repair release. It does NOT add GDN2,
COCONUT, model merging, SAE, PPO/GRPO, or multi-agent systems. Instead
it fixes the scientific validity of the existing system: canonical
utility, honest benchmarking, frozen evaluation discipline, source-hash
enforcement, and a release-gate claim contract (G1-G32).

**884 tests collected; all pass** (1 hardware-dependent skip on machines
without Qwen model access).

---

## 2. What changed (by section)

| Section | Change | Gate(s) | Test file |
|---------|--------|---------|-----------|
| 1 | Version unified to 0.3.10.3.1-alpha | G1 | test_version_claims_discipline.py |
| 2 | Centroid zero-weight fail-closed + ZeroWeightPolicy | G2 | test_zero_weight_policy_v0310_3_1.py |
| 3 | One canonical utility function (backend_utility) | G3 | test_canonical_utility_v0310_3_1.py |
| 4-5 | BackendOutcome contract + confidence != quality | G4, G5 | test_outcome_contract_v0310_3_1.py |
| 6-7 | True weighted vs unweighted ablation | G6 | test_benchmark_stats_v0310_3_1.py |
| 8-9 | Within-split dedup + grouped bootstrap | G7, G8 | test_benchmark_stats_v0310_3_1.py |
| 10-12 | Within-family crossover benchmark | G9, G12 | test_crossover_benchmark_v0310_3_1.py |
| 13 | Strong hand router baseline | G13 | test_benchmark_stats_v0310_3_1.py |
| 14-15 | ExperimentStage enum + final access ledger | G10, G11 | test_stage_access_v0310_3_1.py |
| 20 | Fail closed on missing components | G14 | test_kl_failclosed_v0310_3_1.py |
| 21-25 | Steering utility (ΔU(α)) + flips + oracle + random | G15-G18 | test_steering_utility_v0310_3_1.py |
| 26 | Neutral KL release gate (3 cases) | G19 | test_kl_failclosed_v0310_3_1.py |
| 27 | Verifier naming: exact vs constrained_answer | G20 | test_v0310_2_real_gates.py |
| 28 | Capture representation ablation (DEV only) | G21 | test_capture_comparison_v0310_3_1.py |
| 29-31 | Decisive policy comparison + tie-aware + ESS | G22-G24 | test_capture_comparison_v0310_3_1.py |
| 32-35 | Artifact discipline + source hash + re-run report | G25-G28 | test_artifact_integrity_v0310_3_1.py |
| 37-40 | Release-gate claim contract + G1-G32 registry | G31, G32 | test_release_gates_v0310_3_1.py |

---

## 3. The two most important changes

### 3.1 Within-family crossover benchmark (Section 10-12)

The old benchmark was too easy: `arithmetic -> symbolic` vs
`letter counting -> LLM`. A router could solve it by task-family
classification alone.

The new `structured_math` family contains subtypes A-F where both
symbolic and LLM can win on different instances inside the SAME family:

- **A** (direct exact arithmetic): symbolic wins (LLM errs on large products)
- **B** (semantic extraction + arithmetic): LLM wins (no structured inputs)
- **C** (ambiguous expression): LLM wins (requires semantic interpretation)
- **D** (structured modular arithmetic): symbolic wins (LLM errs on large dividends)
- **E** (comparison/relation): LLM wins (multi-step reasoning)
- **F** (multi-step NL arithmetic): LLM wins (requires parsing word problem)

**P(symbolic optimal | family) = 0.333** — stably in (0.2, 0.8) across
seeds. This makes instance-level routing non-trivial: a constant-action
policy cannot achieve zero regret inside this family.

The optimal backend is NEVER stored in task metadata. It is derived
only after both backends execute, the verifier runs, and canonical
utilities are computed.

### 3.2 Steering utility evaluation (Section 21-25)

The previous build showed that pushing harder in the learned symbolic
direction can raise `P(symbolic | +v)` while LOWERING utility at the
strongest dose. This release preserves that negative result and
evaluates steering strictly through:

```
ΔU(α) = U(π(h + α v)) - U(π(h))
```

NOT through `max_alpha P(symbolic | h + α v)`.

For every intervention strength α, the full chain is executed:
intervention -> route change -> backend execution -> verification ->
utility/regret change. Flips are classified as beneficial/harmful/
neutral by utility change, not by P(symbolic) change.

---

## 4. Negative results supported

This release explicitly supports negative results as valid scientific
outcomes:

- **Weighting did not help** (Section 6-7): the weighted-vs-unweighted
  ablation may show no significant improvement. This is reported
  honestly, not hidden.
- **Steering hurt utility** (Section 21-25): the steering utility
  report may show `best_global_alpha_utility < baseline_utility`. This
  is recorded, not suppressed.
- **Best policy not decisive** (Section 29-31): if the best policy is
  tied with another, `best_is_decisive=False` is reported with a note.

---

## 5. Gate status (G1-G32)

All 32 gates pass. See `src/daph_learning/evaluation/release_gates.py`
for the full registry mapping each gate to its section, claims,
and test files.

---

## 6. New modules

| Module | Purpose |
|--------|---------|
| `data/crossover_benchmark.py` | Within-family crossover benchmark + optimal distribution derivation |
| `routing/hand_router.py` | Strong hand-router baseline |
| `evaluation/grouped_stats.py` | Grouped bootstrap + ESS + weight diagnostics |
| `policy/stage.py` | ExperimentStage enum + final access ledger |
| `steering/utility_eval.py` | Steering utility evaluation via ΔU(α) |
| `evaluation/artifact_integrity.py` | Artifact discipline + source hash + re-run report |
| `evaluation/capture_ablation.py` | Capture representation ablation (DEV only) |
| `evaluation/policy_comparison.py` | Decisive policy comparison + tie-aware + ESS |
| `evaluation/release_gates.py` | G1-G32 gate registry + claim contract |

---

## 7. Modified modules

| Module | Change |
|--------|--------|
| `policy/types.py` | BackendOutcome: `verified_correct` vs `correct`, `verifier_status` |
| `policy/utility.py` | Canonical `backend_utility` + `utility_for_route` |
| `policy/centroid_policy.py` | Zero-weight fail-closed + ZeroWeightPolicy |
| `policy/config.py` | Version + ZeroWeightPolicy + intervention_alpha_grid |
| `policy/provenance.py` | Version + zero-weight fallback recording |
| `policy/calibration.py` | CalibrationArtifact expanded with hashes + Brier + ECE |
| `execution/real_backends.py` | VerifierMode enum + verify_exact_string tightened + verify_constrained_answer |
| `interventions/kl_gate.py` | p95 KL + threshold recording |
| `cli/commands/policy.py` | Uses canonical utility |
| `autolearn/counterfactual.py` | Uses canonical utility |

---

## 8. Test summary

- **884 tests collected**
- **883 pass, 1 skip** (hardware-dependent Qwen model test)
- **0 failures, 0 errors**

New test files:
- `test_crossover_benchmark_v0310_3_1.py` (9 tests)
- `test_benchmark_stats_v0310_3_1.py` (14 tests)
- `test_stage_access_v0310_3_1.py` (10 tests)
- `test_steering_utility_v0310_3_1.py` (7 tests)
- `test_kl_failclosed_v0310_3_1.py` (7 tests)
- `test_artifact_integrity_v0310_3_1.py` (11 tests)
- `test_capture_comparison_v0310_3_1.py` (7 tests)
- `test_release_gates_v0310_3_1.py` (8 tests)
