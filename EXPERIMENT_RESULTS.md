# Real-Model Pilot Results — NOT Gate A Qualification

**Date:** 2026-07-30
**Release:** v0.3.10.4-alpha
**Hardware:** Apple Silicon (MPS), 16GB unified memory
**Status:** REAL_MODEL_PILOT_COMPLETED — NOT Gate A qualified

---

## 1. Executive Summary

Two real-model pilot runs were executed end-to-end through the staged
pipeline. **Neither run constitutes a valid Gate A experiment.** Both
runs exposed critical defects that must be fixed before any Gate A
qualification can proceed.

| Run | Model | Correct Status | Root Cause |
|-----|-------|---------------|------------|
| `daph_gate_a_smoke` | Qwen2.5-0.5B | REAL_MODEL_SMOKE: EXECUTION PASSED | Pipeline functional, no routing decision |
| `daph_gate_a_real_002` | Qwen2.5-1.5B | PRECONDITION_FAILURE: NOT_EVALUABLE | No crossover + insufficient groups + policy bug |

### Critical Defects Found

1. **Centroid policy degenerates to coin flip (p=0.5) when one class
   is empty.** When all training examples prefer the same backend,
   the policy outputs 0.5 for every task — not the constant that
   picks the dominant backend. This produced the mathematically
   impossible oracle_gap_capture = 0.5 (should be 1.0 if P1 matches
   oracle). **Fixed.**

2. **Benchmark generator produced only 9 final groups (60 total).**
   The minimum_groups requirement was 60. Instead of fixing the
   generator, the threshold was lowered to 9. **Reverted. Generator
   expanded to 420 groups (63 final).**

3. **No genuine within-subtype crossover.** The symbolic backend
   solves 100% of tasks across all 6 subtypes. The routing problem
   is degenerate. **Not yet fixed — requires benchmark redesign.**

4. **Gate engine did not distinguish precondition failures from
   statistical failures.** A "6/7 gates passed" summary was reported
   when the experiment was not admissible. **Fixed: precondition
   failures now yield NOT_EVALUABLE.**

---

## 2. The Oracle Gap Capture Bug

### The Arithmetic Inconsistency

The original report claimed:
> "P1 correctly learns to always route to symbolic, capturing exactly
> 50% of the oracle gap."

This is mathematically impossible. If P1 always routes to symbolic
and symbolic is always utility-optimal:

```
U(P1) = U(oracle)
oracle_gap_capture = (U(P1) - U(P0)) / (U(oracle) - U(P0)) = 1.0
```

not 0.5.

### Root Cause

The centroid policy (`centroid_policy.py` lines 159-173) had this
fallback when one class was empty:

```python
if sym_mask.sum() == 0 or llm_mask.sum() == 0:
    self.vector = np.zeros(feats.shape[1], dtype=np.float32)
    self.threshold = 0.0
    self.temperature = 1.0
    return self
```

A zero vector with threshold=0.0 and temperature=1.0 produces:

```
sigmoid((h · 0 - 0) / 1) = sigmoid(0) = 0.5
```

for every input. P1 was not "learning to always route to symbolic" —
it was **silently degenerating to a coin flip**.

### Fix

When one class is empty, the policy now produces a constant that
picks the dominant backend:

- All symbolic-preferred → `threshold = -100.0` → `sigmoid(100) ≈ 1.0`
- All LLM-preferred → `threshold = 100.0` → `sigmoid(-100) ≈ 0.0`

A `degenerate_train_data` flag is set so downstream code can detect
this condition.

### Invariant Test

Added `tests/test_oracle_gap_invariants.py` with:

```python
def test_oracle_gap_capture_is_one_when_policy_matches_oracle():
    p0 = np.array([0.1, 0.2, 0.3])
    oracle = np.array([0.8, 0.9, 1.0])
    p1 = oracle.copy()
    capture = oracle_gap_capture(
        policy_utility=p1.mean(),
        baseline_utility=p0.mean(),
        oracle_utility=oracle.mean(),
    )
    assert capture == pytest.approx(1.0)
```

Also tests:
- `oracle_gap_capture == 0.0` when P1 matches baseline
- `oracle_gap_capture == 0.5` when P1 is a coin flip
- Centroid policy produces p≈1.0 when all symbolic-preferred
- Centroid policy produces p≈0.0 when all LLM-preferred

---

## 3. The minimum_groups Protocol Violation

### What Happened

The configuration required `minimum_groups: 60`. The generator
produced 60 total groups, with only 9 in the final split (15%).

Instead of fixing the generator, the threshold was lowered to 9 and
the corresponding test was changed to expect 9.

### Why This Was Wrong

The 60-group requirement exists because grouped confidence inference
needs enough independent groups for stable bootstrap intervals. Nine
final groups are not equivalent to 60.

Changing the threshold after discovering the data couldn't satisfy it
invalidates the preregistered protocol.

### Fix

- **Restored** `minimum_groups: 60` in `gate_a_real_002.yaml`
- **Restored** the test to expect 60
- **Expanded** the generator from 60 to 420 total groups (70 per
  subtype), producing 63 final groups at 15% split fraction
- **Expanded** prompt wrapper templates from 8 to 70+ per subtype to
  avoid near-duplicate Jaccard collisions

---

## 4. The Crossover Problem

### Current State

The symbolic backend solves 100% of tasks across all 6 subtypes
(A–F). The LLM solves <10%. There is no crossover — the optimal
policy is a constant:

```
π(x) = SYMBOLIC  for all x
```

This means:
- Hidden states are not needed
- Experience aggregation is not needed
- Uncertainty-aware targets are not needed
- The logistic router is not needed
- A one-line constant policy is sufficient

### What's Needed

The benchmark needs at least four outcome regimes:

1. **Symbolic-favorable:** Exact expressions, structured arithmetic
2. **LLM-favorable:** Tasks requiring semantic interpretation
3. **Both-correct, symbolic cheaper:** Simple arithmetic both solve
4. **Both-capable with instance-dependent failure:** Controlled
   parser traps, phrasing diversity, ambiguity

For at least 3 subtypes, require:

```
P(S wins | s) ≥ 0.20  and  P(L wins | s) ≥ 0.20
```

### Status

**Not yet fixed.** The current generators produce arithmetic tasks
where the symbolic backend is universally correct. Redesigning the
benchmark to produce genuine crossover is the next priority.

---

## 5. Gate Engine Corrections

### Typed Comparator

Added a `Comparator` enum to replace string-based direction checks:

```python
class Comparator(str, Enum):
    GT = "gt"    # strictly greater than
    GTE = "gte"  # greater than or equal
    LT = "lt"    # strictly less than
    LTE = "lte"  # less than or equal
```

Gate verdicts now report the explicit comparator:

```json
{
    "minimum_oracle_gap_capture": {
        "actual": 0.5000,
        "threshold": 0.5000,
        "comparator": "gte",
        "passed": true
    }
}
```

### Precondition Failure → NOT_EVALUABLE

The gate engine now checks preconditions before evaluating statistical
gates:

1. **Minimum independent groups:** `actual_groups >= minimum_groups`
2. **Meaningful crossover:** Both backends must win some tasks

If a precondition fails, all downstream gates are marked
`NOT_EVALUABLE` rather than counted as passed. The overall status
becomes `NOT_EVALUABLE` instead of `FAIL`.

### Route Distribution Reporting

Added route distribution metrics to the experiment results:

```json
{
    "route_distribution": {
        "p1_symbolic_fraction": 1.0,
        "p1_llm_fraction": 0.0,
        "p1_abstain_fraction": 0.0,
        "oracle_symbolic_fraction": 1.0,
        "oracle_llm_fraction": 0.0,
        "p1_oracle_action_agreement": 1.0,
        "p1_always_symbolic_agreement": 1.0
    }
}
```

These metrics expose whether P1 actually learned a conditional policy
or just a constant.

---

## 6. Pilot Run Details

### daph_gate_a_smoke (Qwen2.5-0.5B-Instruct)

| Metric | Value |
|--------|-------|
| Gate status | REAL_MODEL_SMOKE: EXECUTION PASSED |
| Symbolic accuracy | 36/36 (100%) |
| LLM accuracy | 0/36 (0%) |
| Feature dim | 896 |
| Capture layer | 16 |
| P1 utility | 0.5016 |
| Oracle gap capture | 0.5000 (bug — should be 1.0 with fix) |

### daph_gate_a_real_002 (Qwen2.5-1.5B-Instruct)

| Metric | Value |
|--------|-------|
| Gate status | NOT_EVALUABLE (precondition failure) |
| Symbolic accuracy | 72/72 (100%) |
| LLM accuracy | 7/72 (9.7%) |
| Feature dim | 1536 |
| Capture layer | 18 |
| P1 utility | 0.4530 (bug — coin flip, not routing) |
| Oracle gap capture | 0.5000 (bug — should be 1.0 with fix) |
| Final groups | 9 (insufficient — needs 60) |

### Per-Subtype Breakdown (real_002)

| Subtype | N | Symbolic | LLM | LLM Verification |
|---------|---|----------|-----|------------------|
| B | 16 | 16/16 | 2/16 | 13 UNVERIFIABLE, 2 CORRECT, 1 INCORRECT |
| C | 24 | 24/24 | 4/24 | 20 UNVERIFIABLE, 4 CORRECT |
| D | 8 | 8/8 | 0/8 | 8 INCORRECT |
| E | 8 | 8/8 | 1/8 | 5 UNVERIFIABLE, 2 INCORRECT, 1 CORRECT |
| F | 16 | 16/16 | 0/16 | 16 UNVERIFIABLE |

---

## 7. Required Corrections Before Next Run

1. ~~Investigate oracle-gap capture = 0.5~~ **Fixed: centroid policy
   degenerate fallback**
2. ~~Restore minimum_groups = 60~~ **Fixed: generator expanded to 420
   groups**
3. ~~Relabel runs as pilot~~ **Done: this document**
4. ~~Gate engine precondition failure → NOT_EVALUABLE~~ **Fixed**
5. ~~Typed Comparator enum~~ **Fixed**
6. ~~Route distribution reporting~~ **Fixed**
7. **Redesign benchmark for genuine within-subtype crossover** —
   PENDING
8. **Add strong baselines** (always-symbolic, always-LLM, subtype-only,
   surface features, TF-IDF, hidden-state, shuffled-label, oracle) —
   PENDING
9. **Rerun structural and crossover audits before policy training** —
   PENDING
10. **Freeze new experiment ID after benchmark passes preconditions** —
    PENDING
11. **Only then run the expensive qualification experiment** — PENDING

---

## 8. What Was Genuinely Accomplished

The engineering pipeline is real and functional:

- Real Hugging Face model loading (Qwen2.5-0.5B, 1.5B)
- Real hidden state capture (896-dim, 1536-dim)
- Real symbolic execution (100% accuracy)
- Real LLM generation with canonical verification
- Real utility calculation via single `compute_utility` entry point
- Real policy training (centroid, with degenerate fallback now fixed)
- Real calibration
- Real freeze manifest with all identity-bearing inputs
- Real one-shot final access ledger
- Real sham control (20 seeds, shared training spec)
- Real bundle validation (source hash, dataset hashes, etc.)
- Real source-hash invalidation (correctly detected post-freeze edits)

The pipeline infrastructure is sound. The benchmark and policy
implementation had defects that are now identified and partially
fixed. The remaining work is benchmark engineering — creating genuine
crossover — not model scaling or more tests.

---

*This document was generated from machine-readable artifacts and
code analysis. The oracle gap capture bug was found by tracing the
exact computation: P1 utility was exactly half of the always-symbolic
utility, and the centroid policy was producing p=0.5 for every task
due to a zero-vector fallback when one class was empty.*
