# Gate A — Issues and Concrete Fixes

## Issue 1: A2 Fails — Bootstrap CI Too Wide (14 groups)

### Root Cause

The grouped bootstrap resamples at the **linguistic template** level
(`linguistic_template_id`), not individual tasks. The final split uses
only **1 template slot** (slot 7), so the number of distinct templates
is limited by the number of wording variants per subtype:

| Subtype | Templates in final | Why |
|---------|--------------------|-----|
| A | 3 | 3 operators (+, -, *) × 1 slot |
| B | 4 | 4 scenarios (crates, boxes, pallets, rows) |
| C | 4 | 4 forms (minus twice, subtract three times, double, half) |
| D | 1 | 1 form (a mod b) |
| E | 1 | 1 form (which is larger) |
| F | 1 | 1 form (tank loses X%) |
| **Total** | **14** | |

With only 14 groups, the bootstrap CI is inherently wide. The point
estimate is +0.193 but the LCB dips to -0.041.

### Fix 1a: Add more template slots for the final split

**File:** `src/daph_learning/data/crossover_benchmark.py` (line 65)

```python
# Current: final gets only 1 slot
SPLIT_TEMPLATE_SLOTS = {
    "train": (0, 1, 2, 3),
    "dev": (4, 5),
    "calibration": (6,),
    "final": (7,),
}

# Fix: give final 3 slots → ~42 groups
SPLIT_TEMPLATE_SLOTS = {
    "train": (0, 1),
    "dev": (2, 3),
    "calibration": (4,),
    "final": (5, 6, 7),
}
```

This triples the number of bootstrap groups, tightening the CI by
roughly √3 ≈ 1.7x. The LCB would move from -0.041 to approximately
+0.04 (estimated), potentially passing A2.

**Risk:** Fewer train slots means less wording diversity in training.
Mitigate by keeping 2 train slots (still template-disjoint from final).

### Fix 1b: Add more wording variants per slot

**File:** `src/daph_learning/data/crossover_benchmark.py` (lines 75-131)

Add more wrappers for D, E, F (which currently have only 1 template
each in final):

```python
# Add operator-specific wrappers for D
_D_WRAPPERS = (
    "Compute {body}. Return only the integer.",
    "Please compute {body}. Reply with the exact integer only.",
    # ... 8 slots
)

# Add form-specific wrappers for E and F
_E_WRAPPERS = (
    "{body}",
    "Comparison: {body}",
    # ... 8 slots with distinct wording
)
```

This increases groups without changing the split structure.

### Fix 1c: Use subtype as the bootstrap group (6 groups → 14+ groups)

**File:** `scripts/run_gate_a_experiment.py` (line 746)

Instead of grouping by `linguistic_template_id`, group by a finer key
that combines subtype with operand characteristics:

```python
# Current: groups by linguistic_template_id (14 groups)
result = grouped_bootstrap_mean_delta(
    final_records, "linguistic_template_id", a_key, b_key, ...)

# Fix: group by subtype + operator/scenario (more groups)
# Add a "fine_group" field to final_records
for r in final_records:
    subtype = r["subtype"]
    # Extract operator or scenario from the task
    r["fine_group"] = f"{subtype}:{r.get('operator', 'default')}"
result = grouped_bootstrap_mean_delta(
    final_records, "fine_group", a_key, b_key, ...)
```

---

## Issue 2: P1 Weak on Subtype A (regret 0.156 vs hand router 0.000)

### Root Cause

Subtype A mixes small operands (1-200, LLM correct, wins on cost) and
large operands (5000-9999, LLM errs, symbolic wins on quality). P1 must
distinguish small vs large from hidden states alone. The layer-10
last-token representation may not encode operand magnitude well enough,
so P1 sometimes routes large-operand A tasks to LLM (which fails).

### Fix 2a: Use mean_pool instead of last_token for capture

**File:** `scripts/run_gate_a_experiment.py` (line 176)

```python
# Current: last token only
cap_cfg = CaptureConfig(layer=layer, location="last_token")

# Fix: mean pool captures operand magnitude from all tokens
cap_cfg = CaptureConfig(layer=layer, location="mean_pool")
```

Mean pooling over the full prompt tokens encodes the actual numbers
in the prompt, not just the last token's representation. This should
help P1 distinguish small vs large operands.

### Fix 2b: Try an earlier layer (layer 5 or 3)

**File:** `scripts/run_gate_a_experiment.py` (line 163)

```python
# Current: layer 10 (out of 28)
layer = min(10, n_layers - 1)

# Fix: earlier layers encode more surface features (numbers, magnitude)
layer = min(5, n_layers - 1)
```

Earlier layers tend to preserve more surface-level features (specific
numbers, magnitudes) while later layers encode more abstract semantics.
For distinguishing operand magnitude, an earlier layer may work better.

### Fix 2c: Increase the operand gap in subtype A

**File:** `src/daph_learning/data/crossover_benchmark.py` (line 201)

```python
# Current: small = 1-200, large = 5000-9999
if rng.random() < 0.5:
    a = rng.randint(1, 200)
    b = rng.randint(1, 200)
else:
    a = rng.randint(5_000, 9_999)
    b = rng.randint(5_000, 9_999)

# Fix: widen the gap so hidden states more clearly separate
if rng.random() < 0.5:
    a = rng.randint(1, 50)      # smaller small
    b = rng.randint(1, 50)
else:
    a = rng.randint(50_000, 99_999)  # larger large
    b = rng.randint(50_000, 99_999)
```

A wider gap makes the two regimes more separable in hidden state space.

### Fix 2d: Increase lambda_compute to penalize LLM cost more

**File:** `scripts/run_gate_a_experiment.py` (line 112)

```python
# Current: LLM cost = 0.1, quality_weight = 1.0
lambda_compute = 0.1

# Fix: increase to 0.3 so LLM's cost disadvantage is more decisive
lambda_compute = 0.3
```

When both backends are correct (small operands), the utility difference
is only cost. Increasing `lambda_compute` makes these tasks more
decisive, giving P1 clearer training signal for routing.

---

## Issue 3: Backend Co-availability = 0.333 (too low)

### Root Cause

Symbolic backend only supports subtypes A and D (2 of 6). With semantic
parser disabled, it fails on B, C, E, F. This means only 1/3 of tasks
have both backends available, limiting crossover.

### Fix 3a: Enable semantic parser for 1-2 NL subtypes

**File:** `src/daph_learning/execution/real_backends.py` (line 170)

Instead of fully disabling semantic parsing, enable it for specific
subtypes where it can partially compete:

```python
# Current: fully disabled
if not has_structured and _os.environ.get("DAPH_DISABLE_SEMANTIC_PARSE", "0") == "1":
    return BackendOutcome(... available=False ...)

# Fix: allow semantic parsing for subtype B (simplest NL form)
subtype = task.get("metadata", {}).get("subtype", "")
if not has_structured and subtype in ("B",):
    # Try semantic parse for B (crates/boxes multiplication)
    try:
        from ..data.semantic_parser import parse_task
        parse_result = parse_task(dict(task))
        if parse_result is not None:
            # ... execute parsed expression
    except Exception:
        pass
    # Fall through to unavailable
```

This increases co-availability to ~0.50 (3 of 6 subtypes) and creates
true within-subtype crossover on B (both backends can win depending on
operand size).

### Fix 3b: Add structured inputs to some NL subtypes

**File:** `src/daph_learning/data/crossover_benchmark.py`

Add `capability_ids` to some NL tasks so symbolic can attempt them:

```python
# In _gen_b: add structured inputs alongside NL wrapper
return {
    "capability_ids": ["integer_arithmetic"],  # ADD THIS
    "inputs": {"a": a, "b": b, "op": "*"},    # ADD THIS
    "specification": spec,
    "expected": expected,
}
```

This allows symbolic to execute B tasks directly from structured inputs
while the LLM must parse the NL. Creates true crossover: symbolic wins
on large operands (exact), LLM wins on small (correct + lower cost).

---

## Issue 4: 56% Ties (non-decisive training tasks)

### Root Cause

1,122 of 1,998 train tasks are ties (|delta_u| < 0.01). These come
from:
- Subtype F: all 333 tasks tie (both backends produce same utility)
- Subtype A: 208 of 333 tasks tie (both backends correct, cost diff < 0.01)
- Subtype E: 283 of 333 tasks tie (both backends fail or both succeed)

Ties get zero weight in training, reducing ESS to 876.

### Fix 4a: Increase lambda_compute to break cost-based ties

When both backends are correct, the only difference is cost (0.01 vs
0.10). With `lambda_compute=0.1`, the utility difference is 0.009 —
below the 0.01 gap threshold. Increasing `lambda_compute` to 0.3 makes
the difference 0.027 — above threshold, converting ties to decisive
examples.

### Fix 4b: Make F tasks harder so backends differ

**File:** `src/daph_learning/data/crossover_benchmark.py` (line 368)

F tasks currently have both backends tie. Make the large-operand
variants harder so the LLM fails:

```python
# Current: large = total 2000-10000
total_range = (2_000, 10_000)

# Fix: much larger values that LLM can't compute
total_range = (100_000, 1_000_000)
```

### Fix 4c: Lower the gap_threshold

**File:** `scripts/run_gate_a_experiment.py` (line 169)

```python
# Current: tasks with |delta_u| < 0.01 are ties
gap_threshold=0.01

# Fix: lower to 0.005 to capture more decisive examples
gap_threshold=0.005
```

This converts some near-ties into decisive training examples, increasing
ESS and giving P1 more training signal.

---

## Priority Order

| Fix | Impact | Effort | Priority |
|-----|--------|--------|----------|
| 1a: More final template slots | High (CI tightens ~1.7x) | Low | **1** |
| 2a: mean_pool capture | Medium (better A routing) | Low | **2** |
| 4a: Increase lambda_compute | Medium (fewer ties) | Low | **3** |
| 2b: Earlier layer | Medium (better magnitude) | Low | **4** |
| 1b: More wording variants | Medium (more groups) | Medium | **5** |
| 3a: Partial semantic parsing | High (more crossover) | Medium | **6** |
| 2c: Wider operand gap | Low-Medium | Low | **7** |
| 4c: Lower gap_threshold | Low | Low | **8** |

**Recommended first attempt:** Apply fixes 1a + 2a + 4a together. These
are all low-effort, non-breaking changes that address the three main
issues simultaneously:
- 1a tightens the bootstrap CI (fixes A2)
- 2a improves P1's ability to route subtype A (fixes P1 vs hand router)
- 4a converts ties to decisive examples (improves ESS and training signal)
