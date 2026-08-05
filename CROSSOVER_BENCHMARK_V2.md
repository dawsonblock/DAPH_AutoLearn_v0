# CROSSOVER_BENCHMARK_V2 — v0.3.10.3.2-alpha

## Within-Subtype Crossover Benchmark Design

**Release:** v0.3.10.3.2-alpha
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`

---

## 1. Design Goal

The previous benchmark (v0.3.10.3.1) was subtype classification:
entire subtypes were labeled "symbolic-preferred" or "LLM-preferred."
This release introduces **within-subtype crossover**: individual
instances within the same subtype have different optimal backends,
determined by actual executed utility.

The benchmark now tests:

```
same subtype + different instance state → different best backend
```

with both backends available.

---

## 2. Subtypes

| Subtype | Description | NL? | Crossover Mechanism |
|---------|-------------|-----|---------------------|
| A | Integer arithmetic (a op b) | No | Small operands → LLM wins (cost); Large → Symbolic wins (quality) |
| B | NL arithmetic word problem | Yes | Small → LLM wins (correct + cheaper); Large → Symbolic wins (LLM arithmetic errors) |
| C | NL "half of x" | Yes | Small even → LLM wins; Large even → Symbolic wins |
| D | Modular multiplication (a * b mod m) | No | Small mod → LLM wins; Large mod → Symbolic wins |
| E | NL equal-product comparison | Yes | Small products → LLM wins; Large products → Symbolic wins |
| F | NL percentage loss | Yes | Small total → LLM wins; Large total → Symbolic wins |

---

## 3. Crossover Results (DEV split, n=30 per subtype)

| Subtype | N | S wins | L wins | Ties | S win fraction | Always-S regret | Always-L regret |
|---------|---|--------|--------|------|----------------|-----------------|-----------------|
| A | 30 | 17 | 13 | 0 | 0.57 | 0.000 | 0.630 |
| B | 30 | 14 | 16 | 0 | 0.47 | 0.003 | 0.587 |
| C | 30 | 20 | 10 | 0 | 0.67 | 0.000 | 0.560 |
| D | 30 | 22 | 8 | 0 | 0.73 | 0.000 | 0.690 |
| E | 30 | 15 | 15 | 0 | 0.50 | 0.003 | 0.590 |
| F | 30 | 19 | 11 | 0 | 0.63 | 0.000 | 0.570 |

**Qualification:** All 6 subtypes have within-subtype crossover
(0.47 ≤ symbolic_win_fraction ≤ 0.73). The requirement of ≥ 3
subtypes with crossover is exceeded (6/6).

---

## 4. Crossover Mechanism Details

### Case 1: Small operands (both correct)
- LLM correct, symbolic correct
- LLM cost (0.001) < symbolic cost (0.01)
- **LLM wins** on utility (same quality, lower cost)

### Case 2: Large operands (LLM arithmetic error)
- LLM incorrect (arithmetic overflow/error), symbolic exact
- Symbolic quality (1.0) > LLM quality (0.0)
- **Symbolic wins** on utility (higher quality dominates cost)

### Case 3: NL extraction + small calculation
- LLM interprets correctly and computes correctly
- Symbolic semantic parser extracts and computes correctly
- LLM cost < symbolic cost
- **LLM wins** on utility

### Case 4: NL extraction + large calculation
- LLM interprets correctly but arithmetic wrong
- Symbolic semantic parser extracts correctly and computes exactly
- **Symbolic wins** on utility

---

## 5. Utility Configuration

```
quality_weight = 1.0
lambda_time = 0.02
lambda_compute = 0.01
lambda_risk = 0.0
```

- Quality dominates (weight 1.0)
- Cost is non-zero enough to resolve both-correct cases
- LLM normalized_cost = 0.001 (cheaper)
- Symbolic normalized_cost = 0.01 (more expensive)

---

## 6. Semantic Parser (Section 14)

A bounded semantic parser (`daph_learning.data.semantic_parser`)
extracts structured expressions from NL arithmetic prompts. This
allows the symbolic backend to compete on NL tasks (subtypes B, C, E, F).

**Key property:** The parser does NOT use the expected answer — it
only uses the prompt text. If parsing fails, it returns `None` (low
confidence), and the LLM backend is preferred.

---

## 7. Both Backends Available (Section 13)

Both backends are executable on 100% of qualification tasks:
- Symbolic backend: available on structured tasks (A, D) and NL tasks
  via semantic parser (B, C, E, F)
- LLM backend: available on all tasks (simulated in synthetic mode,
  real Qwen in integration mode)

---

## 8. Template Disjointness (Section 10)

Splits are disjoint by actual linguistic template ID (computed from
normalized prompt text), not just slot numbers. Each split has
split-specific wording wrappers for NL subtypes.

```
template_ids_train ∩ template_ids_dev == ∅
template_ids_train ∩ template_ids_cal == ∅
template_ids_train ∩ template_ids_final == ∅
```

---

## 9. Deterministic Generation (Section 9)

All task generation uses `deterministic_seed()` — a SHA-derived
integer seed that is deterministic across processes. Python's
process-randomized `hash()` has been removed from all benchmark
provenance.
