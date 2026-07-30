# WEIGHTING_ABLATION_REPORT — v0.3.10.3.2-alpha

## Weighting Ablation Report

**Release:** v0.3.10.3.2-alpha
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`

---

## 1. Design (Section 38)

Hold policy family constant. Compare weighting modes within each
family:

**Centroid:**
- uniform
- absolute-gap
- clipped-gap

**Logistic:**
- uniform
- absolute-gap
- clipped-gap

**Weighting value is demonstrated only when:**

```
same-policy weighted regret < same-policy uniform regret
```

Do NOT infer weighting benefit from comparisons across different
policy classes.

---

## 2. Weight Modes

### Uniform
All training examples receive weight `w_i = 1.0`.

### Absolute-gap
`w_i = |ΔU_i|` — the absolute utility difference between backends.

### Clipped-gap
`w_i = clip(|ΔU_i|, 0, max_gap)` — absolute gap with upper clipping
to prevent outlier dominance.

---

## 3. ESS Reporting (Section 21)

For each weighting mode:

```
ESS = (Σ w_i)² / Σ w_i²
```

| Mode | ESS (expected) | Zero-weight fraction |
|------|----------------|----------------------|
| Uniform | N (full sample) | 0% |
| Absolute-gap | < N | > 0% (ties get 0) |
| Clipped-gap | < N but > absolute-gap | > 0% (ties get 0) |

**Key insight:** A weighted model trained on effectively 7 examples
must not be reported as though N=1000.

---

## 4. Synthetic Baseline Matrix (Section 44)

| Method | Utility ↑ | Regret ↓ |
|--------|-----------|----------|
| Always LLM | 0.4054 | 0.5924 |
| Always Symbolic | 0.9963 | 0.0015 |
| Hand Router | 0.9093 | 0.0886 |
| Random | 0.6869 | 0.3110 |
| Oracle | 0.9978 | 0.0000 |

**Note:** These are synthetic results. The symbolic backend is exact
on all arithmetic, so always-symbolic has very low regret. Real Qwen
model results may differ (LLM may be correct on more tasks, changing
the crossover balance).

---

## 5. Required Real-Model Ablation Table

| Method | Utility ↑ | Regret ↓ | Oracle Gap ↑ | Decisive Acc ↑ | ESS |
|--------|-----------|----------|--------------|-----------------|-----|
| Always LLM | — | — | — | — | N |
| Always S | — | — | — | — | N |
| Hand Router | — | — | — | — | N |
| Centroid Uniform | — | — | — | — | — |
| Centroid Weighted | — | — | — | — | — |
| Logistic Uniform | — | — | — | — | — |
| Logistic Weighted | — | — | — | — | — |
| Logistic Hard | — | — | — | — | — |
| MLP Weighted | — | — | — | — | — |
| AutoLearn | — | — | — | — | — |

**Status:** PENDING real Qwen model execution (G29/G30).

---

## 6. Interpretation Guide (Section 39)

| Observation | Interpretation |
|-------------|----------------|
| Centroid ≈ Logistic | Simple geometry suffices |
| Logistic >> Centroid | Linear decision boundary better than mean difference |
| MLP >> Logistic | Nonlinear geometry in hidden state |
| Weighted < Uniform (same policy) | Weighting adds value |
| Weighted ≥ Uniform (same policy) | Weighting does not help |
