# STEERING_UTILITY_REPORT — v0.3.10.3.2-alpha

## Steering Utility Validation Report

**Release:** v0.3.10.3.2-alpha
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`

---

## 1. Design (Sections 28-32)

Steering utility is measured via **verified downstream utility**:

```
ΔU(α) = U(π(h + αv)) - U(π(h))
```

NOT via `P(symbolic | h + αv)`.

The full causal chain for every alpha:

```
intervention → route change → backend execution → verification → utility
```

---

## 2. Alpha Grid

Default alpha grid: `[-1.0, -0.5, 0.0, +0.5, +1.0]`

α = 0 is the baseline (no intervention).

---

## 3. Required Steering Result Table (Section 47)

| Alpha | Mean P(S) | Utility ↑ | Regret ↓ | Flip % | Beneficial % | Harmful % | Mean KL |
|-------|-----------|-----------|----------|--------|--------------|-----------|---------|
| -1.0 | — | — | — | — | — | — | — |
| -0.5 | — | — | — | — | — | — | — |
| 0.0 | — | — | — | — | — | — | — |
| +0.5 | — | — | — | — | — | — | — |
| +1.0 | — | — | — | — | — | — | — |

**Status:** PENDING real Qwen model execution.

---

## 4. Beneficial/Harmful Flip Analysis (Section 29)

For every route flip (route changes from α=0 to α≠0):

| Classification | Condition |
|----------------|-----------|
| Beneficial | U(α) > U(0) |
| Harmful | U(α) < U(0) |
| Neutral | |U(α) - U(0)| ≤ tolerance |

**Useful steering requires beneficial effect**, not merely more
symbolic routing.

---

## 5. Oracle Alpha Diagnostic (Section 30)

DEV only:

```
alpha_oracle(x) = argmax_alpha U(x, alpha)
```

Compare:
- α = 0 (baseline)
- Best single global α
- Per-task oracle α

```
steering_adaptivity_gap = U(per-task oracle α) - U(best global α)
```

If this is large, context-dependent α may be worth implementing in a
future release.

**Do NOT deploy oracle alpha. Do NOT use it on final.**

---

## 6. Random Steering Controls (Section 31)

Matched controls:
- Same layer
- Same norm
- Same alpha range
- Same tasks
- Same policy
- Same model

| Configuration | N vectors |
|---------------|-----------|
| Smoke | ≥ 20 |
| Research | ≥ 500 |

Empirical p-value:

```
p = (1 + count(random_gain ≥ learned_gain)) / (N + 1)
```

If learned steering does not beat random: **report steering
specificity not demonstrated.**

---

## 7. KL Safety Gate (Section 32)

Candidate steering/promotion fails if:

```
KL > threshold
```

If KL is unavailable in qualified mode: **FAIL CLOSED**.
No null-as-pass behavior.

---

## 8. Synthetic Evidence

The steering utility evaluation framework (`evaluate_steering_utility`)
has been validated with synthetic backends:

- ΔU(α) is computed via canonical `backend_utility`
- Beneficial/harmful flip classification works correctly
- Random control p_emp is computed correctly
- Oracle alpha is DEV-only diagnostic

**Real model validation is pending** (G29/G30).

---

## 9. Scientific Success Tiers for Steering

| Tier | Description | Status |
|------|-------------|--------|
| 6 | Steering improves verified utility | PENDING (real model) |
| 7 | Learned steering beats matched random vectors | PENDING (real model) |

**Key requirement:** Changing the latent score is not enough. The
full chain must be:

```
intervention → route change → executed backend → verified outcome → utility improvement
```
