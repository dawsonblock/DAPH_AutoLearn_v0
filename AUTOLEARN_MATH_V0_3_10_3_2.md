# AUTOLEARN_MATH_V0_3_10_3_2 — v0.3.10.3.2-alpha

## Mathematical Specification

**Release:** v0.3.10.3.2-alpha
**Source tree SHA-256:** `eec93338490a8dafabb1263fac79d76d99d746cd882a88be1ec4c6e834e295bd`

---

## 1. Canonical Utility (Section 3)

For a backend outcome `o` and config `cfg`:

```
U(o) = w_Q · Q(o) - λ_t · T(o)/T_ref - λ_c · C(o)/C_ref - λ_r · R(o)
```

Where:
- `Q(o)` = quality ∈ {0.0, 1.0} (verified correct → 1.0, else 0.0)
- `T(o)` = latency (seconds)
- `C(o)` = normalized cost
- `R(o)` = risk
- `w_Q` = quality weight (default: 1.0)
- `λ_t` = time penalty (default: 0.02)
- `λ_c` = compute penalty (default: 0.01)
- `λ_r` = risk penalty (default: 0.0)

**Default values** (recorded in immutable `ExperimentConfig`):

| Parameter | Value |
|-----------|-------|
| w_Q | 1.0 |
| λ_t | 0.02 |
| λ_c | 0.01 |
| λ_r | 0.0 |
| confidence_threshold | 0.7 |
| gap_threshold | 0.01 |

---

## 2. Regret (Section 42)

Per task:

```
Regret_i = max(U_S(i), U_L(i)) - U_policy(i)
```

Where:
- `U_S(i)` = utility of symbolic backend on task i
- `U_L(i)` = utility of LLM backend on task i
- `U_policy(i)` = utility of the policy's chosen route

Primary qualification metric: **mean regret**.

Confidence intervals via grouped bootstrap (Section 43).

---

## 3. Oracle Gap (Section 41)

```
U_oracle = mean(max(U_S, U_L))
oracle_gap = U_oracle - U_baseline
```

For policy π:

```
η = (U_π - U_baseline) / (U_oracle - U_baseline)
```

Where baseline = always-LLM.

Interpretation:
- η = 0: no recovered routing value
- η = 0.5: captures half available advantage
- η = 1: oracle-level routing

**Current synthetic values:**
- U_oracle = 0.9978
- U_baseline (always-LLM) = 0.4054
- oracle_gap = 0.5924

---

## 4. Steering Utility (Section 28)

For intervention strength α:

```
ΔU(α) = U(π(h + αv)) - U(π(h))
```

Where:
- `h` = pre-execution hidden state
- `v` = steering vector
- `α` = intervention strength
- `π(h + αv)` = policy probability under steered state

The full causal chain:

```
intervention → route change → backend execution → verification → utility
```

**NOT measured by:** P(symbolic | h + αv)

---

## 5. Beneficial/Harmful Flips (Section 29)

For every route flip (route changes from α=0 to α≠0):

```
beneficial: U(α) > U(0)
harmful:    U(α) < U(0)
neutral:    |U(α) - U(0)| ≤ tolerance
```

Report:
- Total flip rate
- Beneficial flip rate
- Harmful flip rate
- Net flip utility

---

## 6. Random Control p-value (Section 31)

```
p_emp = (1 + count(random_gain ≥ learned_gain)) / (N + 1)
```

Where:
- `learned_gain` = ΔU for the learned steering vector
- `random_gains` = ΔU for N matched random vectors
- N ≥ 20 (smoke), N ≥ 500 (research)

---

## 7. Effective Sample Size (Section 21)

For weighted training:

```
ESS = (Σ w_i)² / Σ w_i²
```

Report:
- ESS overall
- ESS symbolic class
- ESS LLM class
- Zero-weight fraction
- Decisive sample count
- Tie count

---

## 8. Grouped Bootstrap (Section 43)

Bootstrap by `template_id` (or another dependency group).

```
For b in 1..B:
    Sample groups with replacement
    Compute mean(value_a) - mean(value_b) on resampled data
CI = [percentile(2.5), percentile(97.5)] of bootstrap distribution
```

Default: B = 10,000 bootstrap samples.

---

## 9. Tie-Aware Routing (Section 20)

```
decisive if |ΔU| > ε_gap
tie if |ΔU| ≤ ε_gap
```

Metrics:
- Mean utility
- Mean regret
- Decisive-route accuracy
- Tie-aware route accuracy (ties not counted as mistakes)
- Abstention rate

---

## 10. Stage Machine (Section 34)

Monotonic FSM:

```
TRAIN → DEV → CALIBRATION → FROZEN → FINAL
```

Disallowed transitions:
- FINAL → anything
- FROZEN → DEV
- CALIBRATION → TRAIN
- DEV → TRAIN

Final access budget: 1 (default).

After final access: experiment state = SEALED.
