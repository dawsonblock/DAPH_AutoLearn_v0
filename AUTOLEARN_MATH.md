# AUTOLEARN_MATH.md — DAPH AutoLearn v0.3.10-alpha

## 1. Utility Equation

For task `x_i`, execute both available computational backends `a ∈ {S, L}`:

```
U_i(a) = w_q · Q_i(a)
       - λ_t · T_i(a) / T_ref
       - λ_c · C_i(a) / C_ref
       - λ_r · R_i(a)
```

where:
- `Q_i(a)` = verified task quality / correctness (1.0 iff correct, 0.0 otherwise)
- `T_i(a)` = latency
- `C_i(a)` = monetary or normalized compute cost
- `R_i(a)` = risk / safety penalty (1.0 if execution failed, 0.0 otherwise)
- `w_q, λ_t, λ_c, λ_r` = configurable coefficients (frozen in `UtilityConfig`)

The utility difference:

```
ΔU_i = U_i(S) - U_i(L)
```

- `ΔU > 0` → symbolic was better
- `ΔU < 0` → LLM was better
- `ΔU ≈ 0` → essentially tied

The optimal action: `a_i* = argmax_a U_i(a)`. The training system retains the
continuous utility difference rather than discarding it into a hard label.

## 2. Regret (Primary Metric)

Per-task regret:

```
Regret_i = max_a U_i(a) - U_i(π(x_i))
```

Mean regret:

```
R̄ = (1/N) Σ_i Regret_i
```

The primary goal: `E[Regret_new] < E[Regret_incumbent]`.

Routing accuracy is secondary. A policy can have high accuracy but make its
mistakes on tasks where the utility penalty is enormous.

## 3. Reward-Gap Weighting

For experience `i`:

```
g_i = |ΔU_i|          (utility gap)
c_i ∈ [0, 1]          (combined confidence)
w_i = clip(|ΔU_i| · c_i, w_min, w_max)
```

**Gap-threshold tie truncation (Section 7):** near-ties get zero weight, not a
positive floor:

```
if |ΔU_i| <= gap_threshold:
    w_i = 0
```

The numerical stability floor `w_min` is applied **only after** gap filtering.
It is NOT a statistical minimum importance.

## 4. Weighted Centroid (Baseline)

Weighted class means:

```
μ_S = Σ_{i∈S} w_i h_i / Σ_{i∈S} w_i
μ_L = Σ_{i∈L} w_i h_i / Σ_{i∈L} w_i
v_centroid = μ_S - μ_L
```

**Theoretical correction (Section 4):** the utility gap measures the
*importance* of choosing the correct action. It does NOT identify the causal
activation direction that changes the policy. The weighted centroid is a
baseline, compared against a learned decision boundary.

## 5. Soft Targets

```
q_i = σ(ΔU_i / τ)
```

where `τ` is the temperature:
- `ΔU >> 0` → `q ≈ 1` (strongly prefer symbolic)
- `ΔU << 0` → `q ≈ 0` (strongly prefer LLM)
- `ΔU ≈ 0` → `q ≈ 0.5` (tied)

Smaller `τ` → sharper preference. Larger `τ` → softer preference.

## 6. Logistic Objective

Policy:

```
p_i = P(S | h_i) = σ(w^T h_i + b)
```

Weighted binary cross entropy:

```
L = - Σ_i ω_i [ q_i log p_i + (1 - q_i) log(1 - p_i) ]
```

where `ω_i = f(|ΔU_i|, confidence_i)`. With L2 regularization (weight decay).

## 7. Calibration

Brier score:

```
Brier = (1/N) Σ_i (p_i - y_i)^2
```

Expected Calibration Error:

```
ECE = Σ_b (n_b / N) |conf_b - acc_b|
```

## 8. OOD Detection (Mahalanobis)

```
d_M(h) = sqrt((h - μ)^T Σ^{-1} (h - μ))
```

with regularized covariance `Σ + ridge · I`. If `d_M(h) > τ_OOD`, route to
ABSTAIN.

## 9. Calibrated Abstention

```
conf = max(p, 1 - p)
if conf < τ_conf:  action = ABSTAIN
elif p >= 0.5:     action = SYMBOLIC
else:              action = LLM
```

## 10. Intervention Equations

Given candidate direction `v`:

```
h'(δ) = h + δ v
```

Measure route probability, selected action, and downstream utility across
`δ ∈ {-α, -α/2, 0, α/2, α}`.

```
ΔU_intervention(δ) = U(policy under δ) - U(policy under δ=0)
```

Direction reversal test: `P(S|h+αv) > P(S|h) > P(S|h-αv)` must hold
statistically in aggregate.

## 11. KL / Capability Promotion Criterion

Candidate promotion requires ALL:

```
utility_gain >= minimum_gain
AND  neutral_KL <= KL_budget
AND  capability_drop <= allowed_drop
AND  n_samples >= min_samples
```

## 12. Paired Bootstrap

```
CI = bootstrap percentiles of mean(deltas)
```

where `deltas_i = U_candidate_i - U_incumbent_i`.

## 13. Doubly-Robust Utility (Future)

```
U_DR = U_hat(x, a)
     + I[a = observed_action] / p(a|x) · (U_observed - U_hat(x, a))
```

Valid only when propensities are logged and overlap conditions hold.

## 14. Low-Rank Multi-Vector Controller (Experimental, v0.4 direction)

```
V = [v_1, ..., v_K]  ∈ R^{D×K}
α(h) = g_φ(h)
h' = h + V α(h)
```

Redundancy regularization:

```
L_orth = || V_hat^T V_hat - I ||_F^2
```

Interference matrix:

```
I_jk = U(v_j + v_k) - U(v_j) - U(v_k) + U(0)
```
