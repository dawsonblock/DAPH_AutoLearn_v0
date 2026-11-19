# AUTOLEARN MATH — v0.3.10.1-alpha

> Source of truth: `src/daph_learning/policy/` modules and
> `src/daph_learning/interventions/`. This document transcribes the exact math
> implemented in the v0.3.10.1-alpha policy learner. Citations use
> `<ref_file>:<ref_snippet>` form.

## 1. Utility `U_i(a)` and `ΔU`

Each backend `b ∈ {symbolic, llm}` is executed counterfactually on every task
and independently verified. The frozen per-backend utility is:

```
U_b = quality_weight · Q_b  −  λ_t · (T_b / time_reference_ms)
                            −  λ_c · (C_b / compute_reference)
                            −  λ_r · R_b
```

where `Q_b` is the verified-quality reward, `T_b` the latency, `C_b` the
normalized compute cost, and `R_b` the execution-risk penalty. All weights and
normalization references are frozen and hashed so a recorded `U_b` is
reproducible.

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
def _utility(outcome, cfg):
    t = outcome.latency_sec * 1000.0
    time_term = cfg.lambda_time * (t / cfg.time_reference_ms)
    compute_term = cfg.lambda_compute * (outcome.normalized_cost / cfg.compute_reference)
    risk_term = cfg.lambda_risk * outcome.risk
    return (cfg.quality_weight * outcome.quality
            - time_term - compute_term - risk_term)
```

The utility difference (the learning signal) is:

```
ΔU_i = U_symbolic,i − U_llm,i
```

The preferred action outside the abstention band is:

```
preferred_action_i = SYMBOLIC  if ΔU_i > abstention_band
                    LLM        if ΔU_i < -abstention_band
                    ABSTAIN    otherwise
```

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
delta_u = u_sym - u_llm
if delta_u > cfg.abstention_band:
    preferred = Route.SYMBOLIC
elif delta_u < -cfg.abstention_band:
    preferred = Route.LLM
else:
    preferred = Route.ABSTAIN
```

The combined outcome confidence is the conservative product of components
(verifier × measurement × stability × ood):

<ref_file>src/daph_learning/policy/confidence.py</ref_snippet>
```python
def combined(self):
    result = 1.0
    for value in (self.verifier, self.measurement, self.stability, self.ood):
        result *= value
    return float(result)
```

## 2. Weighting (4 modes)

Per-example training weight `ω_i = f(|ΔU_i|, confidence_i, mode)`. Near-ties
(`|ΔU| <= gap_threshold`) are truncated to zero weight in all non-uniform
modes (Section 7 Mode A); a positive floor would accidentally make near-ties
matter again.

```
UNIFORM:       ω_i = 1
ABSOLUTE_GAP:  ω_i = max(|ΔU_i| · c_i, min_weight)
CLIPPED_GAP:   ω_i = min(max(|ΔU_i| · c_i, min_weight), max_weight)
SNR:           ω_i = min(max(|ΔU_i| / (σ_ΔU_i + ε) · c_i, min_weight), max_weight)
```

`min_weight` is a numerical-stability floor applied **only after** gap
filtering; it is not a statistical minimum importance. SNR requires
`sigma_delta_u` and fails closed if absent.

<ref_file>src/daph_learning/policy/weighting.py</ref_snippet>
```python
def compute_weight(delta_u, confidence, mode, gap_threshold, max_weight, *,
                   sigma_delta_u=None, min_weight=0.0, eps=1e-8):
    if mode == WeightMode.UNIFORM:
        return 1.0
    if abs(delta_u) <= gap_threshold:      # tie truncation
        return 0.0
    if mode == WeightMode.ABSOLUTE_GAP:
        return float(max(abs(delta_u) * confidence, min_weight))
    if mode == WeightMode.CLIPPED_GAP:
        return float(min(max(abs(delta_u) * confidence, min_weight), max_weight))
    if mode == WeightMode.SNR:
        if sigma_delta_u is None:
            raise ValueError("SNR mode requires sigma_delta_u")
        raw = abs(delta_u) / (sigma_delta_u + eps)
        return float(min(max(raw * confidence, min_weight), max_weight))
```

## 3. Hard / soft targets

The continuous `ΔU` is converted into a training target `q_i` via
`build_preference_targets(delta_u, mode, temperature, gap_threshold)`:

**Soft target (SOFT):**

```
q_i = sigmoid(ΔU_i / τ)
valid_mask_i = True   (all examples contribute)
```

**Hard target (HARD):**

```
y_i = 1.0  if ΔU_i >  gap_threshold
     0.0  if ΔU_i < -gap_threshold
     IGNORE  if |ΔU_i| <= gap_threshold   (valid_mask_i = False)
```

Hard-mode ties are *ignored* via the mask, not coerced to 0.5. Coercing ties
to 0.5 trains the router to output 0.5 on near-ties, silently changing the
learned policy geometry. Masked positions are filled with 0.5 so they are
harmless if the caller forgets the mask, but the mask MUST be applied for the
loss to be correct.

<ref_file>src/daph_learning/policy/targets.py</ref_snippet>
```python
if mode == TargetMode.HARD:
    targets = np.full(du.shape, 0.5, dtype=np.float64)
    valid_mask = np.abs(du) > gap_threshold
    targets = np.where(du > gap_threshold, 1.0,
                np.where(du < -gap_threshold, 0.0, targets))
```

## 4. Logistic objective with mask

The logistic router learns `P(S | h) = σ(w^T h + b)`. The objective is the
weighted masked binary cross entropy with logits:

```
L = - Σ_i ω_i · m_i [ q_i log p_i + (1 - q_i) log(1 - p_i) ]
    / max(eps, Σ_i ω_i · m_i)
```

where `ω_i` is the per-example utility weight and `m_i` is the target validity
mask. Zero-weight or masked examples are effectively excluded.

<ref_file>src/daph_learning/policy/logistic.py</ref_snippet>
```python
def weighted_policy_loss(logits, target_prob, sample_weight, valid_mask=None):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target_prob, reduction="none")
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    eff_weight = sample_weight * valid_mask.to(sample_weight.dtype)
    denom = eff_weight.sum().clamp_min(1e-8)
    return (loss * eff_weight).sum() / denom
```

## 5. Regret

Regret is the **primary scientific metric**, not routing accuracy.

**Per-task regret:**

```
Regret_i = max_a U_i(a) − U_i(π(x_i))
```

where `max_a U_i(a)` is the oracle counterfactual used ONLY for scoring regret
AFTER the policy acts — it must never replace the policy's actual decision.

**Mean regret:**

```
R̄ = (1/N) Σ_i Regret_i
```

Tiny negative values from floating-point error are clamped to 0 (a policy
cannot beat the oracle).

<ref_file>src/daph_learning/policy/regret.py</ref_snippet>
```python
def per_task_regret(policy_utilities, oracle_utilities):
    regrets = ou - pu
    regrets = np.maximum(regrets, 0.0)
    return regrets

def mean_regret(policy_utilities, oracle_utilities):
    return float(per_task_regret(policy_utilities, oracle_utilities).mean())
```

## 6. Calibration

### 6.1 `preference_brier_soft` (Section 8A)

Soft Brier score against the symbolic preference probability:

```
B_soft = (1/N) Σ (p_i − q_i)^2
```

Compares the predicted `P(S|h)` directly against the soft target
`q = σ(ΔU/τ)`. Lower is better. This is the correct probability-calibration
metric when targets are soft.

<ref_file>src/daph_learning/policy/calibration.py</ref_snippet>
```python
def preference_brier_soft(p_symbolic, target_symbolic_prob):
    return float(np.mean((p - q) ** 2))
```

### 6.2 `action_confidence_ece` (Section 8B)

For each example, the predicted action and its expected correctness under the
soft target:

```
a_hat_i = S  if p_i >= 0.5  else L
c_hat_i = max(p_i, 1 - p_i)                        (predicted confidence)
c*_i    = q_i      if a_hat_i = S                   (expected correctness)
          1 - q_i  if a_hat_i = L
```

`c*_i = predicted_action_correctness_target(p, q)`. For hard labels
(`q_i ∈ {0,1}`), `c*_i` reduces to ordinary 0/1 correctness.

```
ECE = Σ_b (n_b / N) |mean(c_hat_b) − mean(c*_b)|
```

where bins are over `c_hat` (confidence) in `[0.5, 1.0]`.

<ref_file>src/daph_learning/policy/calibration.py</ref_snippet>
```python
def predicted_action_correctness_target(p_symbolic, target_symbolic_prob):
    choose_symbolic = p >= 0.5
    return np.where(choose_symbolic, q, 1.0 - q)

def action_confidence_ece(p_symbolic, target_symbolic_prob, n_bins=10):
    confidences = np.maximum(p, 1.0 - p)
    correctness = predicted_action_correctness_target(p, q)
    ece += (count / n) * abs(float(confidences[mask].mean() - correctness[mask].mean()))
```

The integrated learner reports both corrected metrics plus the legacy ones
(kept for ablation):

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
"preference_brier_soft": float(preference_brier_soft(probs, soft_targets_np)),
"action_confidence_ece": float(action_confidence_ece(probs, soft_targets_np)),
"brier_score": float(brier_score(probs, soft_targets_np)),      # legacy
"ece": float(expected_calibration_error(probs, hard_labels)),   # legacy
```

## 7. OOD detection (Mahalanobis + quantile threshold)

The `MahalanobisOOD` detector estimates the training activation mean `μ` and
regularized covariance `Σ`, then scores new activations:

```
d_M(h) = sqrt( (h − μ)^T Σ^{-1} (h − μ) )
```

with `Σ = cov(X_train) + ridge · I` (ridge = 1e-4 by default), fitted on TRAIN
only. If `d_M(h) > τ_OOD`, the task is out-of-distribution and routes to
ABSTAIN (or a conservative fallback).

<ref_file>src/daph_learning/policy/ood.py</ref_snippet>
```python
def fit(self, x):
    self.mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + self.ridge * np.eye(cov.shape[0])
    self.inv_cov = np.linalg.pinv(cov)

def _score_single(self, h):
    delta = h - self.mean
    value = float(delta @ self.inv_cov @ delta)
    return float(np.sqrt(max(value, 0.0)))
```

The threshold `τ_OOD` is calibrated (quantile-based `ood_quantile`, default
0.99) rather than infinite by default in qualified runs:

<ref_file>src/daph_learning/policy/config.py</ref_snippet>
```python
ood_threshold: float = float("inf")
ood_ridge: float = 1e-4
ood_quantile: float = 0.99
```

## 8. Intervention

### 8.1 Dose-response: `h' = h + alpha · v`

For a candidate direction `v`, the dose-response experiment tests several
`α ∈ {-A, -A/2, 0, A/2, A}` (default `A = 1.0`):

```
h'(α) = h + α · v
p'(α) = policy_prob_fn(h'(α))
route'(α) = choose_route(p'(α), confidence_threshold)
U'(α) = utility_fn(h'(α), route'(α))
```

The baseline (`α = 0`) is included. Results record baseline/intervened route,
utility, and probability for each dose.

<ref_file>src/daph_learning/interventions/__init__.py</ref_snippet>
```python
def run_intervention_experiment(task_id, direction_id, h, v, *,
        policy_prob_fn, utility_fn,
        alphas=(-1.0, -0.5, 0.0, 0.5, 1.0), confidence_threshold=0.5):
    p0 = float(policy_prob_fn(h))
    route0 = choose_route(p0, confidence_threshold)
    u0 = float(utility_fn(h, route0))
    for delta in alphas:
        h_int = h + float(delta) * v
        p_int = float(policy_prob_fn(h_int))
        route_int = choose_route(p_int, confidence_threshold)
        u_int = float(utility_fn(h_int, route_int))
        results.append(InterventionResult(...))
```

### 8.2 Direction reversal

A genuine causal direction should produce utility changes of opposite sign
across `+α` and `−α`. The `DirectionReversalResult` reports whether
`P(S|h+αv) > P(S|h)` and `P(S|h-αv) < P(S|h)` hold for more than half the
tasks (`reversal_consistent`), with a sign-test p-value.

<ref_file>src/daph_learning/interventions/__init__.py</ref_snippet>
```python
@dataclass(frozen=True)
class DirectionReversalResult:
    direction_id: str
    plus_greater_than_zero: int   # count where P(S|h+αv) > P(S|h)
    minus_less_than_zero: int     # count where P(S|h-αv) < P(S|h)
    reversal_consistent: bool     # both counts > n/2
    p_value_sign_test: float
```

### 8.3 Real-model residual-stream intervention

For a loaded transformer with a residual-stream hook at `layer`, the baseline
hidden state is captured, then the model runs with `h' = h + α · v` for each
`α` in the dose grid, subject to norm/cosine safety clamps. The
`RealInterventionResult` records route, utility, KL, and clamp telemetry with
`evidence_level = "real_model_causal"`.

<ref_file>src/daph_learning/interventions/real_pipeline.py</ref_snippet>
```python
# 1. Resolve target layer.  2. Install residual-stream hook.
# 3. Capture baseline hidden state.  4. Run with alpha ∈ {-A,-A/2,0,A/2,A}.
# 5. Measure route probability, selected action, downstream utility,
#    token KL, and safety-clamp telemetry.
# h' = h + alpha * v, subject to norm/cosine safety clamps.
```

### 8.4 Evidence levels

Every intervention result is tagged to distinguish mechanical sanity from
causal evidence:

<ref_file>src/daph_learning/interventions/__init__.py</ref_snippet>
```python
# InterventionResult:  evidence_level = "unit_sanity"
# RealInterventionResult: evidence_level = "real_model_causal"
```

The full taxonomy: `unit_sanity` | `synthetic_causal` | `real_model_causal`.

## 9. Promotion (candidate vs incumbent actual decisions, paired bootstrap)

### 9.1 Actual policy decisions (Section 22)

The candidate action is `candidate_policy(h_i)` and the incumbent action is
`incumbent_policy(h_i)`. The oracle is used ONLY for scoring regret, never for
choosing the candidate action.

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
decision = choose_route_with_reason(p, cfg.confidence_threshold,
                                    ood_score=ood_score,
                                    ood_threshold=cfg.ood_threshold)
c_route = decision.route
cu = utility_fn(task, c_route)
# Incumbent: ACTUAL incumbent policy decision (Section 22).
i_route_str = incumbent_route_fn(dev_features[i])
iu = utility_fn(task, i_route)
# Oracle: max_a U(a) — used ONLY for scoring regret.
ora_utils.append(max(u_sym, u_llm))
```

### 9.2 Paired promotion statistics (Section 23)

Reports mean/median utility delta, mean regret delta, win/loss/tie rates, and
a 10,000-draw paired bootstrap CI by task.

```
δ_i = U_candidate,i − U_incumbent,i
mean_utility_delta   = mean_i(δ_i)
median_utility_delta = median_i(δ_i)
win_rate  = count(δ_i > 1e-9) / N
loss_rate = count(δ_i < -1e-9) / N
tie_rate  = 1 − win_rate − loss_rate
```

Bootstrap CI: resample `δ` with replacement `n_bootstrap = 10_000` times,
take the `α/2` and `1 − α/2` quantiles of the resampled means.

<ref_file>src/daph_learning/policy/regret.py</ref_snippet>
```python
def paired_promotion_statistics(candidate_utilities, incumbent_utilities,
                                oracle_utilities=None, *, n_bootstrap=10000,
                                alpha=0.05, seed=0):
    deltas = cu - iu
    wins = int(np.sum(deltas > 1e-9))
    losses = int(np.sum(deltas < -1e-9))
    low, high = paired_bootstrap_mean_ci(deltas, n_bootstrap, alpha, seed)
    ...
    regret_deltas = inc_regret - cand_regret   # positive => candidate better
```

<ref_file>src/daph_learning/policy/regret.py</ref_snippet>
```python
def paired_bootstrap_mean_ci(deltas, n_bootstrap=10000, alpha=0.05, seed=0):
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = deltas[idx].mean()
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return low, high
```

### 9.3 Promotion gate

The gate promotes only when all of: adequate coverage, paired improvement
(exact paired binomial p-value), bounded regressions (per-capability
thresholds), and confidence (Clopper-Pearson lower bound) hold. Otherwise it
rolls back to the incumbent.

<ref_file>src/daph_learning/autolearn/promotion.py</ref_snippet>
```python
@dataclass(frozen=True)
class PromotionGateConfig:
    min_coverage: float = 0.8
    min_discordant_n: int = 5
    max_p_value: float = 0.05
    max_regression_rate: float = 0.2
    min_improved_probability_lower: float = 0.55
    capability_regression_thresholds: Mapping[str, float] = ...
    protected_capabilities: frozenset[str] = ...
```

## 10. Calibrated abstention at inference

For binary policy probability `p = P(S | h)`:

```
conf = max(p, 1 − p)
if OOD(h) > τ_ood:        action = ABSTAIN   (reason: ood)
elif conf < τ_conf:       action = ABSTAIN   (reason: low_confidence)
elif p == 0.5:            action = ABSTAIN   (reason: policy_tie)
elif p >= 0.5:            action = SYMBOLIC
else:                     action = LLM
```

`τ_conf ∈ [0.5, 1.0]` (default 0.70). Abstention reasons are recorded
explicitly — no opaque single-category abstention.

<ref_file>src/daph_learning/policy/abstention.py</ref_snippet>
```python
def choose_route_with_reason(p_symbolic, confidence_threshold=0.70, *,
                             ood_score=None, ood_threshold=float("inf")):
    confidence = max(p_symbolic, 1.0 - p_symbolic)
    if ood_score is not None and ood_score > ood_threshold:
        return RouteDecision(route=ABSTAIN, reason="ood", ...)
    if confidence < confidence_threshold:
        return RouteDecision(route=ABSTAIN, reason="low_confidence", ...)
    if p_symbolic == 0.5:
        return RouteDecision(route=ABSTAIN, reason="policy_tie", ...)
    if p_symbolic > 0.5:
        return RouteDecision(route=SYMBOLIC, reason="symbolic", ...)
    return RouteDecision(route=LLM, reason="llm", ...)
```

## 11. Early stopping (dev regret)

Early stopping is selectable: `dev_loss` | `dev_regret` | `dev_utility`
(default `dev_regret`). The `dev_regret` / `dev_utility` paths execute the
actual utility function with the deployment `confidence_threshold` (not 0.5),
so early stopping optimizes the same decision rule used at inference.

```
dev_loss    : weighted masked BCE on dev                      (lower is better)
dev_regret  : mean dev regret = mean_i(max_a U_i(a) − U_i(π)) (lower is better)
dev_utility : −mean_i U_i(π(x_i))                             (lower is better)
```

<ref_file>src/daph_learning/policy/logistic.py</ref_snippet>
```python
if metric == "dev_regret":
    return float(mean_regret(cand_utils, ora_utils))
# dev_utility: higher is better → return -utility.
return float(-cand_utils.mean())
```

## 12. Centroid policy geometry

The centroid baseline fits weighted class centroids and maps the signed
projection to a calibrated probability:

```
v     = μ_S − μ_L                                (weighted contrastive mean)
μ_S   = Σ_{i∈S} ω_i h_i / Σ_{i∈S} ω_i
s(h)  = h · v
s0    = 0.5 · (mean_{i∈S} s(h_i) + mean_{i∈L} s(h_i))    (mid-point)
τ_cal = std_i(s(h_i) − s0)                                (calibration temp)
p     = sigmoid((s(h) − s0) / τ_cal)
```

<ref_file>src/daph_learning/policy/centroid_policy.py</ref_snippet>
```python
v = weighted_contrastive_mean(sym_feats, sym_w, llm_feats, llm_w, normalize=False)
self.vector = v
s_sym = float((sym_feats @ v).mean())
s_llm = float((llm_feats @ v).mean())
self.threshold = 0.5 * (s_sym + s_llm)
centered = scores - self.threshold
self.temperature = float(max(centered.std(), 1e-8))
```

## 13. Small MLP router (experimental)

Diagnostic baseline: `p = P(S|h) = sigmoid(MLP(h))` where
`MLP(h) = Linear(GELU(Linear(h)))` with `hidden_dim = 64`. Trained with the
same targets, weights, dev-regret, and calibration path as the logistic
router. Only enabled by explicit config (`policy_type="mlp_experimental"`);
never the default. Purpose: if `MLP >> logistic`, routing geometry is
nonlinear.

<ref_file>src/daph_learning/policy/mlp.py</ref_snippet>
```python
class SmallMLPRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
```
