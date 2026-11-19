# CORRECTNESS AUDIT — v0.3.10.1-alpha

> This audit documents every correctness bug found and fixed in the
> v0.3.10.1-alpha release pass. Each entry names the bug, the silent failure
> it caused, the fix, and the source location. Citations use
> `<ref_file>:<ref_snippet>` form. The source of truth is
> `src/daph_learning/` and the `CHANGELOG.md` v0.3.10.1-alpha entry.

## 1. Config / CLI mismatches

### 1.1 `soft_targets: bool` → `TargetMode` enum (P0-1 / G1)

**Bug.** The config exposed `soft_targets: bool` implying two modes (soft vs.
hard labels), but the logistic trainer always produced
`q_i = sigmoid(ΔU_i / τ)` regardless of the flag. "Hard logistic" was never
actually hard-label logistic — the flag was a no-op.

<ref_file>src/daph_learning/policy/logistic.py</ref_snippet>
```python
# v0.3.10.1 — breaking changes (clean break):
# * The trainer now accepts a `target_mode` (TargetMode.SOFT or
#   TargetMode.HARD) and applies a validity mask to the loss. The old
#   `soft_targets: bool` flag was a *correctness bug*: the trainer
#   always produced q_i = sigmoid(ΔU_i / τ) regardless of the flag
```

**Fix.** Replaced the bool with an explicit `TargetMode` enum
(`SOFT` | `HARD`) and `build_preference_targets(delta_u, mode, temperature,
gap_threshold)` which returns `(targets, valid_mask)`.

<ref_file>src/daph_learning/policy/targets.py</ref_snippet>
```python
class TargetMode(str, Enum):
    SOFT = "soft"
    HARD = "hard"

def build_preference_targets(delta_u, mode, temperature, gap_threshold):
    # SOFT:  q_i = sigmoid(ΔU_i / τ),  valid_mask = all True
    # HARD:  y_i ∈ {0,1} with |ΔU|<=gap masked out (valid_mask=False)
```

Hard-mode ties (`|ΔU| <= gap_threshold`) are *ignored* via the mask, not
coerced to 0.5. Coercing ties to 0.5 would train the router to output 0.5 on
near-ties, silently changing the learned policy geometry. The config field is
now `target_mode: str` ∈ `{"soft", "hard"}`.

<ref_file>src/daph_learning/policy/config.py</ref_snippet>
```python
# Targets (Section 1). v0.3.10.1 — replaces soft_targets: bool.
target_mode: str = "soft"
target_temperature: float = 1.0
```

The loss applies the mask:

<ref_file>src/daph_learning/policy/logistic.py</ref_snippet>
```python
eff_weight = sample_weight * valid_mask.to(sample_weight.dtype)
denom = eff_weight.sum().clamp_min(1e-8)
return (loss * eff_weight).sum() / denom
```

### 1.2 `weight_mode` gap|snr → 4-mode enum (P0-2 / G2)

**Bug.** The config exposed `weight_mode: "gap" | "snr"`, but the experience
construction always called the gap function, silently ignoring `"snr"`. A user
who selected `"snr"` got gap weighting with no error.

<ref_file>src/daph_learning/policy/weighting.py</ref_snippet>
```python
# v0.3.10.1 — breaking change: the old WeightMode.GAP / WeightMode.SNR
# two-mode enum was a *correctness bug*. The configuration exposed
# weight_mode: "gap" | "snr" but the experience construction code always
# called utility_weight (the gap function), silently ignoring snr.
```

**Fix.** Replaced the two-mode enum with a four-mode `WeightMode` enum
(`UNIFORM`, `ABSOLUTE_GAP`, `CLIPPED_GAP`, `SNR`) and a unified
`compute_weight(...)` that actually dispatches on mode.

<ref_file>src/daph_learning/policy/weighting.py</ref_snippet>
```python
class WeightMode(str, Enum):
    UNIFORM = "uniform"
    ABSOLUTE_GAP = "absolute_gap"
    CLIPPED_GAP = "clipped_gap"
    SNR = "snr"

def compute_weight(delta_u, confidence, mode, gap_threshold, max_weight, *,
                   sigma_delta_u=None, min_weight=0.0, eps=1e-8):
    if mode == WeightMode.UNIFORM:   return 1.0
    if abs(delta_u) <= gap_threshold: return 0.0          # tie truncation
    if mode == WeightMode.ABSOLUTE_GAP: return max(|ΔU|*c, min_weight)
    if mode == WeightMode.CLIPPED_GAP:  return min(max(|ΔU|*c, min_weight), w_max)
    if mode == WeightMode.SNR:
        if sigma_delta_u is None:
            raise ValueError("SNR mode requires sigma_delta_u")  # fail closed
        return min(max(|ΔU|/(σ+ε)*c, min_weight), w_max)
```

SNR requires `sigma_delta_u` and **fails closed** if absent. The experience
builder now passes the configured mode through to `compute_weight`:

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
weight = compute_weight(
    delta_u, conf, cfg.weight_mode,
    gap_threshold=cfg.gap_threshold,
    max_weight=cfg.max_weight,
    min_weight=cfg.min_weight,
    sigma_delta_u=sigma,
)
```

**Clean break:** the old `"gap"` string raises `ValueError` at config
construction (via `WeightMode.from_str`). Callers must migrate
`gap` → `absolute_gap` (or `clipped_gap` to keep the historical
max_weight=1.0 clipping) and `snr` → `snr`.

<ref_file>src/daph_learning/policy/weighting.py</ref_snippet>
```python
# The legacy GAP and SNR string aliases from v0.3.10 are NOT accepted —
# clean break per v0.3.10.1-alpha. Callers must migrate gap -> absolute_gap
```

### 1.3 `policy_type` always trained logistic → real dispatch (P0-3 / G3)

**Bug.** The config accepted multiple policy types but the integrated learner
always trained logistic routing. Selecting `centroid` or any other type had no
effect on the trained model.

**Fix.** Added a `PolicyModel` protocol and real `centroid` / `logistic` /
`mlp_experimental` implementations. A factory `fit_policy(...)` dispatches on
`config.policy_type`.

<ref_file>src/daph_learning/policy/policy_factory.py</ref_snippet>
```python
@runtime_checkable
class PolicyModel(Protocol):
    def fit(self, train_features, train_delta_u, train_weights, **kwargs): ...
    def predict_proba(self, h): ...
    def save(self, path): ...

def fit_policy(config, train_features, train_delta_u, train_weights, **kwargs):
    pt = config.policy_type
    if pt == "centroid":         return train_centroid_policy(...)
    if pt == "logistic":         return train_weighted_logistic_router(...)
    if pt == "mlp_experimental": return train_small_mlp_router(...)
    raise ValueError(f"unknown policy_type {pt!r}; ...")
```

The `centroid` implementation is a real calibrated policy
(`p = sigmoid((h·v - s0)/τ_cal)`), not just a direction extractor:

<ref_file>src/daph_learning/policy/centroid_policy.py</ref_snippet>
```python
class CentroidPolicy:
    # v = μ_S - μ_L  (weighted contrastive mean)
    # s(h) = h · v
    # p = sigmoid((s - s0) / τ_cal)
```

The config validates `policy_type` ∈ `{"centroid", "logistic",
"mlp_experimental"}` and rejects `"lowrank"` (removed from v0.3.10.1 spec).

<ref_file>src/daph_learning/policy/config.py</ref_snippet>
```python
if self.policy_type not in ("centroid", "logistic", "mlp_experimental"):
    raise ValueError("policy_type must be 'centroid', 'logistic', "
                     "or 'mlp_experimental'")
```

## 2. Fixed silent fallbacks

### 2.1 Zero-utility synthesis when `utility_fn` missing (P0-4 / G4)

**Bug.** Held-out evaluation defaulted to zero utility when `utility_fn` was
absent. This silently made every candidate look identical (all zero utility ⇒
zero regret ⇒ no discrimination), so a broken utility callback was invisible.

**Fix.** `dev_tasks is not None and utility_fn is None` now raises
`ValueError`. All scorer/verifier/utility callbacks fail closed.

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
# --- Fail closed on missing utility_fn (Section 4) --------------
if dev_tasks is not None and utility_fn is None:
    raise ValueError(
        "utility_fn is required for held-out policy evaluation "
        "(Section 4): refusing to synthesize zero utility")
```

### 2.2 Silent zip truncation (P0-5 / P0-6 / G5 / G6)

**Bug.** Every `zip(experiences, activations)` and
`zip(dev_tasks, dev_experiences)` in AutoLearn code inferred alignment from
array order. A shuffled feature order would silently misalign features from
experiences; a missing capture would silently truncate the loss/eval set
without any warning (Python `zip` stops at the shorter sequence).

**Fix.** Naked row arrays at cross-module boundaries replaced with task-bound
`FeatureRecord` objects. `join_by_task_id(experiences, feature_records)`
asserts no duplicate `task_id` in either input, reports missing feature
records explicitly (no silent truncation), and is order-independent.

<ref_file>src/daph_learning/policy/alignment.py</ref_snippet>
```python
def join_by_task_id(experiences, feature_records, *, strict=True):
    exp_dups = _find_duplicates(e.task_id for e in experiences)
    feat_dups = _find_duplicates(r.task_id for r in feature_records)
    # ... always raise on duplicates — silent dedup would be unsafe
    for exp in experiences:
        feats = feature_by_id.get(exp.task_id)
        if feats is None:
            missing.append(exp.task_id)   # reported, not silently dropped
            continue
```

The learner's dev-task join also refuses to silently zip-misalign:

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
for exp in dev_exps_aligned:
    t = task_by_id.get(exp.task_id)
    if t is None:
        raise ValueError(
            f"dev_tasks is missing task_id {exp.task_id!r} that is present "
            f"in dev_experiences; refusing to silently zip-misalign")
```

A legacy naked-array path is retained for existing synthetic tests, but it
asserts length equality:

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
if len(experiences) != len(arr):
    raise ValueError(f"{label}: experiences ({len(experiences)}) and "
                     f"activations ({len(arr)}) length mismatch; pass "
                     f"FeatureRecord objects to join by task_id instead")
```

### 2.3 SNR mode ignored — covered in §1.2

The `"snr"` selection being silently ignored is the same root cause as §1.2.
The unified `compute_weight` dispatch closes it, and SNR now fails closed when
`sigma_delta_u` is absent rather than falling back to gap weighting.

## 3. Alignment bugs

### 3.1 Naked array order inference → FeatureRecord task_id join (P0-5 / G5)

**Bug.** The previous policy APIs accepted `experiences` and an `activations`
ndarray and inferred alignment from array order. A shuffled feature order
would silently misalign features from experiences, and a missing capture would
silently truncate the loss/eval set.

<ref_file>src/daph_learning/policy/alignment.py</ref_snippet>
```python
# The previous policy APIs accepted experiences and an activations
# ndarray and inferred alignment from array order. This is unsafe: a
# shuffled feature order would silently misalign features from
# experiences, and a missing capture would silently truncate the
# loss/eval set without any warning.
```

**Fix.** `FeatureRecord` carries the `task_id` with the feature vector, and
`join_by_task_id` produces an order-independent, length-safe alignment. An
`assert_unique_task_ids` helper is provided for callers that build their own
records.

<ref_file>src/daph_learning/policy/types.py</ref_snippet>
```python
@dataclass(frozen=True)
class FeatureRecord:
    # Replaces naked np.ndarray rows at cross-module boundaries so that
    # experiences and activations can be joined by task_id instead of
    # inferred from array order.
    task_id: str
    features: np.ndarray
```

## 4. Calibration correction (P0-8 / G8)

### 4.1 Old ECE was mathematically wrong for soft labels

**Bug.** The old ECE used `confidence = max(p, 1-p)` with
`accuracy = y` (the hard 0/1 label) even when the training targets were soft
(`q_i = sigmoid(ΔU/τ)`). For soft labels this throws away the continuous
preference information: a prediction of `p=0.6` against a soft target
`q=0.55` is scored as "correct" (argmax matches) with confidence 0.6, even
though the probability is badly calibrated.

<ref_file>src/daph_learning/policy/calibration.py</ref_snippet>
```python
# v0.3.10.1 — Section 8: correct calibration math for soft/tie labels.
# the old ECE used confidence = max(p,1-p) with accuracy = y for soft
# labels, which is mathematically wrong.
```

### 4.2 New `preference_brier_soft`

Compares the predicted `P(S|h)` directly against the soft target `q`:

```
B_soft = (1/N) Σ (p_i - q_i)^2
```

<ref_file>src/daph_learning/policy/calibration.py</ref_snippet>
```python
def preference_brier_soft(p_symbolic, target_symbolic_prob):
    # B_soft = (1/N) Σ (p_i - q_i)^2
    # Compares the predicted P(S|h) directly against the soft target q = σ(ΔU/τ).
    return float(np.mean((p - q) ** 2))
```

This is the correct probability-calibration metric when targets are soft; the
legacy `brier_score` against hard 0/1 labels silently discards the continuous
preference information.

### 4.3 New `action_confidence_ece`

Uses the predicted action's expected correctness under the soft target,
`c*_i = predicted_action_correctness_target(p, q)`, instead of the hard label:

```
a_hat_i = S  if p_i >= 0.5  else L
c*_i     = q_i      if a_hat_i = S
           1 - q_i  if a_hat_i = L
ECE = Σ_b (n_b/N) |mean(c_hat_b) - mean(c*_b)|
```

For hard labels (`q_i ∈ {0,1}`), `c*_i` reduces to ordinary 0/1 correctness, so
this reduces to the standard ECE.

<ref_file>src/daph_learning/policy/calibration.py</ref_snippet>
```python
def predicted_action_correctness_target(p_symbolic, target_symbolic_prob):
    choose_symbolic = p >= 0.5
    return np.where(choose_symbolic, q, 1.0 - q)

def action_confidence_ece(p_symbolic, target_symbolic_prob, n_bins=10):
    confidences = np.maximum(p, 1.0 - p)
    correctness = predicted_action_correctness_target(p, q)
    # ECE = Σ_b (n_b/N) |mean(c_hat_b) - mean(c*_b)|
```

The integrated learner now reports both corrected metrics alongside the
legacy ones (kept for ablation):

<ref_file>src/daph_learning/policy/learner.py</ref_snippet>
```python
"preference_brier_soft": float(preference_brier_soft(probs, soft_targets_np)),
"action_confidence_ece": float(action_confidence_ece(probs, soft_targets_np)),
"brier_score": float(brier_score(probs, soft_targets_np)),   # legacy
"ece": float(expected_calibration_error(probs, hard_labels)), # legacy
```

## 5. `weighted_mean` negative-weight rejection (P0-7 / G7)

**Bug.** `weighted_mean` accepted any numeric weights, including negative
values. Negative weights have no statistical interpretation for a centroid
mean and would silently produce a malformed steering direction.

**Fix.** `weighted_mean` now rejects negative weights, NaN, Inf, all-zero
effective weights, and non-finite activations.

<ref_file>src/daph_learning/policy/centroid.py</ref_snippet>
```python
def weighted_mean(activations, weights, eps=1e-8):
    if not np.all(np.isfinite(activations)):
        raise ValueError("activations contain NaN/Inf")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain NaN/Inf")
    # v0.3.10.1 — Section 7: reject negative weights.
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= eps:
        raise ValueError("effective sample weight is zero")
    return (activations * weights[:, None]).sum(axis=0) / total
```

## 6. `dev_regret` early stopping actually using regret (P0-9 / G9)

**Bug.** The trainer used dev BCE loss for early stopping even when the config
said `dev_regret`. The "regret-based" early stopping never measured regret —
it optimized the surrogate loss, not the scientific metric.

<ref_file>src/daph_learning/policy/logistic.py</ref_snippet>
```python
# v0.3.10.1 — breaking changes (clean break):
# * Early stopping is now selectable (dev_loss | dev_regret | dev_utility),
#   default dev_regret. The previous trainer always used dev BCE loss even
#   when the config said dev_regret.
```

**Fix.** Early stopping is now selectable (`dev_loss` | `dev_regret` |
`dev_utility`), default `dev_regret`. The `dev_regret` path executes the
actual utility function with the deployment `confidence_threshold` (not 0.5),
so early stopping optimizes the same decision rule used at inference.

<ref_file>src/daph_learning/policy/logistic.py</ref_snippet>
```python
def _evaluate_dev_metric(router, metric, h_dev, du_dev, w_dev,
                         targets_dev, mask_dev, *, dev_tasks=None,
                         utility_fn=None, cfg=None, confidence_threshold=0.5):
    # dev_regret / dev_utility use choose_route with the deployment
    # confidence_threshold (not 0.5), so early stopping optimizes the same
    # decision rule used at inference.
    if metric in ("dev_regret", "dev_utility"):
        for i, task in enumerate(dev_tasks):
            p = float(p_np[i])
            route = choose_route(p, confidence_threshold)   # deployment rule
            cu = utility_fn(task, route)
            ...
        if metric == "dev_regret":
            return float(mean_regret(cand_utils, ora_utils))
        return float(-cand_utils.mean())   # dev_utility: higher is better
```

The config field `early_stopping_metric` defaults to `"dev_regret"` and is
validated:

<ref_file>src/daph_learning/policy/config.py</ref_snippet>
```python
early_stopping_metric: str = "dev_regret"  # dev_loss | dev_regret | dev_utility
...
if self.early_stopping_metric not in ("dev_loss", "dev_regret", "dev_utility"):
    raise ValueError("early_stopping_metric must be 'dev_loss', "
                     "'dev_regret', or 'dev_utility'")
```

`dev_loss` remains available for ablation. If `dev_regret`/`dev_utility` is
selected but `dev_tasks`/`utility_fn` are absent, the trainer falls back to
`dev_loss` (the integrated learner always supplies both and validates before
calling).

## 7. Summary table

| ID  | Bug                                              | Silent failure                       | Fix                                           |
|-----|--------------------------------------------------|--------------------------------------|-----------------------------------------------|
| P0-1| `soft_targets: bool` ignored                     | hard logistic never hard             | `TargetMode` enum + validity mask             |
| P0-2| `weight_mode` "snr" ignored                      | SNR user got gap weighting           | 4-mode `WeightMode` + real dispatch           |
| P0-3| `policy_type` always trained logistic           | centroid/mlp selections no-op        | `PolicyModel` protocol + `fit_policy` factory |
| P0-4| zero-utility fallback when `utility_fn` missing | all candidates look identical        | fail closed `ValueError`                      |
| P0-5| naked array order inference                      | shuffled order misaligns silently    | `FeatureRecord` + `join_by_task_id`           |
| P0-6| silent zip truncation                            | missing capture drops examples       | task_id join / length assertions              |
| P0-7| `weighted_mean` accepted negative weights        | malformed steering direction         | reject negative/NaN/Inf/all-zero              |
| P0-8| ECE wrong for soft labels                        | calibration appears fine when wrong  | `preference_brier_soft` + `action_confidence_ece` |
| P0-9| `dev_regret` used dev BCE loss                   | regret early stopping never measured | selectable metric, real utility path          |
