# SYNTHETIC BENCHMARK DESIGN — v0.3.10.1-alpha

> Source of truth: `src/daph_learning/environment_benchmark.py`.
>
> The v0.3.10 synthetic benchmark was too easy: one coordinate almost directly
> encoded class membership, so a matched random direction could achieve
> perfect regret. This document describes the four redesigned mathematical
> benchmark environments plus the random-direction control, each designed so a
> specific method can fail. Every formula below is transcribed from the
> implementation in `environment_benchmark.py`.

## Shared utility construction (Section 10A)

All four environments share the same bounded-utility construction so the
existing execute/utility pipeline works unchanged:

```
p_i  = sigmoid(ΔU_i)
U_S  = p_i
U_L  = 1 - p_i
```

This gives a bounded utility in `[0, 1]` and a `ΔU` that directly reflects the
latent preference score. For environments where the preference is
deterministic (linear, multimodal, xor), `p_i` is clipped to `[eps, 1-eps]`
with `eps = 0.05` so the utility gap is non-trivial but the correct backend
always has strictly higher utility.

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def _build_task(idx, family, h, delta_u, *, prefix="t", noise_sigma=0.0):
    eps = 0.05
    p = float(_sigmoid(np.clip(delta_u, -10.0, 10.0)))
    p = min(max(p, eps), 1.0 - eps)
    u_sym = p
    u_llm = 1.0 - p
    return {
        "task_id": f"{prefix}{idx}",
        "family": family,
        "activation": h.astype(np.float32),
        "delta_utility": float(delta_u),
        "u_symbolic": float(u_sym),
        "u_llm": float(u_llm),
        "noise_sigma": float(noise_sigma),
    }
```

Oracle utility is `max(U_S, U_L)`:

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def benchmark_oracle_utility(task):
    return max(float(task.get("u_symbolic", 0.5)),
               float(task.get("u_llm", 0.5)))
```

## Environment 1 — Linear (Section 10A)

**Generative model.**

```
h_i ~ N(0, I_d)                         (d = 8 by default)
w*  = random unit vector  (or fixed w_star)
latent_score_i = h_i^T w* + ε_i
ΔU_i = latent_score_i
p_i = sigmoid(ΔU_i),  clipped to [0.05, 0.95]
U_S = p_i,   U_L = 1 - p_i
family_i = "S" if ΔU_i > 0.1
           "L" if ΔU_i < -0.1
           "tie" otherwise
```

`ε_i ~ N(0, noise_std^2)` with `noise_std = 0.1` by default.

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def make_linear_environment(n=200, dim=8, seed=0, w_star=None, noise_std=0.1):
    rng = np.random.default_rng(seed)
    if w_star is None:
        w = rng.normal(0, 1, size=dim)
        w = w / np.linalg.norm(w)
    H = rng.normal(0, 1, size=(n, dim))
    noise = rng.normal(0, noise_std, size=n)
    scores = H @ w + noise
    ...
    du = float(scores[i])
    family = "S" if du > 0.1 else ("L" if du < -0.1 else "tie")
```

**Utility construction.** `U_S = sigmoid(ΔU)`, `U_L = 1 - U_S` (shared formula
above).

**Purpose.** Test recovery of a linear preference boundary. The optimal
decision boundary is the hyperplane orthogonal to `w*`. Logistic should be
strong; centroid may be competitive depending on geometry.

**Expected method failure.** Centroid may be competitive (not a failure per
se) — this environment is the "linear is recoverable" baseline. It establishes
that a linear router can succeed when the true boundary is linear.

**Random-direction control.** A matched random unit direction `v` routes via
`p = sigmoid(h · v)`. Because the true boundary is a single hyperplane, some
random directions will partially solve it; the control distribution should be
broad, confirming the environment is non-trivial but linearly separable.

## Environment 2 — Near-tie / heteroskedastic (Section 10B)

**Generative model.**

```
h_i ~ N(0, I_d)
w* = random unit vector
70% of samples: near-ties,  |ΔU| ~ Uniform(0, small_gap)      small_gap = 0.05
30% of samples: decisive,    |ΔU| ~ Uniform(large_gap/2, large_gap)  large_gap = 1.5
sign_i ~ Uniform{-1, +1}
ΔU_i = sign_i * gap_i  +  0.3 * (h_i^T w*)
```

The small latent component `0.3 * (h_i^T w*)` makes the direction recoverable,
but the dominant signal is the near-tie/decisive mixture.

**Noise-dependent confidence.**

```
conf_i = min(1.0, |ΔU_i| / large_gap + 0.3)
noise_sigma_i = 1.0 - conf_i      (near-ties have lower confidence)
```

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def make_near_tie_environment(n=200, dim=8, seed=0, w_star=None,
                              small_frac=0.7, small_gap=0.05,
                              large_gap=1.5, noise_std=0.1):
    n_small = int(n * small_frac)
    n_large = n - n_small
    signs = rng.choice([-1.0, 1.0], size=n)
    gaps = np.concatenate([
        rng.uniform(0.0, small_gap, size=n_small),
        rng.uniform(large_gap * 0.5, large_gap, size=n_large),
    ])
    rng.shuffle(gaps)
    scores = signs * gaps
    scores = scores + 0.3 * (H @ w)
    ...
    conf = float(min(1.0, abs(du) / large_gap + 0.3))
    task = _build_task(i, family, H[i], du, prefix="nt",
                       noise_sigma=1.0 - conf)
    task["confidence"] = conf
```

**Utility construction.** Shared `U_S = sigmoid(ΔU)`, `U_L = 1 - U_S`.

**Purpose.** Test whether utility weighting beats equal weighting. 70% of
samples are low-information near-ties; 30% are decisive. An unweighted
learner wastes capacity on the near-ties, while a utility-weighted learner
concentrates on the decisive 30%.

**Expected method failure.** **Unweighted > weighted regret** (i.e., unweighted
is worse). The required scientific expectation is that the weighted method
reduces regret relative to the unweighted method under this controlled regime.
If unweighted beats weighted here, utility weighting has no value.

**Random-direction control.** Matched random directions project onto `h`. The
near-tie majority means most tasks have tiny `|ΔU|`, so a random direction
that happens to align with `w*` recovers only the weak `0.3 * (h·w*)`
component — the control confirms the signal is not trivially recoverable by
arbitrary directions.

## Environment 3 — Covariance / multimodal (Section 10C)

**Generative model.** Four clusters in `d = 8` dimensions with cluster
separation `a = cluster_sep` (default 2.0), each cluster sampled as
`N(center, 0.3^2 I)`:

```
Symbolic-preferred cluster 1:  center = (+a, 0, ..., 0)   ΔU = +1.5
Symbolic-preferred cluster 2:  center = (-a, 0, ..., 0)   ΔU = +1.5
LLM-preferred cluster 1:       center = (0, +a, ..., 0)   ΔU = -1.5
LLM-preferred cluster 2:       center = (0, -a, ..., 0)   ΔU = -1.5
Remainder (few):               center ≈ 0                 ΔU = 0.0  (near-ties)
```

Therefore `μ_S ≈ μ_L ≈ 0` and the centroid difference `v = μ_S - μ_L ≈ 0`.

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def make_multimodal_environment(n=200, dim=8, seed=0, cluster_sep=2.0):
    rng = np.random.default_rng(seed)
    n_per_cluster = n // 4
    # Symbolic-preferred cluster 1: (+a, 0, ...)
    for _ in range(n_per_cluster):
        h = rng.normal(0, 0.3, size=dim)
        h[0] = cluster_sep + h[0]
        tasks.append(_build_task(idx, "S", h, 1.5, prefix="mm"))
    # Symbolic-preferred cluster 2: (-a, 0, ...)
    ...
        h[0] = -cluster_sep + h[0]
    # LLM-preferred cluster 1: (0, +a, ...)
    ...
        h[1] = cluster_sep + h[1]
        tasks.append(_build_task(idx, "L", h, -1.5, prefix="mm"))
    # LLM-preferred cluster 2: (0, -a, ...)
    ...
        h[1] = -cluster_sep + h[1]
    # Remainder: a few near-ties at origin.
    ...
        tasks.append(_build_task(idx, "tie", h, 0.0, prefix="mm"))
```

**Utility construction.** Shared formula. Symbolic clusters get `ΔU = +1.5`
(`U_S > U_L`); LLM clusters get `ΔU = -1.5` (`U_L > U_S`); near-ties get
`ΔU = 0`.

**Purpose.** Demonstrate a failure mode of mean-difference steering. Because
the two symbolic clusters are at `+a` and `-a` along coordinate 0 (and the two
LLM clusters at `+a` and `-a` along coordinate 1), the class means cancel.

**Expected method failure.** **Centroid degrades materially.** The centroid
direction `v = μ_S - μ_L ≈ 0` has near-zero norm, so the signed-projection
router `p = sigmoid(h · v)` is near-uniform and routes essentially randomly. A
linear classifier may also fail depending on construction; this is acceptable.
Required: centroid performance degrades materially.

**Random-direction control.** A random direction cannot distinguish the
quadrant structure either (the class label depends on *which coordinate* is
large, not on a single linear projection). The control distribution should be
near the random-baseline regret, confirming the environment is hard for
projection-based methods.

## Environment 4 — Nonlinear XOR (Section 10D)

**Generative model.**

```
h_i ~ N(0, I_d)        (d = 8; only first two coords carry signal, rest are noise)
prefer_symbolic_i  iff  h_i,0 * h_i,1 > 0
ΔU_i = +gap  if prefer_symbolic
       -gap  otherwise            gap = 1.5
family_i = "S" if prefer_symbolic else "L"
```

The decision boundary is the XOR of the signs of the first two coordinates —
two quadrants prefer symbolic, two prefer LLM.

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def make_xor_environment(n=200, dim=8, seed=0, gap=1.5):
    rng = np.random.default_rng(seed)
    H = rng.normal(0, 1, size=(n, dim))
    for i in range(n):
        prefer_symbolic = H[i, 0] * H[i, 1] > 0
        du = gap if prefer_symbolic else -gap
        family = "S" if prefer_symbolic else "L"
        tasks.append(_build_task(i, family, H[i], du, prefix="xor"))
```

**Utility construction.** Shared formula with `ΔU = ±gap`. Because `|ΔU|` is
large and deterministic, `p_i` is clipped to `[0.05, 0.95]` so both backends
have non-trivial utility but the correct backend always wins.

**Purpose.** Detect when nonlinear routing is required. The XOR boundary is
not linearly separable, so a linear logistic router cannot solve it.

**Expected method failure.** **Logistic fails; MLP succeeds.** A linear
classifier `σ(w^T h + b)` cannot represent the XOR boundary. The small MLP
router (`Linear → GELU → Linear`) can. The diagnostic conclusion: if
`MLP >> logistic` on this environment, routing geometry is nonlinear.

**Random-direction control.** A single random direction is a linear projection
and therefore cannot solve XOR. The control distribution should sit at the
random-routing regret (≈ 0.5 misroute rate), confirming the environment is not
trivially solvable by projection.

## Random-direction control (Section 10E)

`random_direction_control(activations, delta_u, *, n_random=500, seed=0,
metric="regret")` generates `n_random` matched random directions (same norm = 1,
same dimension) and reports the distribution of the metric achieved by using
each direction as a signed-projection router:

```
v_j ~ N(0, I_d),   normalized to ||v_j|| = 1
scores_j = H @ v_j
pred_sym_j = scores_j > 0
pred_u_j   = where(pred_sym_j, U_S, U_L)
oracle_u   = max(U_S, U_L)

if metric == "regret":
    value_j = mean_i(max(0, oracle_u_i - pred_u_{j,i}))     # lower is better
if metric == "accuracy":
    correct_j = where(ΔU > 0, pred_sym_j, ~pred_sym_j)
    value_j   = mean_i(correct_{j,i})                        # higher is better
```

Returns the distribution statistics: `n_random`, `random_values`, `median`,
`mean`, `p05`, `p95`, `min`, `max`.

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def random_direction_control(activations, delta_u, *, n_random=500,
                             seed=0, metric="regret"):
    for _ in range(n_random):
        v = rng.normal(0, 1, size=d)
        v = v / np.linalg.norm(v)            # matched norm = 1
        scores = H @ v
        pred_sym = scores > 0
        pred_u = np.where(pred_sym, u_sym, u_llm)
        if metric == "regret":
            regrets = np.maximum(oracle_u - pred_u, 0.0)
            values.append(float(regrets.mean()))
        elif metric == "accuracy":
            correct = np.where(du > 0, pred_sym, ~pred_sym)
            values.append(float(correct.mean()))
    return {"n_random": ..., "median": ..., "mean": ..., "p05": ..., "p95": ...}
```

**Purpose.** Detect benchmarks where arbitrary random directions trivially
solve the task: if the median random metric is close to the best possible, the
benchmark is too easy. A learned direction must beat the median random
direction to claim it learned anything non-trivial. Use `n_random >= 500`
where compute permits and report the empirical p-value
`p = (1 + count(M_j >= M*)) / (N + 1)`.

## Execute / utility functions

The benchmark environments use the same task-dict shape as
`environment_synthetic`, so the existing pipeline works unchanged:

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
def benchmark_execute_fn(task, backend):
    # symbolic: u = task["u_symbolic"], correct = u > 0.5
    # llm:      u = task["u_llm"],      correct = u > 0.5
    return {"correct": ..., "quality": u, "latency_ms": ...,
            "compute_cost": ..., "risk": 0.0, "verifier_confidence": u}

def benchmark_utility(task, route):
    # abstain -> 0.0; symbolic -> u_symbolic; llm -> u_llm
```

## Registry

<ref_file>src/daph_learning/environment_benchmark.py</ref_snippet>
```python
ENVIRONMENT_REGISTRY = {
    "linear": make_linear_environment,
    "near_tie": make_near_tie_environment,
    "multimodal": make_multimodal_environment,
    "xor": make_xor_environment,
}
def make_environment(name, n=200, dim=8, seed=0):
    return ENVIRONMENT_REGISTRY[name](n=n, dim=dim, seed=seed)
```

## Summary table

| Environment   | Generative model                          | Purpose                          | Expected failure              |
|---------------|-------------------------------------------|----------------------------------|-------------------------------|
| linear        | `ΔU = h·w* + ε`                           | recover a linear boundary        | centroid may be competitive   |
| near_tie      | 70% near-tie / 30% decisive mixture       | test utility weighting value     | unweighted > weighted regret  |
| multimodal    | 4 clusters, `μ_S ≈ μ_L ≈ 0`              | break mean-difference geometry   | centroid degrades materially  |
| xor           | `prefer_S iff h_0·h_1 > 0`               | detect nonlinear routing need    | logistic fails; MLP succeeds  |
