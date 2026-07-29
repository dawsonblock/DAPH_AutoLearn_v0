"""v0.3.10.1-alpha — redesigned synthetic benchmark environments (Section 10).

The v0.3.10 synthetic benchmark was too easy: one coordinate almost
directly encoded class membership, so a matched random direction could
achieve perfect regret. This module replaces it with four mathematical
environments, each designed so a specific method can fail:

============  ============================  ==============================
Environment   Purpose                       Expected failure
============  ============================  ==============================
linear        recover a linear boundary     centroid may be competitive
near_tie      test utility weighting        unweighted > weighted regret
multimodal    break mean-difference geom.   centroid degrades materially
xor           nonlinear boundary            logistic fails; MLP succeeds
============  ============================  ==============================

Plus a :func:`random_direction_control` that generates many matched
random directions (same norm, same dimension) and reports the
distribution of random performance, so a benchmark where arbitrary
random directions trivially solve it is detected.

Each environment returns a list of task dicts compatible with
:mod:`daph_learning.environment_synthetic` (``task_id``, ``family``,
``activation``, ``delta_utility``, ``u_symbolic``, ``u_llm``) so the
existing execute/utility functions work unchanged.

Utility construction (Section 10A)::

    p_i = sigmoid(ΔU_i)
    U_S = p_i
    U_L = 1 - p_i

This gives a bounded utility in ``[0, 1]`` and a ΔU that directly
reflects the latent preference score. For environments where the
preference is deterministic (linear, multimodal, xor), ``p_i`` is
clipped to ``[eps, 1-eps]`` so the utility gap is non-trivial but the
correct backend always has strictly higher utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .policy.types import Route


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_task(
    idx: int,
    family: str,
    h: np.ndarray,
    delta_u: float,
    *,
    prefix: str = "t",
    noise_sigma: float = 0.0,
) -> dict[str, Any]:
    """Build a synthetic task dict with bounded utilities.

    ``U_S = sigmoid(ΔU)``, ``U_L = 1 - U_S``. For deterministic
    preferences (|ΔU| large), the utilities are clipped to
    ``[eps, 1-eps]`` so both backends have non-trivial utility.
    """
    eps = 0.05
    p = float(_sigmoid(np.clip(delta_u, -10.0, 10.0)))
    p = min(max(p, eps), 1.0 - eps)
    u_sym = p
    u_llm = 1.0 - p
    return {
        "task_id": f"{prefix}{idx}",
        "family": family,
        "specification": f"{prefix}_task_{idx}",
        "expected": 42,
        "activation": h.astype(np.float32),
        "delta_utility": float(delta_u),
        "u_symbolic": float(u_sym),
        "u_llm": float(u_llm),
        "noise_sigma": float(noise_sigma),
    }


# ------------------------------------------------------------------
# 10A. Linear environment
# ------------------------------------------------------------------

def make_linear_environment(
    n: int = 200,
    dim: int = 8,
    seed: int = 0,
    w_star: np.ndarray | None = None,
    noise_std: float = 0.1,
) -> list[dict[str, Any]]:
    """Linear environment (Section 10A).

    ``h_i ~ N(0, I)``; ``latent_score_i = h_i^T w* + ε_i``;
    ``ΔU_i = latent_score_i``; ``U_S = sigmoid(ΔU)``, ``U_L = 1 - U_S``.

    Purpose: test recovery of a linear preference boundary. Logistic
    should be strong; centroid may be competitive depending on geometry.

    Parameters
    ----------
    n : int
    dim : int
    seed : int
    w_star : np.ndarray | None
        Fixed hidden vector (shape ``[dim]``). If None, a random unit
        vector is drawn (deterministic given ``seed``).
    noise_std : float
        Standard deviation of the additive noise on the latent score.
    """
    rng = np.random.default_rng(seed)
    if w_star is None:
        w = rng.normal(0, 1, size=dim)
        w = w / np.linalg.norm(w)
    else:
        w = np.asarray(w_star, dtype=np.float64)
    H = rng.normal(0, 1, size=(n, dim))
    noise = rng.normal(0, noise_std, size=n)
    scores = H @ w + noise
    tasks = []
    for i in range(n):
        du = float(scores[i])
        family = "S" if du > 0.1 else ("L" if du < -0.1 else "tie")
        tasks.append(_build_task(i, family, H[i], du, prefix="lin"))
    return tasks


# ------------------------------------------------------------------
# 10B. Near-tie / heteroskedastic environment
# ------------------------------------------------------------------

def make_near_tie_environment(
    n: int = 200,
    dim: int = 8,
    seed: int = 0,
    w_star: np.ndarray | None = None,
    small_frac: float = 0.7,
    small_gap: float = 0.05,
    large_gap: float = 1.5,
    noise_std: float = 0.1,
) -> list[dict[str, Any]]:
    """Near-tie / heteroskedastic environment (Section 10B).

    70% of samples have ``|ΔU| ~ small`` (low-information near-ties),
    30% have ``|ΔU| ~ large`` (decisive). Confidence is noise-dependent.

    Purpose: test whether utility weighting beats equal weighting.
    Required scientific expectation: weighted method should reduce
    regret relative to unweighted method under this controlled regime.

    Parameters
    ----------
    n : int
    dim : int
    seed : int
    w_star : np.ndarray | None
        Direction that determines which backend is preferred.
    small_frac : float
        Fraction of near-tie samples (default 0.7).
    small_gap : float
        ``|ΔU|`` scale for near-tie samples.
    large_gap : float
        ``|ΔU|`` scale for decisive samples.
    noise_std : float
        Additive noise on the hidden representation.
    """
    rng = np.random.default_rng(seed)
    if w_star is None:
        w = rng.normal(0, 1, size=dim)
        w = w / np.linalg.norm(w)
    else:
        w = np.asarray(w_star, dtype=np.float64)
    H = rng.normal(0, 1, size=(n, dim))
    n_small = int(n * small_frac)
    n_large = n - n_small
    signs = rng.choice([-1.0, 1.0], size=n)
    gaps = np.concatenate([
        rng.uniform(0.0, small_gap, size=n_small),
        rng.uniform(large_gap * 0.5, large_gap, size=n_large),
    ])
    rng.shuffle(gaps)
    scores = signs * gaps
    # Add a small latent component so the direction is recoverable.
    scores = scores + 0.3 * (H @ w)
    tasks = []
    for i in range(n):
        du = float(scores[i])
        family = "S" if du > 0.05 else ("L" if du < -0.05 else "tie")
        # Noise-dependent confidence: near-ties have lower confidence.
        conf = float(min(1.0, abs(du) / large_gap + 0.3))
        task = _build_task(i, family, H[i], du, prefix="nt",
                           noise_sigma=1.0 - conf)
        task["confidence"] = conf
        tasks.append(task)
    return tasks


# ------------------------------------------------------------------
# 10C. Covariance / multimodal environment
# ------------------------------------------------------------------

def make_multimodal_environment(
    n: int = 200,
    dim: int = 8,
    seed: int = 0,
    cluster_sep: float = 2.0,
) -> list[dict[str, Any]]:
    """Covariance / multimodal environment (Section 10C).

    Symbolic-preferred samples: two clusters at ``(+a, 0, ...)`` and
    ``(-a, 0, ...)``. LLM-preferred samples: two clusters at
    ``(0, +a, ...)`` and ``(0, -a, ...)``. Therefore
    ``μ_S ≈ μ_L ≈ 0`` and the centroid difference ``v ≈ 0``.

    Purpose: demonstrate a failure mode of mean-difference steering.
    A linear classifier may also fail depending on construction; this
    is acceptable. Required: centroid performance degrades materially.

    Parameters
    ----------
    n : int
    dim : int
    seed : int
    cluster_sep : float
        Cluster separation ``a``.
    """
    rng = np.random.default_rng(seed)
    n_per_cluster = n // 4
    n_remainder = n - 4 * n_per_cluster
    tasks = []
    idx = 0
    # Symbolic-preferred cluster 1: (+a, 0, ...)
    for _ in range(n_per_cluster):
        h = rng.normal(0, 0.3, size=dim)
        h[0] = cluster_sep + h[0]
        tasks.append(_build_task(idx, "S", h, 1.5, prefix="mm"))
        idx += 1
    # Symbolic-preferred cluster 2: (-a, 0, ...)
    for _ in range(n_per_cluster):
        h = rng.normal(0, 0.3, size=dim)
        h[0] = -cluster_sep + h[0]
        tasks.append(_build_task(idx, "S", h, 1.5, prefix="mm"))
        idx += 1
    # LLM-preferred cluster 1: (0, +a, ...)
    for _ in range(n_per_cluster):
        h = rng.normal(0, 0.3, size=dim)
        h[1] = cluster_sep + h[1]
        tasks.append(_build_task(idx, "L", h, -1.5, prefix="mm"))
        idx += 1
    # LLM-preferred cluster 2: (0, -a, ...)
    for _ in range(n_per_cluster):
        h = rng.normal(0, 0.3, size=dim)
        h[1] = -cluster_sep + h[1]
        tasks.append(_build_task(idx, "L", h, -1.5, prefix="mm"))
        idx += 1
    # Remainder: a few near-ties at origin.
    for _ in range(n_remainder):
        h = rng.normal(0, 0.1, size=dim)
        tasks.append(_build_task(idx, "tie", h, 0.0, prefix="mm"))
        idx += 1
    return tasks


# ------------------------------------------------------------------
# 10D. Nonlinear XOR-like environment
# ------------------------------------------------------------------

def make_xor_environment(
    n: int = 200,
    dim: int = 8,
    seed: int = 0,
    gap: float = 1.5,
) -> list[dict[str, Any]]:
    """Nonlinear XOR-like environment (Section 10D).

    ``h_1, h_2 ~ N(0, 1)`` (first two coords; rest are noise).
    Symbolic preferred iff ``h_1 * h_2 > 0``; LLM otherwise.

    Purpose: centroid fails, logistic fails, small MLP should succeed.
    This tells AutoLearn when nonlinear routing is required.

    Parameters
    ----------
    n : int
    dim : int
    seed : int
    gap : float
        ``|ΔU|`` magnitude for decisive samples.
    """
    rng = np.random.default_rng(seed)
    H = rng.normal(0, 1, size=(n, dim))
    tasks = []
    for i in range(n):
        prefer_symbolic = H[i, 0] * H[i, 1] > 0
        du = gap if prefer_symbolic else -gap
        family = "S" if prefer_symbolic else "L"
        tasks.append(_build_task(i, family, H[i], du, prefix="xor"))
    return tasks


# ------------------------------------------------------------------
# 10E. Random direction control
# ------------------------------------------------------------------

def random_direction_control(
    activations: np.ndarray,
    delta_u: np.ndarray,
    *,
    n_random: int = 500,
    seed: int = 0,
    metric: str = "regret",
    learned_metric_value: float | None = None,
) -> dict[str, Any]:
    """Random direction control (Section 10E / 26).

    Generate ``n_random`` matched random directions (same norm, same
    dimension) and report the distribution of the metric achieved by
    using each direction as a signed-projection router
    ``p = sigmoid(h · v / ||v||)``.

    This detects benchmarks where arbitrary random directions trivially
    solve the task: if the median random metric is close to the best
    possible, the benchmark is too easy.

    v0.3.10.1 — Section 26: if ``learned_metric_value`` is provided,
    compute the empirical significance ``p_emp``::

        p_emp = (1 + count(M_j >= M*)) / (N_random + 1)

    For regret (lower-is-better), the comparison is inverted:
    ``count(M_j <= M*)``. This is the random-direction qualification
    formula; do NOT use Gaussian z-score alone as primary significance
    evidence.

    Parameters
    ----------
    activations : np.ndarray  shape ``[N, D]``
    delta_u : np.ndarray  shape ``[N]``
    n_random : int
        Number of random directions (>= 500 when compute permits).
    seed : int
    metric : str
        ``"regret"`` (lower is better) or ``"accuracy"`` (higher is
        better).
    learned_metric_value : float | None
        The learned method's metric value M*. If provided, ``p_emp`` is
        computed.

    Returns
    -------
    dict
        With keys: ``n_random``, ``metric``, ``random_values`` (list),
        ``median``, ``mean``, ``p05``, ``p95``, ``min``, ``max``,
        ``p_emp`` (if learned_metric_value provided), ``learned_metric_value``.
    """
    rng = np.random.default_rng(seed)
    H = np.asarray(activations, dtype=np.float64)
    du = np.asarray(delta_u, dtype=np.float64)
    n, d = H.shape
    # Oracle utility per task: max(U_S, U_L) where U_S = sigmoid(ΔU).
    p_oracle = _sigmoid(du)
    u_sym = p_oracle
    u_llm = 1.0 - p_oracle
    oracle_u = np.maximum(u_sym, u_llm)
    values = []
    for _ in range(n_random):
        v = rng.normal(0, 1, size=d)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        v = v / norm  # matched norm = 1
        scores = H @ v
        # Route: symbolic if score > 0 else llm.
        pred_sym = scores > 0
        pred_u = np.where(pred_sym, u_sym, u_llm)
        if metric == "regret":
            regrets = np.maximum(oracle_u - pred_u, 0.0)
            values.append(float(regrets.mean()))
        elif metric == "accuracy":
            # Accuracy: did the random direction pick the higher-utility
            # backend? (symbolic iff ΔU > 0)
            correct = np.where(du > 0, pred_sym, ~pred_sym)
            values.append(float(correct.mean()))
        else:
            raise ValueError(f"unknown metric {metric!r}")
    values = np.array(values, dtype=np.float64)
    result = {
        "n_random": int(len(values)),
        "metric": metric,
        "random_values": values.tolist(),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    # v0.3.10.1 — Section 26: empirical significance.
    if learned_metric_value is not None:
        m_star = float(learned_metric_value)
        if metric == "regret":
            # Lower is better: count how many random directions are
            # at least as good (<= M*).
            count_at_least_as_good = int(np.sum(values <= m_star))
        else:
            # Higher is better: count how many random directions are
            # at least as good (>= M*).
            count_at_least_as_good = int(np.sum(values >= m_star))
        p_emp = (1 + count_at_least_as_good) / (len(values) + 1)
        result["learned_metric_value"] = m_star
        result["p_emp"] = float(p_emp)
        result["count_random_at_least_as_good"] = count_at_least_as_good
    return result


# ------------------------------------------------------------------
# Unified execute / utility functions for all environments
# ------------------------------------------------------------------

def benchmark_execute_fn(task: dict[str, Any], backend: str) -> dict[str, Any]:
    """Execute function for the redesigned benchmark environments.

    Returns the same dict shape as
    :func:`daph_learning.environment_synthetic.synthetic_execute_fn` so
    the existing experience pipeline works unchanged.
    """
    if backend == "symbolic":
        u = float(task.get("u_symbolic", 0.5))
        correct = u > 0.5
        return {
            "correct": bool(correct),
            "quality": u,
            "latency_ms": 10.0,
            "compute_cost": 0.01,
            "risk": 0.0,
            "verifier_confidence": u,
        }
    if backend == "llm":
        u = float(task.get("u_llm", 0.5))
        correct = u > 0.5
        return {
            "correct": bool(correct),
            "quality": u,
            "latency_ms": 100.0,
            "compute_cost": 0.1,
            "risk": 0.0,
            "verifier_confidence": u,
        }
    raise ValueError(f"unknown backend {backend!r}")


def benchmark_utility(task: dict[str, Any], route: Route | str) -> float:
    """Utility function for the redesigned benchmark environments.

    ``U = quality`` (the bounded utility stored in the task). Abstain
    returns 0.0 (consistent with
    :func:`daph_learning.environment_synthetic.synthetic_utility`).
    """
    if isinstance(route, Route):
        route = route.value
    if route == "abstain":
        return 0.0
    if route == "symbolic":
        return float(task.get("u_symbolic", 0.5))
    if route == "llm":
        return float(task.get("u_llm", 0.5))
    raise ValueError(f"unknown route {route!r}")


def benchmark_oracle_utility(task: dict[str, Any]) -> float:
    """Oracle utility: ``max(U_S, U_L)``."""
    return max(
        float(task.get("u_symbolic", 0.5)),
        float(task.get("u_llm", 0.5)),
    )


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

ENVIRONMENT_REGISTRY: dict[str, Any] = {
    "linear": make_linear_environment,
    "near_tie": make_near_tie_environment,
    "multimodal": make_multimodal_environment,
    "xor": make_xor_environment,
}


def make_environment(
    name: str,
    n: int = 200,
    dim: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Dispatch to the named environment constructor."""
    if name not in ENVIRONMENT_REGISTRY:
        raise ValueError(
            f"unknown environment {name!r}; expected one of "
            f"{list(ENVIRONMENT_REGISTRY)}")
    return ENVIRONMENT_REGISTRY[name](n=n, dim=dim, seed=seed)


__all__ = [
    "ENVIRONMENT_REGISTRY",
    "benchmark_execute_fn",
    "benchmark_oracle_utility",
    "benchmark_utility",
    "make_environment",
    "make_linear_environment",
    "make_multimodal_environment",
    "make_near_tie_environment",
    "make_xor_environment",
    "random_direction_control",
]
