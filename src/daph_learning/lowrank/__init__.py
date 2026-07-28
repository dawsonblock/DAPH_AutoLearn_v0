"""v0.3.10 — low-rank multi-vector controller (Sections 26-28, experimental).

Do NOT make this the first implementation step. Implement only after the
single-vector/logistic system passes its causal gates.

The low-rank controller learns a basis ``V = [v_1, ..., v_k]`` with
``V ∈ R^{D×K}`` and a controller ``α(h) = g_φ(h)`` that produces
context-dependent coefficients::

    h' = h + V α(h)

This is the v0.4 direction, provided here as an experimental module so
the architecture can grow from one latent direction + binary router to a
low-rank latent skill basis + learned controller.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


if _HAS_TORCH:

    class LowRankSteeringController(nn.Module):
        """Low-rank multi-vector steering controller (Section 26).

        ``V ∈ R^{D×K}`` is a learned basis of steering directions.
        ``α(h) = g_φ(h)`` is a learned controller that produces
        context-dependent coefficients. The intervention is::

            h' = h + V α(h)

        Parameters
        ----------
        hidden_dim : int
            Activation dimension ``D``.
        num_vectors : int
            Number of steering directions ``K`` (e.g. 2, 4, 8).
        controller_hidden : int
            Hidden width of the controller MLP.
        """

        def __init__(
            self,
            hidden_dim: int,
            num_vectors: int,
            controller_hidden: int = 64,
        ) -> None:
            super().__init__()
            if hidden_dim <= 0 or num_vectors <= 0:
                raise ValueError("hidden_dim and num_vectors must be > 0")
            self.hidden_dim = hidden_dim
            self.num_vectors = num_vectors
            self.vectors = nn.Parameter(
                torch.randn(hidden_dim, num_vectors) * 0.01
            )
            self.router = nn.Sequential(
                nn.Linear(hidden_dim, controller_hidden),
                nn.GELU(),
                nn.Linear(controller_hidden, num_vectors),
            )

        def forward(
            self, h: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply the low-rank intervention.

            Returns ``(h', α)`` where ``h' = h + V α(h)``.
            """
            if h.dim() == 1:
                h = h.unsqueeze(0)
            coeff = self.router(h)  # [N, K]
            delta = coeff @ self.vectors.T  # [N, D]
            return h + delta, coeff

        def vector_norms(self) -> torch.Tensor:
            """L2 norms of each steering vector."""
            return self.vectors.norm(dim=0)

        def clamp_vector_norms(self, max_norm: float) -> None:
            """Clamp each steering vector's L2 norm to
            ``max_norm`` (in-place)."""
            with torch.no_grad():
                norms = self.vectors.norm(dim=0, keepdim=True)
                scale = (max_norm / norms.clamp_min(1e-8)).clamp(max=1.0)
                self.vectors.mul_(scale)

    def orthogonality_loss(
        vectors: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Redundancy regularization for multi-vector mode (Section 27).

        ``L_orth = || V_hat^T V_hat - I ||_F^2``

        Use only as a regularizer. Do NOT enforce perfect orthogonality as
        a hard rule.
        """
        if vectors.dim() != 2:
            raise ValueError("vectors must be [D, K]")
        normalized = vectors / (
            vectors.norm(dim=0, keepdim=True).clamp_min(eps)
        )
        gram = normalized.T @ normalized
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).mean()


def interference_matrix(
    utility_fn,
    vectors: np.ndarray,
    *,
    h: np.ndarray | None = None,
) -> np.ndarray:
    """Empirical interaction matrix for multiple vectors (Section 28).

    For vectors ``j`` and ``k``::

        I_jk = U(v_j + v_k) - U(v_j) - U(v_k) + U(0)

    Interpretation:
        I_jk > 0  =>  synergy
        I_jk < 0  =>  destructive interference

    Parameters
    ----------
    utility_fn : callable
        ``v -> U(v)`` where ``v`` is a steering vector applied to a fixed
        activation ``h``. If ``h`` is None, ``utility_fn`` must already
        incorporate the base activation.
    vectors : np.ndarray
        Shape ``[D, K]`` steering vectors.
    h : np.ndarray | None
        Base activation (for reference; the utility_fn is responsible for
        applying the vector).
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("vectors must be [D, K]")
    k = vectors.shape[1]
    matrix = np.zeros((k, k), dtype=np.float64)
    u0 = utility_fn(np.zeros_like(vectors[:, 0]))
    u_solo = [utility_fn(vectors[:, j]) for j in range(k)]
    for j in range(k):
        for kk in range(k):
            u_jk = utility_fn(vectors[:, j] + vectors[:, kk])
            matrix[j, kk] = u_jk - u_solo[j] - u_solo[kk] + u0
    return matrix


__all__ = [
    "LowRankSteeringController",
    "interference_matrix",
    "orthogonality_loss",
]
