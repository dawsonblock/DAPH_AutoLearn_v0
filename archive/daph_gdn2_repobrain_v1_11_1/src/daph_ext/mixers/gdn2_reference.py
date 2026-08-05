from __future__ import annotations

import torch
from torch import Tensor


def gdn2_recurrence_reference(
    k: Tensor,
    v: Tensor,
    erase: Tensor,
    write: Tensor,
    log_decay: Tensor,
    initial_state: Tensor | None = None,
    *,
    q: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Slow transparent reference for the core GDN2 recurrence.

    Shapes:
        q,k:       [B,T,H,K]
        v:         [B,T,H,V]
        erase:     [B,T,H,K]
        write:     [B,T,H,V]
        log_decay: [B,T,H,K]
        state:     [B,H,K,V]

    Per token, matching the upstream recurrent-kernel documentation::

        S <- Diag(exp(g)) S
        v_new <- (w * v) - (b * k)^T S
        S <- S + k v_new^T
        o <- S^T q

    ``q`` defaults to ``k`` only for backward compatibility with the v1.x
    scaffold. Kernel qualification should pass the actual query tensor.
    """
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must be [B,T,H,D]")
    if erase.shape != k.shape or write.shape != v.shape or log_decay.shape != k.shape:
        raise ValueError("erase/log_decay must match k; write must match v")
    if q is None:
        q = k
    if q.shape != k.shape:
        raise ValueError("q must match k shape")
    if v.shape[:3] != k.shape[:3]:
        raise ValueError("q/k/v batch, sequence, and head dimensions must match")
    for name, tensor in {"q": q, "k": k, "v": v, "erase": erase, "write": write, "log_decay": log_decay}.items():
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if tensor.device != k.device:
            raise ValueError("all recurrence tensors must be on the same device")

    bsz, seqlen, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(bsz, heads, key_dim, value_dim, dtype=k.dtype, device=k.device)
    else:
        if initial_state.shape != (bsz, heads, key_dim, value_dim):
            raise ValueError("initial_state has wrong shape")
        state = initial_state.to(device=k.device, dtype=k.dtype).clone()

    outputs: list[Tensor] = []
    for t in range(seqlen):
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        bt = erase[:, t]
        wt = write[:, t]
        decay = torch.exp(log_decay[:, t])

        state = state * decay.unsqueeze(-1)
        gated_read = torch.einsum("bhk,bhkv->bhv", bt * kt, state)
        v_new = (wt * vt) - gated_read
        state = state + kt.unsqueeze(-1) * v_new.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", qt, state))

    return torch.stack(outputs, dim=1), state


def relative_l2(a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    if a.shape != b.shape:
        raise ValueError("relative_l2 inputs must have identical shapes")
    return torch.linalg.vector_norm((a - b).float()) / (torch.linalg.vector_norm(b.float()) + eps)
