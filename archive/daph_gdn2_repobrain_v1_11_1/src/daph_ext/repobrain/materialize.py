from __future__ import annotations

from torch import Tensor

from .repo2lora import LoRAFactors


def materialize_group_delta(factors: LoRAFactors, group_idx: int) -> Tensor:
    """Return dense delta W for one layer group.

    Single factors return [Out,In]. Batched factors return [B,Out,In]. Dense
    materialization is intended for diagnostics/correction, not normal inference.
    """
    if not (0 <= group_idx < factors.groups):
        raise IndexError(group_idx)
    scale = factors.alpha / factors.rank
    if factors.is_batched:
        return scale * (
            factors.b[:, group_idx] @ factors.a[:, group_idx]
        )
    return scale * (factors.b[group_idx] @ factors.a[group_idx])
