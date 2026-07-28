from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
from torch import Tensor

from .repo2lora import LoRAFactors


@dataclass(frozen=True)
class AdapterRuntimeContext:
    factors_by_module: Mapping[str, LoRAFactors]
    scale: float = 1.0
    batch_indices: Tensor | None = None

    def __post_init__(self) -> None:
        if self.scale < 0:
            raise ValueError("adapter scale must be >= 0")
        if self.batch_indices is not None:
            if self.batch_indices.ndim != 1:
                raise ValueError("batch_indices must be a rank-1 tensor")
            if self.batch_indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("batch_indices must contain integer indices")
            if self.batch_indices.numel() == 0:
                raise ValueError("batch_indices must be non-empty")
            if bool((self.batch_indices < 0).any().item()):
                raise ValueError("batch_indices must be >= 0")


_ACTIVE_ADAPTER: ContextVar[AdapterRuntimeContext | None] = ContextVar(
    "daph_active_adapter", default=None
)


def get_active_adapter_context() -> AdapterRuntimeContext | None:
    return _ACTIVE_ADAPTER.get()


@contextmanager
def use_adapter_context(
    factors_by_module: Mapping[str, LoRAFactors],
    *,
    scale: float = 1.0,
    batch_indices: Tensor | None = None,
) -> Iterator[AdapterRuntimeContext]:
    """Bind generated factors to the current async/thread context only.

    ``batch_indices`` optionally maps each runtime batch row back to the
    repository-adapter row that should be used. This makes beam expansion,
    ``num_return_sequences`` and explicit micro-batch fan-out safe without
    mutating the generated factors themselves. When it is omitted, wrappers
    retain exact-batch behavior plus a conservative repeat-interleave fallback
    for common Hugging Face generation expansion.
    """
    ctx = AdapterRuntimeContext(
        factors_by_module=factors_by_module,
        scale=scale,
        batch_indices=batch_indices,
    )
    token = _ACTIVE_ADAPTER.set(ctx)
    try:
        yield ctx
    finally:
        _ACTIVE_ADAPTER.reset(token)
