from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class CacheTensor:
    """Tensor plus explicit cache semantics for safe beam reordering."""
    tensor: Tensor
    batch_axis: int | None = None

    def __post_init__(self) -> None:
        if self.batch_axis is not None and not (-self.tensor.ndim <= self.batch_axis < self.tensor.ndim):
            raise ValueError("batch_axis is out of range for cache tensor")


def _move_tensor(t: Tensor, *, device=None, dtype=None) -> Tensor:
    """Move tensors without corrupting integer/bool cache metadata dtypes."""
    target_dtype = dtype if dtype is not None and (t.is_floating_point() or t.is_complex()) else None
    return t.to(device=device, dtype=target_dtype)


def _map_state(value: Any, fn) -> Any:
    if isinstance(value, CacheTensor):
        return CacheTensor(fn(value.tensor), value.batch_axis)
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_state(v, fn) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_state(v, fn) for v in value)
    if isinstance(value, list):
        return [_map_state(v, fn) for v in value]
    return value


@dataclass
class HybridSequenceCache:
    """Unified cache for attention KV pairs and recurrent mixer state.

    Tensor/container states support migration, detaching, cloning and guarded
    batch reordering. Opaque external cache objects remain owned by their
    framework and must use framework-native placement/reorder methods.
    """

    attention_kv: dict[int, tuple[Tensor, Tensor]] = field(default_factory=dict)
    recurrent_states: dict[int, Any] = field(default_factory=dict)
    shared_recurrent_cache: Any | None = None

    def reset(self) -> None:
        self.attention_kv.clear()
        self.recurrent_states.clear()
        self.shared_recurrent_cache = None

    def set_attention(self, layer_idx: int, key: Tensor, value: Tensor) -> None:
        if layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")
        if key.ndim == 0 or value.ndim == 0 or key.shape[0] != value.shape[0]:
            raise ValueError("attention key/value must have matching batch dimensions")
        self.attention_kv[layer_idx] = (key, value)

    def get_attention(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        return self.attention_kv.get(layer_idx)

    def set_recurrent(self, layer_idx: int, state: Any) -> None:
        if layer_idx < 0:
            raise ValueError("layer_idx must be >= 0")
        self.recurrent_states[layer_idx] = state

    def get_recurrent(self, layer_idx: int) -> Any | None:
        return self.recurrent_states.get(layer_idx)

    def set_shared_recurrent_cache(self, cache: Any | None) -> None:
        self.shared_recurrent_cache = cache

    def get_shared_recurrent_cache(self) -> Any | None:
        return self.shared_recurrent_cache

    def to(self, *, device=None, dtype=None) -> "HybridSequenceCache":
        self.attention_kv = {
            i: (_move_tensor(k, device=device, dtype=dtype), _move_tensor(v, device=device, dtype=dtype))
            for i, (k, v) in self.attention_kv.items()
        }
        self.recurrent_states = {
            i: _map_state(state, lambda x: _move_tensor(x, device=device, dtype=dtype))
            for i, state in self.recurrent_states.items()
        }
        if isinstance(self.shared_recurrent_cache, (Tensor, dict, tuple, list)):
            self.shared_recurrent_cache = _map_state(
                self.shared_recurrent_cache,
                lambda x: _move_tensor(x, device=device, dtype=dtype),
            )
        return self

    def detach(self) -> "HybridSequenceCache":
        self.attention_kv = {i: (k.detach(), v.detach()) for i, (k, v) in self.attention_kv.items()}
        self.recurrent_states = {i: _map_state(state, Tensor.detach) for i, state in self.recurrent_states.items()}
        if isinstance(self.shared_recurrent_cache, (Tensor, dict, tuple, list)):
            self.shared_recurrent_cache = _map_state(self.shared_recurrent_cache, Tensor.detach)
        return self

    def clone(self) -> "HybridSequenceCache":
        out = HybridSequenceCache()
        out.attention_kv = {i: (k.clone(), v.clone()) for i, (k, v) in self.attention_kv.items()}
        out.recurrent_states = {
            i: _map_state(state, Tensor.clone) if isinstance(state, (Tensor, dict, tuple, list)) else deepcopy(state)
            for i, state in self.recurrent_states.items()
        }
        if isinstance(self.shared_recurrent_cache, (Tensor, dict, tuple, list)):
            out.shared_recurrent_cache = _map_state(self.shared_recurrent_cache, Tensor.clone)
        elif self.shared_recurrent_cache is not None:
            out.shared_recurrent_cache = deepcopy(self.shared_recurrent_cache)
        return out

    def reorder_batch(self, beam_idx: Tensor, *, batch_size: int | None = None) -> "HybridSequenceCache":
        """Reorder only tensors whose leading dimension is the cache batch.

        ``batch_size`` should be supplied for nested tensor caches so sequence
        metadata or position tensors are not accidentally reordered as if they
        were batch-major. Opaque/shared caches require native framework APIs.
        """
        if beam_idx.ndim != 1 or beam_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("beam_idx must be a 1-D integer Tensor")
        if beam_idx.numel() == 0:
            raise ValueError("beam_idx must be non-empty")
        if self.shared_recurrent_cache is not None:
            raise TypeError("shared recurrent cache requires framework-native beam reorder")

        inferred = batch_size
        if inferred is None and self.attention_kv:
            inferred = next(iter(self.attention_kv.values()))[0].shape[0]
        if inferred is None:
            raise ValueError("batch_size is required when no attention cache is available")
        if inferred <= 0 or int(beam_idx.max().item()) >= inferred or int(beam_idx.min().item()) < 0:
            raise IndexError("beam_idx contains an out-of-range batch index")

        def reorder_tensor(x: Tensor) -> Tensor:
            # Backward-compatible heuristic for untyped tensors. Prefer
            # CacheTensor(batch_axis=...) for production cache schemas.
            if x.ndim > 0 and x.shape[0] == inferred:
                return x.index_select(0, beam_idx.to(x.device))
            return x

        def reorder_state(value: Any) -> Any:
            if isinstance(value, CacheTensor):
                if value.batch_axis is None:
                    return value
                axis = value.batch_axis % value.tensor.ndim
                if value.tensor.shape[axis] != inferred:
                    raise ValueError("typed cache batch axis does not match batch_size")
                return CacheTensor(
                    value.tensor.index_select(axis, beam_idx.to(value.tensor.device)),
                    value.batch_axis,
                )
            if isinstance(value, Tensor):
                return reorder_tensor(value)
            if isinstance(value, dict):
                return {k: reorder_state(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(reorder_state(v) for v in value)
            if isinstance(value, list):
                return [reorder_state(v) for v in value]
            return value

        self.attention_kv = {
            i: (reorder_tensor(k), reorder_tensor(v)) for i, (k, v) in self.attention_kv.items()
        }
        reordered: dict[int, Any] = {}
        for i, state in self.recurrent_states.items():
            if isinstance(state, (CacheTensor, Tensor, dict, tuple, list)):
                reordered[i] = reorder_state(state)
            else:
                raise TypeError(f"cannot generically reorder opaque recurrent cache at layer {i}")
        self.recurrent_states = reordered
        return self
