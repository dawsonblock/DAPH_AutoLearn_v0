from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

from daph_ext.mixers.cache import HybridSequenceCache

Serializer = Callable[[Any], Any]
Deserializer = Callable[[Any], Any]
CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class OpaqueCacheCodec:
    encode: Serializer
    decode: Deserializer


def _tensor_tree_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _tensor_tree_to_cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(v) for v in value]
    return value


def _restore_tree(value: Any, *, device=None, dtype=None) -> Any:
    if isinstance(value, Tensor):
        target_dtype = dtype if dtype is not None and (value.is_floating_point() or value.is_complex()) else None
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, dict):
        return {k: _restore_tree(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_tree(v, device=device, dtype=dtype) for v in value)
    if isinstance(value, list):
        return [_restore_tree(v, device=device, dtype=dtype) for v in value]
    return value


def cache_state_dict(cache: HybridSequenceCache, *, opaque_codec: OpaqueCacheCodec | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {"schema_version": CACHE_SCHEMA_VERSION, "attention_kv": {}, "recurrent_states": {}, "shared_recurrent_cache": None}
    state["attention_kv"] = {i: (k.detach().cpu(), v.detach().cpu()) for i, (k, v) in cache.attention_kv.items()}
    for i, value in cache.recurrent_states.items():
        if isinstance(value, (Tensor, dict, tuple, list)):
            state["recurrent_states"][i] = _tensor_tree_to_cpu(value)
        elif opaque_codec is not None:
            state["recurrent_states"][i] = {"__opaque__": True, "payload": opaque_codec.encode(value)}
        else:
            raise TypeError(f"opaque recurrent state at layer {i} requires an OpaqueCacheCodec")
    shared = cache.shared_recurrent_cache
    if shared is not None:
        if isinstance(shared, (Tensor, dict, tuple, list)):
            state["shared_recurrent_cache"] = _tensor_tree_to_cpu(shared)
        elif opaque_codec is not None:
            state["shared_recurrent_cache"] = {"__opaque__": True, "payload": opaque_codec.encode(shared)}
        else:
            raise TypeError("opaque shared recurrent cache requires an OpaqueCacheCodec")
    return state


def load_cache_state_dict(state: dict[str, Any], *, opaque_codec: OpaqueCacheCodec | None = None, device=None, dtype=None) -> HybridSequenceCache:
    version = int(state.get("schema_version", 1))
    if version not in (1, CACHE_SCHEMA_VERSION):
        raise ValueError(f"unsupported cache schema_version={version}")
    cache = HybridSequenceCache()
    cache.attention_kv = {int(i): tuple(_restore_tree(v, device=device, dtype=dtype)) for i, v in state.get("attention_kv", {}).items()}
    for key, value in state.get("recurrent_states", {}).items():
        if isinstance(value, dict) and value.get("__opaque__"):
            if opaque_codec is None:
                raise TypeError("opaque cache payload requires codec")
            value = opaque_codec.decode(value["payload"])
        else:
            value = _restore_tree(value, device=device, dtype=dtype)
        cache.recurrent_states[int(key)] = value
    shared = state.get("shared_recurrent_cache")
    if isinstance(shared, dict) and shared.get("__opaque__"):
        if opaque_codec is None:
            raise TypeError("opaque shared cache payload requires codec")
        shared = opaque_codec.decode(shared["payload"])
    else:
        shared = _restore_tree(shared, device=device, dtype=dtype)
    cache.shared_recurrent_cache = shared
    return cache


def evolution_checkpoint(module, hidden_state: Tensor) -> dict[str, Any]:
    if not isinstance(hidden_state, Tensor) or hidden_state.ndim not in (1, 2):
        raise ValueError("RepoEvolution hidden_state must be a rank-1 or rank-2 Tensor")
    return {"schema_version": 1, "module_state": {k: v.detach().cpu() for k, v in module.state_dict().items()}, "hidden_state": hidden_state.detach().cpu()}


def restore_evolution_checkpoint(module, payload: dict[str, Any], *, device=None, dtype=None) -> Tensor:
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("unsupported RepoEvolution checkpoint schema")
    module.load_state_dict(payload["module_state"])
    hidden = payload["hidden_state"]
    if not isinstance(hidden, Tensor):
        raise TypeError("checkpoint hidden_state must be a Tensor")
    target_dtype = dtype if dtype is not None and (hidden.is_floating_point() or hidden.is_complex()) else None
    return hidden.to(device=device, dtype=target_dtype)
