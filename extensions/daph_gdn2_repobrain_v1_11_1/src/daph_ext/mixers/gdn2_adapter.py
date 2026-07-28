from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from daph_ext.config import GDN2Config
from daph_ext.mixers.base import MixerOutput


def _move_state(value: Any, *, device, dtype) -> Any:
    """Best-effort migration for tensor/container caches without assuming a cache class."""
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype if (value.is_floating_point() or value.is_complex()) else None)
    if isinstance(value, Mapping):
        return {k: _move_state(v, device=device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_state(v, device=device, dtype=dtype) for v in value)
    if isinstance(value, list):
        return [_move_state(v, device=device, dtype=dtype) for v in value]
    # External FLA/Transformers cache objects own their own placement semantics.
    # Do not blindly call .to(): several mutable cache classes either lack it or
    # return a new incompatible wrapper.
    return value


class ExternalGDN2Adapter(nn.Module):
    """Bind DAPH to an externally installed GDN2 implementation.

    The NVIDIA GDN2 reference integration currently returns
    ``(hidden_states, None, past_key_values)``. v1.5 handles that three-tuple
    explicitly instead of incorrectly treating element 1 as recurrent state.
    Generic two-tuples remain supported for alternate implementations.
    """

    def __init__(self, hidden_size: int, cfg: GDN2Config, layer_idx: int) -> None:
        super().__init__()
        module = importlib.import_module(cfg.external_module)
        cls = getattr(module, cfg.external_class)
        self.impl = cls(
            hidden_size=hidden_size,
            head_dim=cfg.head_dim,
            num_heads=cfg.num_heads,
            expand_v=cfg.expand_v,
            mode=cfg.mode,
            use_short_conv=cfg.use_short_conv,
            allow_neg_eigval=cfg.allow_neg_eigval,
            layer_idx=layer_idx,
        )
        sig = inspect.signature(self.impl.forward)
        self._forward_params = set(sig.parameters.keys())
        self._accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in sig.parameters.values()
        )

    def _supports(self, name: str) -> bool:
        return name in self._forward_params or self._accepts_kwargs

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        recurrent_state: Any | None = None,
        use_cache: bool = False,
    ) -> MixerOutput:
        if hidden_states.ndim != 3:
            raise ValueError("GDN2 hidden_states must be [B,T,H]")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[:2] != hidden_states.shape[:2]:
                raise ValueError("attention_mask must be [B,T] and align with hidden_states")
            attention_mask = attention_mask.to(device=hidden_states.device)

        recurrent_state = _move_state(
            recurrent_state, device=hidden_states.device, dtype=hidden_states.dtype
        )

        kwargs: dict[str, Any] = {}
        if attention_mask is not None and self._supports("attention_mask"):
            kwargs["attention_mask"] = attention_mask

        if recurrent_state is not None:
            if "past_key_values" in self._forward_params:
                kwargs["past_key_values"] = recurrent_state
            elif "recurrent_state" in self._forward_params:
                kwargs["recurrent_state"] = recurrent_state
            elif self._accepts_kwargs:
                # Matches the current upstream GDN2/FLA cache contract.
                kwargs["past_key_values"] = recurrent_state

        if self._supports("use_cache"):
            kwargs["use_cache"] = use_cache

        out = self.impl(hidden_states, **kwargs)
        if isinstance(out, Tensor):
            return MixerOutput(hidden_states=out)
        if isinstance(out, tuple):
            if not out:
                raise TypeError("external GDN2 returned an empty tuple")
            if not isinstance(out[0], Tensor):
                raise TypeError("external GDN2 tuple element 0 must be hidden-state Tensor")
            # Current NVIDIA GDN2 contract: (o, None, past_key_values).
            state = out[2] if len(out) >= 3 else (out[1] if len(out) >= 2 else None)
            return MixerOutput(hidden_states=out[0], recurrent_state=state)
        if hasattr(out, "hidden_states"):
            state = getattr(out, "recurrent_state", getattr(out, "past_key_values", None))
            return MixerOutput(hidden_states=out.hidden_states, recurrent_state=state)
        if hasattr(out, "last_hidden_state"):
            state = getattr(out, "recurrent_state", getattr(out, "past_key_values", None))
            return MixerOutput(hidden_states=out.last_hidden_state, recurrent_state=state)
        raise TypeError(f"Unsupported external GDN2 output type: {type(out)!r}")
