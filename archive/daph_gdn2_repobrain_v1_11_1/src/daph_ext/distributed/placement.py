from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from daph_ext.repobrain.layer import RepoLoRALinear
from daph_ext.repobrain.repo2lora import LoRAFactors


@dataclass(frozen=True)
class ModulePlacement:
    module_name: str
    device: torch.device
    dtype: torch.dtype


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    params = list(module.parameters(recurse=True))
    if not params:
        raise ValueError("module has no parameters from which to infer placement")
    return params[0].device, params[0].dtype


class AdapterPlacementManager:
    """Materialize rank-local factor copies on each patched module's owner device."""

    def inspect(self, wrappers: Mapping[str, RepoLoRALinear]) -> list[ModulePlacement]:
        return [ModulePlacement(name, *_module_device_dtype(wrapper.base_linear)) for name, wrapper in wrappers.items()]

    def apply(
        self,
        wrappers: Mapping[str, RepoLoRALinear],
        factors_by_module: Mapping[str, LoRAFactors],
        *,
        scale: float = 1.0,
    ) -> None:
        cache: dict[tuple[str, str, torch.dtype], LoRAFactors] = {}
        for name, wrapper in wrappers.items():
            suffix = name.split(".")[-1]
            if suffix not in factors_by_module:
                raise KeyError(f"missing factors for {suffix}")
            device, dtype = _module_device_dtype(wrapper.base_linear)
            key = (suffix, str(device), dtype)
            local = cache.get(key)
            if local is None:
                source = factors_by_module[suffix]
                local = LoRAFactors(
                    a=source.a.to(device=device, dtype=dtype),
                    b=source.b.to(device=device, dtype=dtype),
                    alpha=source.alpha,
                )
                cache[key] = local
            wrapper.set_factors(local, scale=scale)
