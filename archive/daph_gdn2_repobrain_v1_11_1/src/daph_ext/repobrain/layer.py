from __future__ import annotations

import re
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .repo2lora import LoRAFactors, layer_to_group
from .adapter_context import get_active_adapter_context


class RepoLoRALinear(nn.Module):
    """Wrap an nn.Linear and apply generated repository LoRA factors dynamically.

    Supports either one repository adapter shared across the input batch or a
    batch of adapters aligned one-to-one with the leading input batch dimension.
    """

    def __init__(self, base_linear: nn.Linear, group_idx: int, module_key: str | None = None) -> None:
        super().__init__()
        if group_idx < 0:
            raise ValueError("group_idx must be >= 0")
        self.base_linear = base_linear
        self.group_idx = group_idx
        self.module_key = module_key
        self.current_factors: LoRAFactors | None = None
        self.active_scale: float = 1.0

    @property
    def in_features(self) -> int:
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        return self.base_linear.out_features

    def set_factors(self, factors: LoRAFactors | None, scale: float = 1.0) -> None:
        if scale < 0:
            raise ValueError("scale must be >= 0")
        if factors is not None:
            if not (0 <= self.group_idx < factors.groups):
                raise IndexError(self.group_idx)
            if factors.a.shape[-1] != self.in_features or factors.b.shape[-2] != self.out_features:
                raise ValueError("LoRA factor shape does not match wrapped linear layer")
        self.current_factors = factors
        self.active_scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base_linear(x)
        runtime = get_active_adapter_context()
        if runtime is not None and self.module_key is not None:
            factors = runtime.factors_by_module.get(self.module_key)
            active_scale = runtime.scale
        else:
            # Legacy single-request path. Production serving should use
            # use_adapter_context() instead of mutating module-global state.
            factors = self.current_factors
            active_scale = self.active_scale
        if factors is None or active_scale == 0.0:
            return base_output

        scaling = (factors.alpha / factors.rank) * active_scale
        if not factors.is_batched:
            a = factors.a[self.group_idx].to(device=x.device, dtype=x.dtype)
            b = factors.b[self.group_idx].to(device=x.device, dtype=x.dtype)
            return base_output + scaling * F.linear(F.linear(x, a), b)

        if x.ndim < 2:
            raise ValueError("batched RepoLoRA factors require an input with a batch dimension")

        runtime_indices = runtime.batch_indices if runtime is not None else None
        if runtime_indices is not None:
            if runtime_indices.numel() != x.shape[0]:
                raise ValueError(
                    "adapter batch_indices length must equal runtime input batch; "
                    f"got {runtime_indices.numel()} indices for batch {x.shape[0]}"
                )
            if int(runtime_indices.max().item()) >= factors.batch_size:
                raise IndexError(
                    "adapter batch_indices reference a factor row outside the generated batch"
                )
            indices = runtime_indices.to(device=factors.a.device, dtype=torch.long)
            a_source = factors.a.index_select(0, indices)
            b_source = factors.b.index_select(0, indices)
        elif x.shape[0] == factors.batch_size:
            a_source = factors.a
            b_source = factors.b
        elif x.shape[0] % factors.batch_size == 0:
            # Hugging Face generation commonly expands each source example into
            # contiguous beams/return sequences via repeat_interleave. Support
            # that common case automatically. Reordered/non-contiguous fan-out
            # must supply explicit batch_indices in use_adapter_context().
            expansion = x.shape[0] // factors.batch_size
            a_source = factors.a.repeat_interleave(expansion, dim=0)
            b_source = factors.b.repeat_interleave(expansion, dim=0)
        else:
            raise ValueError(
                "runtime input batch is incompatible with batched RepoLoRA factors; "
                f"got input batch {x.shape[0]} and factor batch size {factors.batch_size}. "
                "Provide explicit batch_indices via use_adapter_context() for reordered fan-out."
            )

        a = a_source[:, self.group_idx].to(device=x.device, dtype=x.dtype)  # [B,R,In]
        b = b_source[:, self.group_idx].to(device=x.device, dtype=x.dtype)  # [B,Out,R]
        # einsum works for [B,In] and [B,...,In] inputs while preserving all
        # non-feature dimensions. Flattening avoids fragile fixed-rank einsums.
        original_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1])
        hidden = torch.einsum("bni,bri->bnr", x_flat, a)
        delta = torch.einsum("bnr,bor->bno", hidden, b)
        delta = delta.reshape(*original_shape[:-1], self.out_features)
        return base_output + scaling * delta


def _set_child(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        elif isinstance(parent, nn.ModuleDict) and part in parent:
            parent = parent[part]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = replacement
    elif isinstance(parent, nn.ModuleDict) and leaf in parent:
        parent[leaf] = replacement
    else:
        setattr(parent, leaf, replacement)


def patch_repo_lora_linears(
    model: nn.Module,
    *,
    target_modules: tuple[str, ...],
    num_layers: int,
    groups: int,
    layer_pattern: str = r"(?:^|\.)(?:layers|h|blocks|layer)\.(\d+)(?:\.|$)",
    strict_targets: bool = True,
) -> dict[str, RepoLoRALinear]:
    """Patch target linears across common Transformer/SSM layer namespaces.

    Matches paths such as ``model.layers.0``, ``transformer.h.0``,
    ``blocks.0``, ``model.decoder.layers.0`` and ``encoder.layer.0``.
    """
    if not target_modules or len(set(target_modules)) != len(target_modules):
        raise ValueError("target_modules must be non-empty and unique")
    pattern = re.compile(layer_pattern)
    replacements: dict[str, RepoLoRALinear] = {}
    matched_suffixes: set[str] = set()
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.split(".")[-1] not in target_modules:
            continue
        match = pattern.search(name)
        if match is None:
            raise ValueError(f"cannot infer transformer layer index from module path: {name}")
        layer_idx = int(match.group(1))
        wrapper = RepoLoRALinear(module, layer_to_group(layer_idx, num_layers, groups), module_key=name.split(".")[-1])
        _set_child(model, name, wrapper)
        replacements[name] = wrapper
        matched_suffixes.add(name.split(".")[-1])

    if not replacements:
        raise ValueError(
            f"Zero linear layers matched target_modules={target_modules} using pattern {layer_pattern!r}. "
            "Verify the checkpoint's projection names and layer namespace."
        )
    missing_targets = set(target_modules) - matched_suffixes
    if strict_targets and missing_targets:
        raise ValueError(f"requested target modules were not found: {sorted(missing_targets)}")
    return replacements


def apply_generated_factors(
    wrappers: Mapping[str, RepoLoRALinear],
    factors_by_module: Mapping[str, LoRAFactors],
    *,
    scale: float = 1.0,
    strict: bool = True,
) -> None:
    """Apply generated factors to patched modules by target suffix.

    Strict mode prevents a missing generated factor from silently disabling a
    patched projection, which would otherwise invalidate an experiment arm.
    """
    missing = {name.split(".")[-1] for name in wrappers if name.split(".")[-1] not in factors_by_module}
    if strict and missing:
        raise KeyError(f"missing generated factors for patched targets: {sorted(missing)}")
    for name, wrapper in wrappers.items():
        module_type = name.split(".")[-1]
        wrapper.set_factors(factors_by_module.get(module_type), scale=scale)
