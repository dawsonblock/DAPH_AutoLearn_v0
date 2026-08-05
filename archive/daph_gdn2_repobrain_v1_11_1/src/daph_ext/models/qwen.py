from __future__ import annotations

import re
from collections import defaultdict

from torch import nn

from daph_ext.repobrain.repo2lora import ModuleShape


_QWEN_LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)")


class QwenArchitectureAdapter:
    """Checkpoint-introspecting adapter for Qwen-style decoder models.

    The class has no transformers dependency; it works with any nn.Module that
    exposes the usual Qwen config/module names.
    """

    model_type = "qwen"
    _KNOWN_TARGETS = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    def num_layers(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(config, attr, None) if config is not None else None
            if isinstance(value, int) and value > 0:
                return value
        indices = [self.layer_index(name) for name, _ in model.named_modules()]
        found = [idx for idx in indices if idx is not None]
        if not found:
            raise ValueError("could not infer Qwen layer count from config or module paths")
        return max(found) + 1

    def hidden_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attr, None) if config is not None else None
            if isinstance(value, int) and value > 0:
                return value
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                return module.in_features
        raise ValueError("could not infer hidden size")

    def target_linear_names(self, model: nn.Module) -> tuple[str, ...]:
        available = {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        }
        found = tuple(name for name in self._KNOWN_TARGETS if name in available)
        if not found:
            raise ValueError("no Qwen projection names found in checkpoint")
        return found

    def layer_index(self, module_name: str) -> int | None:
        match = _QWEN_LAYER_RE.search(module_name)
        return int(match.group(1)) if match else None

    def module_shapes(self, model: nn.Module) -> dict[str, ModuleShape]:
        shapes: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            suffix = name.rsplit(".", 1)[-1]
            if suffix in self._KNOWN_TARGETS and self.layer_index(name) is not None:
                shapes[suffix].add((module.in_features, module.out_features))
        result: dict[str, ModuleShape] = {}
        for suffix, variants in shapes.items():
            if len(variants) != 1:
                raise ValueError(f"inconsistent shapes for {suffix}: {sorted(variants)}")
            in_features, out_features = next(iter(variants))
            result[suffix] = ModuleShape(in_features=in_features, out_features=out_features)
        if not result:
            raise ValueError("no patchable Qwen linear modules discovered")
        return result
