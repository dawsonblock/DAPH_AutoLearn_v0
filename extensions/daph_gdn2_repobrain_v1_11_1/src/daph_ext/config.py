from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import math

import yaml


class MixerKind(str, Enum):
    ATTENTION = "attention"
    GDN2 = "gdn2"
    MAMBA = "mamba"
    CHEAP = "cheap"


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int = 1024
    num_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_layers", "intermediate_size", "num_attention_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class GDN2Config:
    enabled: bool = True
    every_n_layers: int = 2
    head_dim: int = 64
    num_heads: int = 16
    expand_v: float = 1.0
    mode: str = "chunk"
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    external_module: str = "lit_gpt.gdn2"
    external_class: str = "GatedDeltaNet2"

    def __post_init__(self) -> None:
        if self.every_n_layers <= 0 or self.head_dim <= 0 or self.num_heads <= 0:
            raise ValueError("every_n_layers, head_dim, and num_heads must be > 0")
        if not math.isfinite(self.expand_v) or self.expand_v <= 0:
            raise ValueError("expand_v must be finite and > 0")
        if self.mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("GDN2 mode must be 'chunk' or 'fused_recurrent'")
        if not self.external_module or not self.external_class:
            raise ValueError("external GDN2 module/class names must be non-empty")


@dataclass(frozen=True)
class Repo2LoRAConfig:
    repo_embedding_dim: int = 2048
    hidden_dim: int = 512
    rank: int = 8
    alpha: float = 16.0
    layer_groups: int = 4
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "o_proj", "down_proj")
    output_scale_init: float = 0.01

    def __post_init__(self) -> None:
        for name in ("repo_embedding_dim", "hidden_dim", "rank", "layer_groups"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and > 0")
        if not math.isfinite(self.output_scale_init) or self.output_scale_init <= 0:
            raise ValueError("output_scale_init must be finite and > 0")
        if not self.target_modules or any(not x for x in self.target_modules):
            raise ValueError("target_modules must contain non-empty names")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must not contain duplicates")


@dataclass(frozen=True)
class ControllerConfig:
    # Must remain None until calibrated on validation data.
    max_kl_drift: float | None = None
    max_validation_loss_regression: float = 0.02
    max_execution_regression: float = 0.01
    max_generic_regression: float = 0.01
    geometry_warning_cosine: float = 0.90
    geometry_warning_sign_conflict: float = 0.55
    scale_grid: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25)
    hard_fisher_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)
    soft_fisher_grid: tuple[float, ...] = (0.05, 0.10, 0.25, 0.5, 1.0, 2.0)
    min_retained_energy: float = 0.95
    max_reconstruction_error: float = 0.10
    require_calibrated_kl: bool = False

    def __post_init__(self) -> None:
        if self.max_kl_drift is not None and (not math.isfinite(self.max_kl_drift) or self.max_kl_drift < 0):
            raise ValueError("max_kl_drift must be None or finite and >= 0")
        for name in ("max_validation_loss_regression", "max_execution_regression", "max_generic_regression"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 <= self.geometry_warning_cosine <= 1:
            raise ValueError("geometry_warning_cosine must be in [0,1]")
        if not 0 <= self.geometry_warning_sign_conflict <= 1:
            raise ValueError("geometry_warning_sign_conflict must be in [0,1]")
        if not self.scale_grid or any((not math.isfinite(x) or x < 0) for x in self.scale_grid):
            raise ValueError("scale_grid must contain finite non-negative scales")
        if not self.hard_fisher_grid or any((not math.isfinite(x) or not 0 <= x <= 1) for x in self.hard_fisher_grid):
            raise ValueError("hard_fisher_grid must contain finite fractions in [0,1]")
        if not self.soft_fisher_grid or any((not math.isfinite(x) or x < 0) for x in self.soft_fisher_grid):
            raise ValueError("soft_fisher_grid must contain finite non-negative lambdas")
        if not 0 < self.min_retained_energy <= 1:
            raise ValueError("min_retained_energy must be in (0,1]")
        if not math.isfinite(self.max_reconstruction_error) or self.max_reconstruction_error < 0:
            raise ValueError("max_reconstruction_error must be finite and >= 0")
        if self.require_calibrated_kl and self.max_kl_drift is None:
            raise ValueError("require_calibrated_kl=True requires a calibrated max_kl_drift")


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False
    structural_dim: int = 768
    semantic_dim: int = 1024
    daph_hidden_size: int = 1024
    fusion_tokens: int = 64

    def __post_init__(self) -> None:
        for name in ("structural_dim", "semantic_dim", "daph_hidden_size", "fusion_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class DAPHIntegrationConfig:
    model: ModelShape = field(default_factory=ModelShape)
    gdn2: GDN2Config = field(default_factory=GDN2Config)
    repobrain: Repo2LoRAConfig = field(default_factory=Repo2LoRAConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DAPHIntegrationConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise TypeError("top-level YAML configuration must be a mapping")
        model = ModelShape(**raw.get("model", {}))
        gdn2 = GDN2Config(**raw.get("gdn2", {}))
        repo_raw: dict[str, Any] = dict(raw.get("repobrain", {}))
        if "target_modules" in repo_raw:
            repo_raw["target_modules"] = tuple(repo_raw["target_modules"])
        repobrain = Repo2LoRAConfig(**repo_raw)
        ctrl_raw: dict[str, Any] = dict(raw.get("controller", {}))
        if "scale_grid" in ctrl_raw:
            ctrl_raw["scale_grid"] = tuple(float(x) for x in ctrl_raw["scale_grid"])
        if "hard_fisher_grid" in ctrl_raw:
            ctrl_raw["hard_fisher_grid"] = tuple(float(x) for x in ctrl_raw["hard_fisher_grid"])
        if "soft_fisher_grid" in ctrl_raw:
            ctrl_raw["soft_fisher_grid"] = tuple(float(x) for x in ctrl_raw["soft_fisher_grid"])
        controller = ControllerConfig(**ctrl_raw)
        vision = VisionConfig(**raw.get("vision", {}))
        return cls(model=model, gdn2=gdn2, repobrain=repobrain, controller=controller, vision=vision)
