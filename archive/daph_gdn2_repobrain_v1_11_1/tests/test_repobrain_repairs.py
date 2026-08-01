from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from daph_ext.config import ControllerConfig, GDN2Config, Repo2LoRAConfig
from daph_ext.geometry.fisher import empirical_fisher_diagonal
from daph_ext.mixers.gdn2_adapter import ExternalGDN2Adapter
from daph_ext.probes.base import ProbeResult
from daph_ext.probes.controller import AdapterAction, AdapterController
from daph_ext.repobrain.layer import RepoLoRALinear, patch_repo_lora_linears
from daph_ext.repobrain.materialize import materialize_group_delta
from daph_ext.repobrain.refactor import project_and_refactor_factors
from daph_ext.repobrain.repo2lora import LoRAFactors, ModuleShape, Repo2LoRALite
from daph_ext.vision.fusion import StructuralSemanticFusion


class CustomHFBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(16, 16)


class CustomHFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict({"h": nn.ModuleList([CustomHFBlock()])})


def test_layer_pattern_matches_custom_paths() -> None:
    model = CustomHFModel()
    wrappers = patch_repo_lora_linears(
        model,
        target_modules=("q_proj",),
        num_layers=1,
        groups=1,
    )
    assert list(wrappers) == ["transformer.h.0.q_proj"]
    assert isinstance(model.transformer["h"][0].q_proj, RepoLoRALinear)


def test_batched_repo_hypernetwork_and_runtime_layer() -> None:
    cfg = Repo2LoRAConfig(
        repo_embedding_dim=32,
        hidden_dim=16,
        rank=4,
        alpha=8.0,
        layer_groups=2,
        target_modules=("q_proj",),
    )
    shapes = {"q_proj": ModuleShape(16, 16)}
    hyper = Repo2LoRALite(cfg, shapes)
    out = hyper(torch.randn(3, 32))["q_proj"]
    assert out.a.shape == (3, 2, 4, 16)
    assert out.b.shape == (3, 2, 16, 4)

    layer = RepoLoRALinear(nn.Linear(16, 16), group_idx=1)
    layer.set_factors(out)
    y = layer(torch.randn(3, 5, 16))
    assert y.shape == (3, 5, 16)


def test_single_repo_contract_remains_unbatched() -> None:
    cfg = Repo2LoRAConfig(
        repo_embedding_dim=32,
        hidden_dim=16,
        rank=4,
        alpha=8.0,
        layer_groups=2,
        target_modules=("q_proj",),
    )
    factors = Repo2LoRALite(cfg, {"q_proj": ModuleShape(16, 16)})(torch.randn(1, 32))["q_proj"]
    assert factors.a.shape == (2, 4, 16)
    assert factors.b.shape == (2, 16, 4)


def test_batched_materialize_and_refactor() -> None:
    factors = LoRAFactors(
        a=torch.randn(2, 2, 3, 5),
        b=torch.randn(2, 2, 7, 3),
        alpha=6.0,
    )
    before = materialize_group_delta(factors, 1)
    assert before.shape == (2, 7, 5)
    ref = torch.randn(7, 5)
    updated = project_and_refactor_factors(factors, 1, ref, strength=0.5)
    after = materialize_group_delta(updated, 1)
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_visual_fusion_resampling() -> None:
    fusion = StructuralSemanticFusion(semantic_dim=16, structural_dim=8, hidden_size=12)
    semantic = torch.randn(2, 256, 16)
    structural = torch.randn(2, 196, 8)
    out = fusion(semantic, structural)
    assert out.shape == (2, 256, 12)


def test_fisher_with_list_and_tuple_inputs() -> None:
    model = nn.Linear(4, 2)
    batch = {
        "x": torch.randn(4, 4),
        "labels": ["a", "b", "c", "d"],
        "paths": ("p0", "p1", "p2", "p3"),
    }
    observed: list[tuple[int, int]] = []

    def loss_fn(m: nn.Module, b: dict) -> torch.Tensor:
        observed.append((len(b["labels"]), len(b["paths"])))
        return m(b["x"]).sum()

    fisher = empirical_fisher_diagonal(model, [batch], loss_fn=loss_fn, per_sample=True)
    assert fisher.samples == 4
    assert observed == [(1, 1)] * 4


def test_controller_recognizes_exec_and_val_ce() -> None:
    cfg = ControllerConfig(max_execution_regression=0.01, max_validation_loss_regression=0.02)
    controller = AdapterController(cfg)
    rejected = controller.decide([ProbeResult(name="exec_pass_rate", baseline=1.0, candidate=0.9, higher_is_better=True)])
    assert rejected.action == AdapterAction.REJECT

    scaled = controller.decide([ProbeResult(name="val_ce", baseline=1.0, candidate=1.05, higher_is_better=False)])
    assert scaled.action == AdapterAction.EVALUATE_CORRECTIONS
    assert scaled.requires_revalidation


def test_external_gdn2_passes_state_through_var_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGDN2(nn.Module):
        last_kwargs: dict | None = None

        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(self, x: torch.Tensor, **kwargs: object):
            FakeGDN2.last_kwargs = dict(kwargs)
            return x, kwargs.get("past_key_values")

    monkeypatch.setattr(
        "daph_ext.mixers.gdn2_adapter.importlib.import_module",
        lambda _: SimpleNamespace(GatedDeltaNet2=FakeGDN2),
    )
    cfg = GDN2Config(external_module="fake", external_class="GatedDeltaNet2")
    adapter = ExternalGDN2Adapter(hidden_size=16, cfg=cfg, layer_idx=0)
    x = torch.randn(2, 4, 16)
    state = torch.randn(2, 3, 4)
    mask = torch.ones(2, 4, dtype=torch.long)
    out = adapter(x, recurrent_state=state, attention_mask=mask, use_cache=True)
    assert out.recurrent_state is not None
    assert FakeGDN2.last_kwargs is not None
    assert "past_key_values" in FakeGDN2.last_kwargs
    assert "attention_mask" in FakeGDN2.last_kwargs
    assert FakeGDN2.last_kwargs["use_cache"] is True
