from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from daph_ext.config import ControllerConfig
from daph_ext.experiments.provenance import DataOrigin, ExperimentProvenance
from daph_ext.geometry.corrections_v14 import (
    hard_fisher_topk,
    normalize_fisher,
    soft_fisher_attenuation,
    svd_retained_energy,
)
from daph_ext.models.qwen import QwenArchitectureAdapter
from daph_ext.probes.base import ProbeResult
from daph_ext.probes.controller import AdapterAction, AdapterController


def test_exact_topk_fisher_prunes_exact_fraction():
    x = torch.ones(100)
    fisher = torch.arange(100, dtype=torch.float32)
    corrected, stats = hard_fisher_topk(x, fisher, fraction=0.15)
    assert int((corrected == 0).sum()) == 15
    assert stats.actual_fraction == pytest.approx(0.15)
    assert corrected[-15:].sum() == 0


def test_soft_fisher_is_identity_at_zero_lambda():
    x = torch.randn(12, 8)
    fisher = torch.rand_like(x)
    corrected, _ = soft_fisher_attenuation(x, fisher, lambda_reg=0.0)
    assert torch.equal(corrected, x)


def test_soft_fisher_attenuates_high_sensitivity_more():
    x = torch.ones(4)
    fisher = torch.tensor([1.0, 1.0, 10.0, 10.0])
    corrected, _ = soft_fisher_attenuation(x, fisher, lambda_reg=1.0)
    assert corrected[2] < corrected[0]


def test_normalized_fisher_is_finite_for_all_zero():
    out = normalize_fisher(torch.zeros(10))
    assert torch.isfinite(out).all()
    assert out.sum() == 0


def test_svd_retained_energy_rank_full_is_one():
    x = torch.randn(5, 3)
    assert svd_retained_energy(x, rank=3) == pytest.approx(1.0, abs=1e-6)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(16, 8)


class _Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=2, hidden_size=8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(), _Block()])


def test_qwen_adapter_introspects_real_module_paths():
    model = _Qwen()
    adapter = QwenArchitectureAdapter()
    assert adapter.num_layers(model) == 2
    assert adapter.hidden_size(model) == 8
    assert set(adapter.target_linear_names(model)) == {"q_proj", "v_proj", "down_proj"}
    shapes = adapter.module_shapes(model)
    assert shapes["q_proj"].in_features == 8
    assert shapes["down_proj"].in_features == 16
    assert adapter.layer_index("model.layers.1.q_proj") == 1


def test_controller_no_longer_has_universal_kl_threshold():
    cfg = ControllerConfig(max_kl_drift=None)
    controller = AdapterController(cfg)
    probes = [
        ProbeResult("repo_loss", baseline=1.0, candidate=0.8, higher_is_better=False),
        ProbeResult("kl_drift", baseline=0.0, candidate=100.0, higher_is_better=False),
    ]
    decision = controller.decide(probes)
    assert decision.action == AdapterAction.ACCEPT


def test_controller_requests_revalidation_for_generic_regression():
    controller = AdapterController(ControllerConfig(max_generic_regression=0.01))
    probes = [
        ProbeResult("repo_loss", baseline=1.0, candidate=0.8, higher_is_better=False),
        ProbeResult("generic_loss", baseline=1.0, candidate=1.1, higher_is_better=False),
    ]
    decision = controller.decide(probes)
    assert decision.action == AdapterAction.EVALUATE_CORRECTIONS
    assert decision.requires_revalidation


def test_provenance_marks_origin_explicitly(tmp_path):
    path = tmp_path / "result.json"
    ExperimentProvenance(DataOrigin.SYNTHETIC, "unit").write_json(path, {"score": 1})
    payload = json.loads(path.read_text())
    assert payload["data_origin"] == "synthetic"


def test_immutable_repo_splits_reject_overlap(tmp_path):
    from daph_ext.experiments.splits import ImmutableRepoSplits
    splits = ImmutableRepoSplits.from_sequences(["a"], ["b"], ["c"])
    digest = splits.write(tmp_path / "splits.json")
    assert len(digest) == 64
    with pytest.raises(ValueError):
        ImmutableRepoSplits.from_sequences(["a"], ["a"], ["c"]).validate()


def test_correct_and_refactor_group_reports_energy():
    from daph_ext.repobrain.correction import correct_and_refactor_group
    from daph_ext.repobrain.repo2lora import LoRAFactors
    torch.manual_seed(1)
    factors = LoRAFactors(
        a=torch.randn(1, 2, 6),
        b=torch.randn(1, 5, 2),
        alpha=4.0,
    )
    fisher = torch.rand(5, 6)
    updated, report = correct_and_refactor_group(
        factors,
        0,
        fisher,
        method="soft_fisher",
        value=0.1,
        min_retained_energy=0.0,
    )
    assert updated.rank == factors.rank
    assert 0.0 <= report.retained_energy <= 1.0 + 1e-6
    assert report.relative_error >= 0.0
    assert report.accepted
