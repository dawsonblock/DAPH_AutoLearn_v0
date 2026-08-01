import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from daph_ext.config import ControllerConfig, DAPHIntegrationConfig, GDN2Config
from daph_ext.experiments.provenance import DataOrigin, ExperimentProvenance
from daph_ext.experiments.splits import ImmutableRepoSplits
from daph_ext.geometry.corrections_v14 import hard_fisher_topk
from daph_ext.mixers.cache import HybridSequenceCache
from daph_ext.mixers.gdn2_adapter import ExternalGDN2Adapter
from daph_ext.mixers.gdn2_reference import gdn2_recurrence_reference
from daph_ext.probes.base import ProbeResult
from daph_ext.probes.controller import AdapterAction, AdapterController
from daph_ext.repobrain.encoder import RepoPooler
from daph_ext.repobrain.layer import apply_generated_factors, patch_repo_lora_linears
from daph_ext.repobrain.repo2lora import LoRAFactors


def test_external_gdn2_three_tuple_preserves_third_cache(monkeypatch):
    module = types.ModuleType("fake_gdn2_module")

    class FakeGDN2(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

        def forward(self, hidden_states, **kwargs):
            cache = kwargs.get("past_key_values")
            return hidden_states + 1, None, cache

    module.GatedDeltaNet2 = FakeGDN2
    monkeypatch.setitem(sys.modules, "fake_gdn2_module", module)
    cfg = GDN2Config(external_module="fake_gdn2_module")
    adapter = ExternalGDN2Adapter(hidden_size=8, cfg=cfg, layer_idx=0)
    x = torch.randn(2, 3, 8)
    cache = {"state": torch.randn(2, 4)}
    out = adapter(x, recurrent_state=cache, use_cache=True)
    assert torch.allclose(out.hidden_states, x + 1)
    assert isinstance(out.recurrent_state, dict)
    assert torch.equal(out.recurrent_state["state"], cache["state"])


def test_gdn2_reference_reads_with_q_not_k():
    q = torch.tensor([[[[2.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]]]])
    v = torch.tensor([[[[3.0]]]])
    erase = torch.zeros_like(k)
    write = torch.ones_like(v)
    log_decay = torch.zeros_like(k)
    y, state = gdn2_recurrence_reference(k, v, erase, write, log_decay, q=q)
    # state after write = [[3],[0]], read by q=[2,0] => 6
    assert y.item() == pytest.approx(6.0)
    assert state[0, 0, 0, 0].item() == pytest.approx(3.0)


def test_kl_divergence_is_not_misclassified_as_ce_loss():
    controller = AdapterController(ControllerConfig(max_kl_drift=None, max_validation_loss_regression=0.01))
    probes = [
        ProbeResult("repo_loss", baseline=1.0, candidate=0.8, higher_is_better=False),
        ProbeResult("kl_divergence", baseline=0.0, candidate=100.0, higher_is_better=False),
    ]
    assert controller.decide(probes).action is AdapterAction.ACCEPT


def test_repo_tie_is_not_claimed_as_benefit():
    controller = AdapterController(ControllerConfig())
    probes = [ProbeResult("repo_loss", baseline=1.0, candidate=1.0, higher_is_better=False)]
    decision = controller.decide(probes)
    assert decision.action is AdapterAction.REJECT


def test_hard_fisher_does_not_arbitrarily_prune_all_zero_fisher():
    candidate = torch.randn(20)
    fisher = torch.zeros_like(candidate)
    corrected, stats = hard_fisher_topk(candidate, fisher, fraction=0.5)
    assert torch.equal(corrected, candidate)
    assert stats.actual_fraction == 0.0


def test_split_manifest_rejects_duplicates_and_normalizes_whitespace():
    with pytest.raises(ValueError, match="duplicate"):
        ImmutableRepoSplits.from_sequences(["a", "a"], ["b"], ["c"]).validate()
    splits = ImmutableRepoSplits.from_sequences([" a "], ["b"], ["c"])
    assert splits.train == ("a",)


def test_empirical_provenance_requires_real_identifiers(tmp_path: Path):
    bad = ExperimentProvenance(DataOrigin.EMPIRICAL, "exp")
    with pytest.raises(ValueError, match="checkpoint"):
        bad.write_json(tmp_path / "bad.json")
    good = ExperimentProvenance(
        DataOrigin.EMPIRICAL,
        "exp",
        checkpoint="Qwen/test",
        dataset_hash="abc123",
        seed=7,
    )
    out = tmp_path / "nested" / "result.json"
    good.write_json(out, {"metric": 1.0})
    assert out.exists()


def test_pooler_rejects_zero_total_weights():
    with pytest.raises(ValueError, match="at least one"):
        RepoPooler()(torch.ones(2, 3), torch.zeros(2))


def test_hybrid_cache_reorder_and_shared_cache_guard():
    cache = HybridSequenceCache()
    k = torch.arange(12).reshape(2, 2, 3).float()
    v = k + 100
    cache.set_attention(0, k, v)
    cache.set_recurrent(1, torch.arange(8).reshape(2, 4).float())
    cache.reorder_batch(torch.tensor([1, 0]))
    rk, _ = cache.get_attention(0)
    assert torch.equal(rk[0], k[1])

    cache.set_shared_recurrent_cache(object())
    before = cache.get_attention(0)[0].clone()
    with pytest.raises(TypeError, match="framework-native"):
        cache.reorder_batch(torch.tensor([0, 1]))
    assert torch.equal(cache.get_attention(0)[0], before)


def test_patch_requires_all_requested_targets_by_default():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    with pytest.raises(ValueError, match="requested target modules"):
        patch_repo_lora_linears(
            Model(), target_modules=("q_proj", "v_proj"), num_layers=1, groups=1
        )


def test_apply_generated_factors_is_strict_by_default():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])

    wrappers = patch_repo_lora_linears(Model(), target_modules=("q_proj",), num_layers=1, groups=1)
    with pytest.raises(KeyError, match="missing generated factors"):
        apply_generated_factors(wrappers, {})


def test_research_yaml_has_no_uncalibrated_kl_threshold():
    root = Path(__file__).resolve().parents[1]
    cfg = DAPHIntegrationConfig.from_yaml(root / "configs" / "daph_qwen06b_research.yaml")
    assert cfg.controller.max_kl_drift is None
