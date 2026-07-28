import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from daph_ext.calibration.controller import CalibrationRecord, CandidatePoint, calibrate_kl_envelope, pareto_frontier
from daph_ext.config import ControllerConfig, DAPHIntegrationConfig
from daph_ext.distributed.checkpoint import cache_state_dict, load_cache_state_dict
from daph_ext.distributed.placement import AdapterPlacementManager
from daph_ext.mixers.cache import HybridSequenceCache
from daph_ext.probes.sandbox import DockerSandboxConfig, parse_pytest_summary
from daph_ext.repobrain.correction import correct_and_refactor_group
from daph_ext.repobrain.hf_embedder import chunk_token_ranges
from daph_ext.repobrain.layer import RepoLoRALinear
from daph_ext.repobrain.repo2lora import LoRAFactors


def test_chunk_token_ranges_overlap():
    assert chunk_token_ranges(9000, chunk_size=4096, overlap=512) == [(0, 4096), (3584, 7680), (7168, 9000)]


def test_kl_calibration_uses_only_safe_positive_benefit():
    records = [
        CalibrationRecord(0.01, 0.0, 0.0, 0.1),
        CalibrationRecord(0.02, 0.005, 0.0, 0.2),
        CalibrationRecord(0.90, 0.5, 0.0, 0.2),
        CalibrationRecord(0.80, 0.0, 0.0, 0.0),
    ]
    result = calibrate_kl_envelope(records, percentile=100)
    assert result.threshold == pytest.approx(0.02)
    assert result.safe_count == 2


def test_pareto_frontier_drops_dominated_point():
    good = CandidatePoint("good", 0.8, 0.01, 0.0, 0.02)
    bad = CandidatePoint("bad", 0.7, 0.02, 0.0, 0.03)
    assert [p.name for p in pareto_frontier([bad, good])] == ["good"]


def test_production_config_rejects_uncalibrated_kl():
    path = Path('configs/daph_qwen06b_production_template.yaml')
    with pytest.raises(ValueError, match='calibrated max_kl_drift'):
        DAPHIntegrationConfig.from_yaml(path)


def test_correction_acceptance_checks_error_and_energy():
    a = torch.randn(1, 2, 4)
    b = torch.randn(1, 4, 2)
    factors = LoRAFactors(a=a, b=b, alpha=2.0)
    fisher = torch.ones(4, 4)
    _, report = correct_and_refactor_group(factors, 0, fisher, method='soft_fisher', value=0.1, min_retained_energy=0.0, max_relative_error=0.0)
    assert report.relative_error >= 0
    assert report.accepted is (report.relative_error <= 0.0)


def test_cache_state_dict_roundtrip_tensor_state():
    cache = HybridSequenceCache()
    cache.set_attention(0, torch.randn(2,3), torch.randn(2,3))
    cache.set_recurrent(1, torch.randn(2,4,5))
    state = cache_state_dict(cache)
    restored = load_cache_state_dict(state)
    assert torch.equal(restored.get_attention(0)[0], cache.get_attention(0)[0].cpu())
    assert torch.equal(restored.get_recurrent(1), cache.get_recurrent(1).cpu())


def test_cache_opaque_requires_codec():
    cache = HybridSequenceCache(shared_recurrent_cache=object())
    with pytest.raises(TypeError, match='OpaqueCacheCodec'):
        cache_state_dict(cache)


def test_placement_manager_sets_local_factors():
    wrapper = RepoLoRALinear(nn.Linear(4, 4), group_idx=0)
    factors = LoRAFactors(a=torch.randn(1,2,4), b=torch.randn(1,4,2), alpha=2.0)
    AdapterPlacementManager().apply({'model.layers.0.q_proj': wrapper}, {'q_proj': factors})
    assert wrapper.current_factors is not None
    assert wrapper.current_factors.a.device == wrapper.base_linear.weight.device


def test_parse_pytest_summary():
    summary = parse_pytest_summary('12 passed, 3 failed, 1 skipped in 2.1s')
    assert summary.passed == 12
    assert summary.total == 15


def test_docker_config_validation():
    with pytest.raises(ValueError):
        DockerSandboxConfig(image='', cpus=1)

class _Cfg:
    def __init__(self, model_type):
        self.model_type = model_type
        self.num_hidden_layers = 2
        self.hidden_size = 8


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(16, 8)


class _Decoder(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.config = _Cfg(model_type)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(), _Block()])


def test_registry_supports_common_decoder_families():
    from daph_ext.models.registry import get_architecture_adapter
    for family in ('llama', 'mistral', 'gemma2', 'qwen2'):
        model = _Decoder(family)
        adapter = get_architecture_adapter(model)
        assert adapter.num_layers(model) == 2
        assert 'q_proj' in adapter.module_shapes(model)


def test_build_correction_grid_matches_config():
    from daph_ext.calibration.controller import build_correction_grid
    cfg = ControllerConfig()
    grid = build_correction_grid(cfg)
    assert sum(c.kind == 'scale' for c in grid) == len(cfg.scale_grid)
    assert sum(c.kind == 'hard_fisher' for c in grid) == len(cfg.hard_fisher_grid)
    assert sum(c.kind == 'soft_fisher' for c in grid) == len(cfg.soft_fisher_grid)


def test_evolution_checkpoint_roundtrip():
    from daph_ext.distributed.checkpoint import evolution_checkpoint, restore_evolution_checkpoint
    from daph_ext.repobrain.evolution import RepoEvolutionState
    evo = RepoEvolutionState(4, 3)
    hidden = evo.initialize(torch.randn(4))
    payload = evolution_checkpoint(evo, hidden)
    restored = restore_evolution_checkpoint(evo, payload)
    assert torch.equal(restored, hidden.cpu())
