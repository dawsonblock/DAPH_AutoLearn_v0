import json
from pathlib import Path

import pytest
import torch

from daph_ext.calibration.controller import CalibrationRecord, calibrate_kl_envelope
from daph_ext.distributed.checkpoint import cache_state_dict, load_cache_state_dict
from daph_ext.mixers.cache import HybridSequenceCache
from daph_ext.mixers.gdn2_adapter import _move_state
from daph_ext.probes.sandbox import DockerSandboxConfig


def test_cache_to_preserves_integer_metadata_dtype():
    cache = HybridSequenceCache(recurrent_states={0: {"state": torch.ones(2, 3), "position": torch.tensor([1, 2], dtype=torch.long)}})
    cache.to(dtype=torch.float16)
    assert cache.recurrent_states[0]["state"].dtype == torch.float16
    assert cache.recurrent_states[0]["position"].dtype == torch.long


def test_external_state_move_preserves_integer_metadata_dtype():
    state = {"float": torch.ones(2), "index": torch.tensor([0, 1], dtype=torch.long)}
    out = _move_state(state, device=torch.device("cpu"), dtype=torch.float16)
    assert out["float"].dtype == torch.float16
    assert out["index"].dtype == torch.long


def test_cache_checkpoint_restore_device_dtype_and_schema():
    cache = HybridSequenceCache()
    cache.set_attention(0, torch.ones(2, 3), torch.ones(2, 3))
    cache.set_recurrent(1, {"x": torch.ones(2, 4), "idx": torch.tensor([2, 3], dtype=torch.long)})
    state = cache_state_dict(cache)
    assert state["schema_version"] == 2
    restored = load_cache_state_dict(state, dtype=torch.float16)
    assert restored.attention_kv[0][0].dtype == torch.float16
    assert restored.recurrent_states[1]["x"].dtype == torch.float16
    assert restored.recurrent_states[1]["idx"].dtype == torch.long


def test_reorder_nested_cache_only_batch_major_tensors():
    cache = HybridSequenceCache()
    cache.set_recurrent(0, {"state": torch.arange(12).reshape(3, 4), "meta": torch.tensor([9, 8])})
    cache.reorder_batch(torch.tensor([2, 0]), batch_size=3)
    assert cache.recurrent_states[0]["state"].shape == (2, 4)
    assert cache.recurrent_states[0]["meta"].tolist() == [9, 8]


def test_calibration_requires_minimum_safe_count():
    records = [CalibrationRecord(0.01, 0.0, 0.0, 0.1) for _ in range(4)]
    with pytest.raises(ValueError, match="at least 5"):
        calibrate_kl_envelope(records, min_safe_count=5)


def test_calibration_result_tracks_total_and_fraction():
    records = [CalibrationRecord(0.01 + i * 0.001, 0.0, 0.0, 0.1) for i in range(5)] + [CalibrationRecord(0.3, 0.2, 0.0, 0.1)]
    result = calibrate_kl_envelope(records, min_safe_count=5)
    assert result.safe_count == 5
    assert result.total_count == 6
    assert result.safe_fraction == pytest.approx(5/6)


def test_sandbox_default_uses_ephemeral_workspace():
    cfg = DockerSandboxConfig(image="example:latest")
    assert cfg.workspace_mode == "tmpfs_copy"
