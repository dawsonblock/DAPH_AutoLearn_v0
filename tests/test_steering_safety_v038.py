from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from daph_learning.steering.hooks import (
    SteeringSafetyLimits,
    residual_addition_hook,
)


class OneLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Identity()])
        self.model = inner


def test_relative_perturbation_is_clamped() -> None:
    model = OneLayerModel()
    telemetry: list[dict] = []
    hidden = torch.ones(1, 3, 4)
    with residual_addition_hook(
        model,
        layer_index=0,
        vector=np.ones(4, dtype=np.float32),
        alpha=100.0,
        token_scope="last",
        telemetry_sink=telemetry,
        safety_limits=SteeringSafetyLimits(
            max_relative_perturbation=0.10,
            max_cosine_shift=2.0,
        ),
    ):
        output = model.model.layers[0](hidden)
    assert telemetry[0]["safety_clamped"]
    assert "relative_perturbation" in telemetry[0]["safety_reasons"]
    assert telemetry[0]["relative_perturbation"] <= 0.100001
    assert not torch.equal(output, hidden)


def test_safety_limits_validate_ranges() -> None:
    with pytest.raises(ValueError):
        SteeringSafetyLimits(max_relative_perturbation=0)
    with pytest.raises(ValueError):
        SteeringSafetyLimits(max_cosine_shift=3)
