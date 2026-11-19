from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass
class VisionFeatures:
    global_feature: Tensor
    patch_features: Tensor


class VisionEncoder(Protocol):
    def encode(self, pixel_values: Tensor) -> VisionFeatures: ...
