from __future__ import annotations

from torch import Tensor, nn

from .base import VisionFeatures


class DINOv2Adapter(nn.Module):
    """Thin wrapper over a DINO-like model exposing `forward_features`.

    Expected upstream feature keys:
      - x_norm_clstoken: [B,D]
      - x_norm_patchtokens: [B,N,D]
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def encode(self, pixel_values: Tensor) -> VisionFeatures:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be [B,C,H,W]")
        out = self.model.forward_features(pixel_values)
        if not isinstance(out, dict):
            raise TypeError("DINO forward_features must return a mapping")
        missing = {"x_norm_clstoken", "x_norm_patchtokens"} - set(out)
        if missing:
            raise KeyError(f"DINO features missing keys: {sorted(missing)}")
        global_feature = out["x_norm_clstoken"]
        patch_features = out["x_norm_patchtokens"]
        if not isinstance(global_feature, Tensor) or not isinstance(patch_features, Tensor):
            raise TypeError("DINO feature values must be Tensors")
        if global_feature.ndim != 2 or patch_features.ndim != 3:
            raise ValueError("DINO global/patch features must be [B,D] and [B,N,D]")
        if global_feature.shape[0] != patch_features.shape[0]:
            raise ValueError("DINO global and patch feature batch sizes differ")
        return VisionFeatures(global_feature=global_feature, patch_features=patch_features)
