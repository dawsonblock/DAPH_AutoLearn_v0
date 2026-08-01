import torch
from torch import nn

from daph_ext.vision.dinov2_adapter import DINOv2Adapter
from daph_ext.vision.fusion import StructuralSemanticFusion


class FakeDINO(nn.Module):
    def forward_features(self, x):
        b = x.shape[0]
        return {
            "x_norm_clstoken": torch.randn(b, 8),
            "x_norm_patchtokens": torch.randn(b, 16, 8),
        }


def test_dino_adapter_contract():
    enc = DINOv2Adapter(FakeDINO())
    out = enc.encode(torch.randn(2, 3, 224, 224))
    assert out.global_feature.shape == (2, 8)
    assert out.patch_features.shape == (2, 16, 8)


def test_fusion_shape():
    f = StructuralSemanticFusion(semantic_dim=10, structural_dim=8, hidden_size=12)
    y = f(torch.randn(2, 16, 10), torch.randn(2, 16, 8))
    assert y.shape == (2, 16, 12)
