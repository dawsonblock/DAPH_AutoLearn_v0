import torch

from daph_ext.config import Repo2LoRAConfig
from daph_ext.repobrain.encoder import RepoPooler, chunk_token_ranges
from daph_ext.repobrain.materialize import materialize_group_delta
from daph_ext.repobrain.repo2lora import ModuleShape, Repo2LoRALite, layer_to_group


def test_repo_pool_shape():
    x = torch.randn(5, 16)
    pooled = RepoPooler()(x)
    assert pooled.shape == (32,)


def test_chunk_ranges_overlap():
    ranges = chunk_token_ranges(9000, 4096, 512)
    assert ranges[0] == (0, 4096)
    assert ranges[1][0] == 3584
    assert ranges[-1][1] == 9000


def test_generator_shapes():
    cfg = Repo2LoRAConfig(
        repo_embedding_dim=32,
        hidden_dim=24,
        rank=4,
        alpha=8.0,
        layer_groups=4,
        target_modules=("q_proj", "down_proj"),
    )
    shapes = {
        "q_proj": ModuleShape(in_features=16, out_features=16),
        "down_proj": ModuleShape(in_features=48, out_features=16),
    }
    model = Repo2LoRALite(cfg, shapes)
    out = model(torch.randn(1, 32))
    assert out["q_proj"].a.shape == (4, 4, 16)
    assert out["q_proj"].b.shape == (4, 16, 4)
    assert out["down_proj"].a.shape == (4, 4, 48)
    assert out["down_proj"].b.shape == (4, 16, 4)
    assert materialize_group_delta(out["q_proj"], 0).shape == (16, 16)


def test_layer_groups_cover_all_layers():
    groups = [layer_to_group(i, 28, 4) for i in range(28)]
    assert min(groups) == 0
    assert max(groups) == 3
    assert all(groups.count(g) == 7 for g in range(4))
