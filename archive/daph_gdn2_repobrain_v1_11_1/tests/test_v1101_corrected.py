import torch
import torch.nn as nn

from daph_ext.config import Repo2LoRAConfig
from daph_ext.geometry.fisher import empirical_fisher_diagonal
from daph_ext.repobrain.adapter_context import use_adapter_context
from daph_ext.repobrain.layer import RepoLoRALinear
from daph_ext.repobrain.repo2lora import LoRAFactors, ModuleShape, Repo2LoRALite


def test_repobrain_has_no_scalar_parameters_for_fsdp():
    cfg = Repo2LoRAConfig(
        repo_embedding_dim=8,
        hidden_dim=8,
        rank=2,
        layer_groups=1,
        target_modules=("q_proj",),
    )
    model = Repo2LoRALite(cfg, {"q_proj": ModuleShape(4, 4)})
    scalar = [name for name, p in model.named_parameters() if p.ndim == 0]
    assert scalar == []
    assert model.heads["q_proj"].log_scale.shape == (1,)


def _batched_factors() -> LoRAFactors:
    # Repository 0 adds +sum(x); repository 1 adds +2*sum(x) to each output.
    a = torch.tensor(
        [
            [[[1.0, 1.0]]],
            [[[1.0, 1.0]]],
        ]
    )  # [B=2,G=1,R=1,In=2]
    b = torch.tensor(
        [
            [[[1.0], [1.0]]],
            [[[2.0], [2.0]]],
        ]
    )  # [B=2,G=1,Out=2,R=1]
    return LoRAFactors(a=a, b=b, alpha=1.0)


def test_batched_adapter_auto_repeat_interleave_for_generation_expansion():
    base = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(base.weight)
    wrapper = RepoLoRALinear(base, 0, module_key="q_proj")
    x = torch.ones(4, 2)  # two source repos, expanded by two beams each
    with use_adapter_context({"q_proj": _batched_factors()}):
        y = wrapper(x)
    assert torch.allclose(y[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(y[1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(y[2], torch.tensor([4.0, 4.0]))
    assert torch.allclose(y[3], torch.tensor([4.0, 4.0]))


def test_batched_adapter_explicit_indices_support_reordered_fanout():
    base = nn.Linear(2, 2, bias=False)
    nn.init.zeros_(base.weight)
    wrapper = RepoLoRALinear(base, 0, module_key="q_proj")
    x = torch.ones(4, 2)
    indices = torch.tensor([1, 0, 1, 0])
    with use_adapter_context({"q_proj": _batched_factors()}, batch_indices=indices):
        y = wrapper(x)
    expected = torch.tensor([[4.0, 4.0], [2.0, 2.0], [4.0, 4.0], [2.0, 2.0]])
    assert torch.allclose(y, expected)


def test_empirical_fisher_defaults_to_per_sample_estimator():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    batch = {"x": torch.tensor([[1.0], [-1.0]])}

    def loss_fn(m, one):
        # Per-example gradients are +1 and -1; their mean cancels to zero.
        return m(one["x"]).sum()

    fisher = empirical_fisher_diagonal(model, [batch], loss_fn=loss_fn)
    assert fisher.samples == 2
    assert torch.allclose(fisher.values["weight"], torch.ones_like(fisher.values["weight"]))
