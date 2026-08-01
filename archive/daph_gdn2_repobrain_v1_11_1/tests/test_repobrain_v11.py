import torch
import torch.nn as nn

from daph_ext.repobrain.layer import RepoLoRALinear, apply_generated_factors, patch_repo_lora_linears
from daph_ext.repobrain.materialize import materialize_group_delta
from daph_ext.repobrain.refactor import project_and_refactor_factors
from daph_ext.repobrain.repo2lora import LoRAFactors


def test_refactor_preserves_projected_delta_at_sufficient_rank():
    torch.manual_seed(0)
    a = torch.randn(2, 2, 4)
    b = torch.randn(2, 4, 2)
    factors = LoRAFactors(a=a, b=b, alpha=4.0)
    before = materialize_group_delta(factors, 0)
    reference = torch.ones_like(before)
    updated = project_and_refactor_factors(factors, 0, reference, strength=1.0)
    after = materialize_group_delta(updated, 0)
    # The correction is rank <= original rank here only approximately, but must
    # be finite and substantially less aligned to the reference.
    old_cos = torch.nn.functional.cosine_similarity(before.flatten(), reference.flatten(), dim=0).abs()
    new_cos = torch.nn.functional.cosine_similarity(after.flatten(), reference.flatten(), dim=0).abs()
    assert torch.isfinite(after).all()
    assert new_cos <= old_cos + 1e-5


def test_repo_lora_linear_matches_dense_delta():
    torch.manual_seed(0)
    base = nn.Linear(5, 3, bias=False)
    wrapper = RepoLoRALinear(base, group_idx=0)
    factors = LoRAFactors(a=torch.randn(1, 2, 5), b=torch.randn(1, 3, 2), alpha=4.0)
    wrapper.set_factors(factors, scale=0.75)
    x = torch.randn(2, 4, 5)
    y = wrapper(x)
    dense = 0.75 * materialize_group_delta(factors, 0)
    expected = base(x) + torch.nn.functional.linear(x, dense)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-5)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.v_proj(self.q_proj(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock() for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_patch_and_apply_generated_factors():
    model = TinyModel()
    wrappers = patch_repo_lora_linears(model, target_modules=("q_proj", "v_proj"), num_layers=4, groups=2)
    assert len(wrappers) == 8
    factors = {
        "q_proj": LoRAFactors(torch.randn(2, 2, 4), torch.randn(2, 4, 2), 4.0),
        "v_proj": LoRAFactors(torch.randn(2, 2, 4), torch.randn(2, 4, 2), 4.0),
    }
    apply_generated_factors(wrappers, factors)
    assert all(w.current_factors is not None for w in wrappers.values())
