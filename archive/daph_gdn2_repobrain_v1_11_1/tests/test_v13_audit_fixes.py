import pytest
import torch
import torch.nn as nn

from daph_ext.geometry.fisher import empirical_fisher_diagonal
from daph_ext.probes.concrete import LogitKLDriftProbe, ValidationLossProbe
from daph_ext.repobrain.encoder import RepoPooler
from daph_ext.repobrain.evolution import RepoEvolutionState
from daph_ext.repobrain.layer import patch_repo_lora_linears
from daph_ext.repobrain.refactor import _truncated_balanced_factors


def test_evolution_state_supports_1d_tensors():
    evo = RepoEvolutionState(input_dim=16, state_dim=8)
    snapshot = torch.randn(16)
    state = evo.initialize(snapshot)
    assert state.shape == (8,)

    diff = torch.randn(16)
    new_state = evo.update(diff, state)
    assert new_state.shape == (8,)


def test_evolution_state_supports_batched_tensors():
    evo = RepoEvolutionState(input_dim=16, state_dim=8)
    snapshot = torch.randn(3, 16)
    state = evo.initialize(snapshot)
    assert state.shape == (3, 8)
    new_state = evo.update(torch.randn(3, 16), state)
    assert new_state.shape == (3, 8)


class _ConstantLM(nn.Module):
    def __init__(self, vocab: int = 10, bias_token: int = 1):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab = vocab
        self.bias_token = bias_token

    def forward(self, input_ids, attention_mask=None):
        bsz, seq = input_ids.shape
        logits = torch.zeros(bsz, seq, self.vocab, device=input_ids.device)
        logits[..., self.bias_token] = 2.0 + self.anchor * 0.0
        return logits


def test_validation_loss_probe_masks_inferred_padding():
    model = _ConstantLM()
    batch = {
        "input_ids": torch.tensor([[1, 1, 1, 1], [1, 1, 9, 9]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    }
    probe = ValidationLossProbe([batch])
    masked = probe._loss(model)

    # Explicit labels encode the same valid targets. The two paths must agree.
    explicit = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": torch.tensor([[1, 1, 1, 1], [1, 1, -100, -100]]),
    }
    explicit_loss = ValidationLossProbe([explicit])._loss(model)
    assert masked == pytest.approx(explicit_loss, rel=1e-6, abs=1e-6)


def test_kl_probe_ignores_padding_tokens():
    base = _ConstantLM(bias_token=1)
    candidate = _ConstantLM(bias_token=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
    }
    result = LogitKLDriftProbe([batch]).run(base, candidate)
    assert result.candidate == pytest.approx(0.0, abs=1e-7)
    assert result.metadata["tokens"] == "2"


def test_patch_repo_lora_linears_raises_on_empty_match():
    model = nn.Sequential(nn.Linear(8, 8))
    with pytest.raises(ValueError, match="Zero linear layers matched"):
        patch_repo_lora_linears(
            model,
            target_modules=("non_existent_proj",),
            num_layers=1,
            groups=1,
        )


class _ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.always = nn.Linear(2, 1, bias=False)
        self.conditional = nn.Linear(2, 1, bias=False)

    def forward(self, x, use_conditional=False):
        out = self.always(x)
        if use_conditional:
            out = out + self.conditional(x)
        return out


def test_fisher_normalizes_per_parameter_gradient_occurrence():
    model = _ConditionalModel()
    batches = [
        {"x": torch.ones(1, 2), "use_conditional": False},
        {"x": torch.ones(1, 2), "use_conditional": True},
    ]

    def loss_fn(m, batch):
        return m(batch["x"], batch["use_conditional"]).sum()

    fisher = empirical_fisher_diagonal(model, batches, loss_fn=loss_fn)
    assert fisher.samples == 2
    assert fisher.parameter_samples is not None
    assert fisher.parameter_samples["always.weight"] == 2
    assert fisher.parameter_samples["conditional.weight"] == 1
    # Gradients are exactly ones in both active cases; both diagonals should be 1.
    assert torch.allclose(fisher.values["always.weight"], torch.ones_like(fisher.values["always.weight"]))
    assert torch.allclose(
        fisher.values["conditional.weight"],
        torch.ones_like(fisher.values["conditional.weight"]),
    )


def test_svd_balanced_factor_sqrt_has_positive_floor():
    delta = torch.diag(torch.tensor([2.0, 1e-20, 0.0], requires_grad=True))
    b, a = _truncated_balanced_factors(delta, rank=3, lora_scale=2.0)
    assert torch.isfinite(a).all()
    assert torch.isfinite(b).all()
    assert a.shape == (3, 3)
    assert b.shape == (3, 3)


def test_repo_pooler_fp16_overflow_safety():
    pooler = RepoPooler()
    file_vectors = torch.full((1000, 16), 500.0, dtype=torch.float16)
    pooled = pooler(file_vectors)
    assert pooled.dtype == torch.float16
    assert torch.isfinite(pooled).all()
    assert torch.allclose(pooled.float(), torch.full_like(pooled.float(), 500.0))


def test_repo_pooler_weight_accumulation_stays_finite():
    pooler = RepoPooler()
    file_vectors = torch.full((1000, 8), 100.0, dtype=torch.float16)
    weights = torch.full((1000,), 100.0, dtype=torch.float16)
    pooled = pooler(file_vectors, weights)
    assert torch.isfinite(pooled).all()
