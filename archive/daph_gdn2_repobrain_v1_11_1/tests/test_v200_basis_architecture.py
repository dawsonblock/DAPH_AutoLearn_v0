from __future__ import annotations

import torch
from torch import nn

from daph_ext.geometry.adapter_basis import (
    JointAdapterBasis,
    export_basis_payload,
    fit_joint_adapter_basis,
    load_basis_payload,
)
from daph_ext.mixers.base import MixerOutput
from daph_ext.mixers.residual_gdn2 import ResidualGDN2Mixer
from daph_ext.repobrain.basis_repo2lora import BasisRepoBrain, RepoCoefficientPredictor
from daph_ext.repobrain.materialize import materialize_group_delta
from daph_ext.repobrain.repo2lora import LoRAFactors


def _factor(delta: torch.Tensor) -> LoRAFactors:
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    r = min(delta.shape)
    root = s[:r].sqrt()
    a = root[:, None] * vh[:r]
    b = u[:, :r] * root[None, :]
    return LoRAFactors(a=a.unsqueeze(0), b=b.unsqueeze(0), alpha=float(r))


def test_basis_compose_is_exact_weighted_sum_for_low_rank_blocks() -> None:
    # K=2, G=1, R=1, In=2, Out=2
    a = torch.tensor([
        [[[1.0, 0.0]]],
        [[[0.0, 1.0]]],
    ])
    b = torch.tensor([
        [[[[1.0], [2.0]]]],
        [[[[3.0], [4.0]]]],
    ]).squeeze(1)  # [K,G,Out,R]
    basis = JointAdapterBasis({"q_proj": (a, b)})
    c = torch.tensor([2.0, -0.5])
    factors = basis.compose(c)["q_proj"]
    got = materialize_group_delta(factors, 0)
    expected = 2.0 * (b[0, 0] @ a[0, 0]) - 0.5 * (b[1, 0] @ a[1, 0])
    assert torch.allclose(got, expected, atol=1e-6)


def test_basis_compose_batched_coefficients() -> None:
    a = torch.randn(3, 2, 2, 4)
    b = torch.randn(3, 2, 5, 2)
    basis = JointAdapterBasis({"v_proj": (a, b)})
    c = torch.randn(4, 3)
    factors = basis.compose(c)["v_proj"]
    assert factors.is_batched
    assert factors.batch_size == 4
    assert factors.rank == 6
    for row in range(4):
        got = materialize_group_delta(factors, 1)[row]
        expected = sum(c[row, k] * (b[k, 1] @ a[k, 1]) for k in range(3))
        assert torch.allclose(got, expected, atol=1e-5)


def test_joint_basis_fit_recovers_simple_teacher_subspace() -> None:
    d1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    d2 = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    teachers = [{"q_proj": _factor(d1)}, {"q_proj": _factor(d2)}]
    basis, coeff, diag = fit_joint_adapter_basis(teachers, basis_size=2, basis_rank=2)
    assert basis.basis_size == 2
    assert coeff.shape == (2, 2)
    assert diag.teacher_count == 2
    assert diag.cumulative_explained_energy[-1] > 0.99999
    for i in range(2):
        recon = materialize_group_delta(basis.compose(coeff[i])["q_proj"], 0)
        assert torch.allclose(recon, (d1, d2)[i], atol=1e-5)


def test_coefficients_are_refit_after_rank_compression() -> None:
    torch.manual_seed(7)
    dense_teachers = [torch.randn(4, 4) for _ in range(4)]
    teachers = [{"q_proj": _factor(delta)} for delta in dense_teachers]
    basis, coefficients, diag = fit_joint_adapter_basis(
        teachers,
        basis_size=3,
        basis_rank=1,
    )

    matrix = torch.stack([delta.reshape(-1) for delta in dense_teachers])
    compressed = torch.stack(
        [
            basis.dense_direction("q_proj", index, 0).detach().reshape(-1)
            for index in range(basis.basis_size)
        ]
    )
    new_error = torch.linalg.vector_norm(
        matrix - coefficients @ compressed
    )

    # This is the stale coefficient calculation used by v1.11.0: projection
    # against the uncompressed orthonormal SVD basis, followed by composition
    # against a different rank-truncated basis.
    _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
    stale_coefficients = matrix @ vh[: basis.basis_size].T
    stale_error = torch.linalg.vector_norm(
        matrix - stale_coefficients @ compressed
    )

    assert new_error <= stale_error + 1e-6
    assert float(new_error) < float(stale_error)
    assert diag.coefficient_fit_method == "least_squares_after_rank_compression"
    assert len(diag.teacher_relative_errors) == len(teachers)
    assert diag.post_compression_explained_energy < 1.0


def test_basis_payload_round_trip() -> None:
    basis = JointAdapterBasis({"q_proj": (torch.randn(2, 1, 1, 3), torch.randn(2, 1, 4, 1))})
    loaded = load_basis_payload(export_basis_payload(basis))
    c = torch.tensor([0.2, -0.3])
    before = materialize_group_delta(basis.compose(c)["q_proj"], 0)
    after = materialize_group_delta(loaded.compose(c)["q_proj"], 0)
    assert torch.allclose(before, after)


def test_coefficient_predictor_is_fsdp_safe_and_signed() -> None:
    predictor = RepoCoefficientPredictor(8, 16, 4)
    assert predictor.log_scale.shape == (1,)
    assert all(p.ndim > 0 for p in predictor.parameters())
    out = predictor(torch.randn(3, 8))
    assert out.shape == (3, 4)
    assert torch.isfinite(out).all()


def test_basis_repobrain_emits_runtime_factors() -> None:
    basis = JointAdapterBasis({"q_proj": (torch.randn(3, 2, 1, 4), torch.randn(3, 2, 5, 1))})
    model = BasisRepoBrain(basis, repo_embedding_dim=7, hidden_dim=11)
    out = model(torch.randn(2, 7))
    assert out.coefficients.shape == (2, 3)
    f = out.factors_by_module["q_proj"]
    assert f.is_batched and f.batch_size == 2 and f.groups == 2 and f.rank == 3


class _DoubleMixer(nn.Module):
    def forward(self, hidden_states, *, attention_mask=None, recurrent_state=None, use_cache=False):
        del attention_mask, use_cache
        return MixerOutput(hidden_states=2 * hidden_states, recurrent_state=recurrent_state)


def test_residual_gdn2_is_exact_identity_at_zero_gate() -> None:
    mixer = ResidualGDN2Mixer(_DoubleMixer(), gate_init=0.0)
    x = torch.randn(2, 4, 6)
    state = {"s": torch.ones(1)}
    out = mixer(x, recurrent_state=state)
    assert torch.equal(out.hidden_states, x)
    assert out.recurrent_state is state


def test_residual_gdn2_interpolates_to_mixer() -> None:
    mixer = ResidualGDN2Mixer(_DoubleMixer(), gate_init=0.5)
    x = torch.ones(1, 2, 3)
    out = mixer(x)
    w = torch.tanh(torch.tensor(0.5))
    assert torch.allclose(out.hidden_states, x + w * x)


def test_residual_gdn2_gate_is_fsdp_safe() -> None:
    mixer = ResidualGDN2Mixer(_DoubleMixer())
    assert mixer.gate.shape == (1,)
