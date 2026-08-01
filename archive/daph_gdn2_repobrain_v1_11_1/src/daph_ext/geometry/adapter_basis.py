from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from daph_ext.repobrain.materialize import materialize_group_delta
from daph_ext.repobrain.repo2lora import LoRAFactors


@dataclass(frozen=True)
class BasisFitDiagnostics:
    basis_size: int
    teacher_count: int
    explained_energy: tuple[float, ...]
    cumulative_explained_energy: tuple[float, ...]
    block_relative_errors: dict[str, float]
    teacher_relative_errors: tuple[float, ...]
    post_compression_explained_energy: float
    coefficient_condition_number: float
    coefficient_fit_method: str


class _ModuleBasis(nn.Module):
    """Low-rank factors for K whole-model basis directions for one module type.

    A: [K,G,R,In]
    B: [K,G,Out,R]
    Each basis direction materializes as B[k,g] @ A[k,g].
    """

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__()
        if a.ndim != 4 or b.ndim != 4:
            raise ValueError("basis A/B must be rank-4 tensors")
        if a.shape[:2] != b.shape[:2] or a.shape[2] != b.shape[-1]:
            raise ValueError("basis A/B dimensions are incompatible")
        self.a = nn.Parameter(a.detach().clone(), requires_grad=False)
        self.b = nn.Parameter(b.detach().clone(), requires_grad=False)

    @property
    def basis_size(self) -> int:
        return self.a.shape[0]

    @property
    def groups(self) -> int:
        return self.a.shape[1]

    @property
    def rank(self) -> int:
        return self.a.shape[2]


class JointAdapterBasis(nn.Module):
    """A shared whole-model adapter/task basis U.

    Basis index k has a low-rank block for every target module and layer group.
    A repository coefficient vector c(repo) composes an exact weighted sum of
    these low-rank blocks without materializing dense weight matrices.
    """

    def __init__(self, module_factors: Mapping[str, tuple[Tensor, Tensor]]) -> None:
        super().__init__()
        if not module_factors:
            raise ValueError("module_factors must be non-empty")
        self.modules_basis = nn.ModuleDict({
            name: _ModuleBasis(a, b) for name, (a, b) in module_factors.items()
        })
        sizes = {m.basis_size for m in self.modules_basis.values()}
        if len(sizes) != 1:
            raise ValueError("all module bases must share the same basis size")

    @property
    def basis_size(self) -> int:
        return next(iter(self.modules_basis.values())).basis_size

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(self.modules_basis.keys())

    def compose(self, coefficients: Tensor) -> dict[str, LoRAFactors]:
        """Compose c(repo) into runtime LoRA factors.

        coefficients: [B,K] or [K]. Signed coefficients are supported.
        The returned composite rank is K*R and alpha is set equal to that rank,
        so RepoLoRALinear's alpha/rank scaling is exactly 1.0. Thus the runtime
        delta equals sum_k c_k U_k (up to the stored low-rank basis approximation).
        """
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)
        if coefficients.ndim != 2 or coefficients.shape[1] != self.basis_size:
            raise ValueError(
                f"coefficients must be [B,{self.basis_size}] or [{self.basis_size}], "
                f"got {tuple(coefficients.shape)}"
            )
        result: dict[str, LoRAFactors] = {}
        batch = coefficients.shape[0]
        for name, bank in self.modules_basis.items():
            a0 = bank.a.to(device=coefficients.device, dtype=coefficients.dtype)
            b0 = bank.b.to(device=coefficients.device, dtype=coefficients.dtype)
            # [K,G,R,In] -> [B,G,K*R,In]
            a = a0.permute(1, 0, 2, 3).reshape(
                bank.groups, self.basis_size * bank.rank, a0.shape[-1]
            ).unsqueeze(0).expand(batch, -1, -1, -1)
            # Weight each B block by its repository coefficient.
            weighted_b = b0.unsqueeze(0) * coefficients[:, :, None, None, None]
            b = weighted_b.permute(0, 2, 3, 1, 4).reshape(
                batch, bank.groups, b0.shape[-2], self.basis_size * bank.rank
            )
            composite_rank = self.basis_size * bank.rank
            if batch == 1:
                a = a.squeeze(0)
                b = b.squeeze(0)
            result[name] = LoRAFactors(a=a, b=b, alpha=float(composite_rank))
        return result

    def dense_direction(self, module_name: str, basis_idx: int, group_idx: int) -> Tensor:
        bank = self.modules_basis[module_name]
        if not (0 <= basis_idx < self.basis_size):
            raise IndexError(basis_idx)
        if not (0 <= group_idx < bank.groups):
            raise IndexError(group_idx)
        return bank.b[basis_idx, group_idx] @ bank.a[basis_idx, group_idx]


def _factor_dense(delta: Tensor, rank: int) -> tuple[Tensor, Tensor, float]:
    if delta.ndim != 2:
        raise ValueError("dense adapter block must be rank-2")
    if rank <= 0:
        raise ValueError("rank must be > 0")
    work = delta.float()
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    r = min(rank, s.numel())
    root = torch.sqrt(torch.clamp(s[:r], min=0))
    b = u[:, :r] * root.unsqueeze(0)
    a = root.unsqueeze(1) * vh[:r]
    if r < rank:
        a = torch.cat([a, torch.zeros(rank-r, a.shape[1], dtype=a.dtype, device=a.device)], dim=0)
        b = torch.cat([b, torch.zeros(b.shape[0], rank-r, dtype=b.dtype, device=b.device)], dim=1)
    recon = b @ a
    denom = torch.linalg.vector_norm(work).clamp_min(1e-12)
    rel = float((torch.linalg.vector_norm(work - recon) / denom).item())
    return a.to(delta.dtype), b.to(delta.dtype), rel


def fit_joint_adapter_basis(
    teachers: Sequence[Mapping[str, LoRAFactors]],
    *,
    basis_size: int,
    basis_rank: int,
) -> tuple[JointAdapterBasis, Tensor, BasisFitDiagnostics]:
    """Fit whole-model principal adapter directions from direct-LoRA teachers.

    The teacher matrix is built from dense, *scaled* LoRA deltas across every
    target module and layer group. SVD learns joint directions, preserving the
    cross-module geometry of each teacher adapter. Each dense basis block is then
    refactored to `basis_rank` for efficient runtime composition.

    Returns: (basis, teacher_coefficients [N,K], diagnostics).
    """
    if not teachers:
        raise ValueError("at least one teacher adapter is required")
    if basis_size <= 0 or basis_rank <= 0:
        raise ValueError("basis_size and basis_rank must be > 0")
    module_names = tuple(sorted(teachers[0].keys()))
    if not module_names:
        raise ValueError("teacher adapters must contain at least one module")

    layout: list[tuple[str, int, tuple[int, int], int, int]] = []
    rows: list[Tensor] = []
    offset = 0
    for teacher_idx, teacher in enumerate(teachers):
        if tuple(sorted(teacher.keys())) != module_names:
            raise ValueError(f"teacher {teacher_idx} module keys differ from teacher 0")
        pieces: list[Tensor] = []
        if teacher_idx == 0:
            layout = []
            offset = 0
        for name in module_names:
            f = teacher[name]
            if f.is_batched:
                raise ValueError("basis fitting expects one adapter per teacher, not batched factors")
            for g in range(f.groups):
                dense = materialize_group_delta(f, g).detach().float().cpu()
                if teacher_idx == 0:
                    n = dense.numel()
                    layout.append((name, g, tuple(dense.shape), offset, offset+n))
                    offset += n
                else:
                    expected = next(x for x in layout if x[0] == name and x[1] == g)[2]
                    if tuple(dense.shape) != expected:
                        raise ValueError(f"teacher block shape mismatch for {name} group {g}")
                pieces.append(dense.reshape(-1))
        rows.append(torch.cat(pieces))

    matrix = torch.stack(rows, dim=0)
    _, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    k = min(basis_size, vh.shape[0])
    basis_flat = vh[:k]
    total_energy = torch.sum(s.square()).clamp_min(1e-12)
    explained = (s[:k].square() / total_energy)
    cumulative = torch.cumsum(explained, dim=0)

    by_module: dict[str, list[tuple[Tensor, Tensor]]] = {name: [] for name in module_names}
    errors: dict[str, list[float]] = {name: [] for name in module_names}
    # For each whole-model principal direction, refactor every module/group block.
    for basis_idx in range(k):
        for name in module_names:
            group_entries = [x for x in layout if x[0] == name]
            a_groups: list[Tensor] = []
            b_groups: list[Tensor] = []
            for _, g, shape, start, end in sorted(group_entries, key=lambda x: x[1]):
                dense = basis_flat[basis_idx, start:end].reshape(shape)
                a, b, rel = _factor_dense(dense, basis_rank)
                a_groups.append(a)
                b_groups.append(b)
                errors[name].append(rel)
            by_module[name].append((torch.stack(a_groups), torch.stack(b_groups)))

    module_factors: dict[str, tuple[Tensor, Tensor]] = {}
    for name, dirs in by_module.items():
        module_factors[name] = (
            torch.stack([a for a, _ in dirs], dim=0),
            torch.stack([b for _, b in dirs], dim=0),
        )

    basis = JointAdapterBasis(module_factors)

    # Rank truncation changes every dense SVD direction independently.  The
    # original projection coefficients are therefore stale after compression.
    # Rebuild the actual compressed design matrix and solve for the teacher
    # coordinates against that basis.
    compressed_rows: list[Tensor] = []
    for basis_idx in range(k):
        pieces = [
            basis.dense_direction(name, basis_idx, group).detach().float().cpu().reshape(-1)
            for name, group, _shape, _start, _end in layout
        ]
        compressed_rows.append(torch.cat(pieces))
    compressed_basis_flat = torch.stack(compressed_rows, dim=0)
    coeff = torch.linalg.lstsq(
        compressed_basis_flat.T,
        matrix.T,
    ).solution.T
    reconstructed = coeff @ compressed_basis_flat
    teacher_norms = torch.linalg.vector_norm(matrix, dim=1).clamp_min(1e-12)
    teacher_errors = (
        torch.linalg.vector_norm(matrix - reconstructed, dim=1)
        / teacher_norms
    )
    post_energy = 1.0 - float(
        (
            torch.sum((matrix - reconstructed).square())
            / torch.sum(matrix.square()).clamp_min(1e-12)
        ).item()
    )
    compressed_singular_values = torch.linalg.svdvals(compressed_basis_flat)
    if (
        compressed_singular_values.numel() == 0
        or float(compressed_singular_values[-1]) <= 1e-12
    ):
        condition_number = float("inf")
    else:
        condition_number = float(
            (
                compressed_singular_values[0]
                / compressed_singular_values[-1]
            ).item()
        )

    diag = BasisFitDiagnostics(
        basis_size=k,
        teacher_count=len(teachers),
        explained_energy=tuple(float(x) for x in explained.tolist()),
        cumulative_explained_energy=tuple(float(x) for x in cumulative.tolist()),
        block_relative_errors={
            name: float(sum(vals) / max(1, len(vals))) for name, vals in errors.items()
        },
        teacher_relative_errors=tuple(float(x) for x in teacher_errors.tolist()),
        post_compression_explained_energy=post_energy,
        coefficient_condition_number=condition_number,
        coefficient_fit_method="least_squares_after_rank_compression",
    )
    return basis, coeff, diag


def export_basis_payload(basis: JointAdapterBasis) -> dict[str, object]:
    """Return a torch.save-friendly payload with all basis tensors and metadata."""
    modules: dict[str, dict[str, Tensor]] = {}
    for name, bank in basis.modules_basis.items():
        modules[name] = {"a": bank.a.detach().cpu(), "b": bank.b.detach().cpu()}
    return {"format": "daph_joint_adapter_basis_v1", "modules": modules}


def load_basis_payload(payload: Mapping[str, object]) -> JointAdapterBasis:
    if payload.get("format") != "daph_joint_adapter_basis_v1":
        raise ValueError("unsupported adapter basis payload format")
    raw_modules = payload.get("modules")
    if not isinstance(raw_modules, Mapping) or not raw_modules:
        raise ValueError("basis payload has no modules")
    module_factors: dict[str, tuple[Tensor, Tensor]] = {}
    for name, item in raw_modules.items():
        if not isinstance(name, str) or not isinstance(item, Mapping):
            raise TypeError("invalid basis module payload")
        a, b = item.get("a"), item.get("b")
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise TypeError(f"basis module {name!r} is missing tensor A/B")
        module_factors[name] = (a, b)
    return JointAdapterBasis(module_factors)
