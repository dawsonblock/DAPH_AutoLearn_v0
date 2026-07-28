from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class FisherDiagonal:
    values: dict[str, Tensor]
    samples: int
    parameter_samples: dict[str, int] | None = None


def _slice_item(value: Any, index: int, batch_size: int) -> Any:
    """Slice one example from common batched Python/container inputs."""
    if isinstance(value, Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index : index + 1]
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return value[index : index + 1]
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[index : index + 1]
    if isinstance(value, Mapping):
        return {key: _slice_item(item, index, batch_size) for key, item in value.items()}
    return value


def empirical_fisher_diagonal(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    *,
    loss_fn: Callable[[nn.Module, Mapping[str, Any]], Tensor],
    parameter_filter: Callable[[str, nn.Parameter], bool] | None = None,
    per_sample: bool = True,
    eval_mode: bool = True,
) -> FisherDiagonal:
    """Accumulate a diagonal gradient-square Fisher estimate in FP32.

    By default this computes the usual empirical per-example gradient-square
    estimator (``per_sample=True``). ``per_sample=False`` is an explicitly
    opt-in, cheaper batch-gradient-square approximation and should not be used
    for scientific acceptance/rejection decisions. ``eval_mode=True`` disables
    dropout during calibration and restores the model's prior training mode.
    Each parameter is normalized only over backward passes where it received a
    gradient, which is required for conditional/MoE execution.
    """
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (parameter_filter is None or parameter_filter(name, parameter))
    }
    if not selected:
        raise ValueError("no trainable parameters selected for Fisher accumulation")

    accum = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
        for name, parameter in selected.items()
    }
    parameter_counts = {name: 0 for name in selected}
    total_samples = 0

    def accumulate(one_batch: Mapping[str, Any]) -> None:
        nonlocal total_samples
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, one_batch)
        if loss.ndim != 0:
            loss = loss.mean()
        loss.backward()

        for name, parameter in selected.items():
            if parameter.grad is not None:
                accum[name].add_(parameter.grad.detach().float().square())
                parameter_counts[name] += 1
        total_samples += 1

    was_training = model.training
    if eval_mode:
        model.eval()
    try:
        for batch in batches:
            if per_sample:
                tensor_values = [
                    value
                    for value in batch.values()
                    if isinstance(value, Tensor) and value.ndim > 0
                ]
                if not tensor_values:
                    raise ValueError("per_sample Fisher requires batched tensor inputs")
                batch_size = tensor_values[0].shape[0]
                if any(value.shape[0] != batch_size for value in tensor_values):
                    raise ValueError("inconsistent batch dimension")
                for index in range(batch_size):
                    sliced = {
                        key: _slice_item(value, index, batch_size)
                        for key, value in batch.items()
                    }
                    accumulate(sliced)
            else:
                accumulate(batch)
    finally:
        model.train(was_training)

    if total_samples == 0:
        raise ValueError("no Fisher samples accumulated")

    final_values: dict[str, Tensor] = {}
    for name, value in accum.items():
        count = parameter_counts[name]
        if count == 0:
            # A selected parameter that is never active has no empirical Fisher
            # evidence in this calibration stream. Keep its diagonal at zero.
            final_values[name] = value
        else:
            final_values[name] = value / count

    return FisherDiagonal(
        values=final_values,
        samples=total_samples,
        parameter_samples=parameter_counts,
    )
