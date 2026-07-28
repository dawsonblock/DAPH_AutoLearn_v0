from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import ProbeResult


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    raise TypeError("model output does not expose logits")


def _get_model_device(model: nn.Module) -> torch.device:
    """Best-effort primary device for ordinary single-device model invocation.

    This intentionally does not attempt to override Accelerate/FSDP device maps.
    Callers using sharded models should provide batches through their framework's
    normal input placement path instead of relying on this helper.
    """
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return torch.device("cpu")
    if parameter.device.type == "meta":
        return torch.device("cpu")
    return parameter.device


def _move_batch(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    exclude: tuple[str, ...] = (),
    tensor_inputs_only: bool = True,
) -> dict[str, Any]:
    """Move model inputs while dropping ordinary dataset metadata by default.

    Repository batches commonly include strings such as ``repo_id`` or
    ``file_path``. Passing those through ``model(**batch)`` causes unrelated
    forward-signature failures, so probes default to Tensor-valued model inputs.
    """
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key in exclude:
            continue
        if isinstance(value, Tensor):
            out[key] = value.to(device)
        elif not tensor_inputs_only:
            out[key] = value
    return out


@contextmanager
def _eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


class ValidationLossProbe:
    """Compare token-weighted next-token cross entropy for two models."""

    def __init__(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        ignore_index: int = -100,
        name: str = "validation_loss",
    ) -> None:
        self.batches = list(batches)
        self.ignore_index = ignore_index
        self.name = name

    @torch.no_grad()
    def _loss_and_tokens(self, model: nn.Module) -> tuple[float, int]:
        total_loss = 0.0
        total_tokens = 0
        device = _get_model_device(model)

        with _eval_mode(model):
            for batch in self.batches:
                if "input_ids" not in batch:
                    raise KeyError("ValidationLossProbe batch requires input_ids")
                if not isinstance(batch["input_ids"], Tensor):
                    raise TypeError("input_ids must be a Tensor")

                inputs = _move_batch(batch, device, exclude=("labels",))
                has_explicit_labels = "labels" in batch
                raw_labels = batch.get("labels", batch["input_ids"])
                if not isinstance(raw_labels, Tensor):
                    raise TypeError("labels must be a Tensor")
                labels = raw_labels.to(device)

                logits = _extract_logits(model(**inputs))
                shift_logits = logits[..., :-1, :].float().contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # When labels are inferred from input_ids, padding is not already
                # encoded as ignore_index. Convert masked targets explicitly.
                if not has_explicit_labels and "attention_mask" in inputs:
                    mask = inputs["attention_mask"]
                    if not isinstance(mask, Tensor):
                        raise TypeError("attention_mask must be a Tensor")
                    shift_mask = mask[..., 1:].to(device=shift_labels.device)
                    if shift_mask.shape != shift_labels.shape:
                        raise ValueError("attention_mask and labels must have matching token dimensions")
                    shift_labels = shift_labels.clone()
                    shift_labels[shift_mask == 0] = self.ignore_index

                loss_sum = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=self.ignore_index,
                    reduction="sum",
                )
                valid = int((shift_labels != self.ignore_index).sum().item())
                total_loss += float(loss_sum.item())
                total_tokens += valid

        if total_tokens == 0:
            raise ValueError("ValidationLossProbe saw no valid target tokens")
        return total_loss / total_tokens, total_tokens

    @torch.no_grad()
    def _loss(self, model: nn.Module) -> float:
        return self._loss_and_tokens(model)[0]

    def run(self, base_model: nn.Module, candidate_model: nn.Module) -> ProbeResult:
        baseline, base_tokens = self._loss_and_tokens(base_model)
        candidate, candidate_tokens = self._loss_and_tokens(candidate_model)
        return ProbeResult(
            self.name,
            baseline,
            candidate,
            higher_is_better=False,
            metadata={"base_tokens": str(base_tokens), "candidate_tokens": str(candidate_tokens)},
        )


class LogitKLDriftProbe:
    """Measure token-weighted forward KL: KL(P_base || P_candidate).

    Models may reside on different devices. Candidate logits are transferred to
    the base-logit device only after each forward pass. Padding tokens are
    excluded when an attention mask is supplied.
    """

    def __init__(
        self,
        batches: Iterable[Mapping[str, Any]],
        *,
        name: str = "kl_drift",
    ) -> None:
        self.batches = list(batches)
        self.name = name

    @torch.no_grad()
    def run(self, base_model: nn.Module, candidate_model: nn.Module) -> ProbeResult:
        total = 0.0
        count = 0
        base_device = _get_model_device(base_model)
        candidate_device = _get_model_device(candidate_model)

        with _eval_mode(base_model), _eval_mode(candidate_model):
            for batch in self.batches:
                base_inputs = _move_batch(batch, base_device, exclude=("labels",))
                candidate_inputs = _move_batch(batch, candidate_device, exclude=("labels",))

                base_logits = _extract_logits(base_model(**base_inputs)).float()
                candidate_logits = _extract_logits(candidate_model(**candidate_inputs)).float()
                if candidate_logits.shape != base_logits.shape:
                    raise ValueError("base/candidate logits shape mismatch")
                candidate_logits = candidate_logits.to(base_logits.device)

                log_p = F.log_softmax(base_logits, dim=-1)
                log_q = F.log_softmax(candidate_logits, dim=-1)
                p = log_p.exp()
                kl = (p * (log_p - log_q)).sum(dim=-1)

                mask = base_inputs.get("attention_mask")
                if isinstance(mask, Tensor):
                    mask = mask.to(device=kl.device, dtype=torch.bool)
                    if mask.shape != kl.shape:
                        raise ValueError("attention_mask and logits must have matching token dimensions")
                    total += float(kl.masked_select(mask).sum().item())
                    count += int(mask.sum().item())
                else:
                    total += float(kl.sum().item())
                    count += kl.numel()

        if count == 0:
            raise ValueError("LogitKLDriftProbe saw no valid tokens")
        value = total / count
        return ProbeResult(
            self.name,
            0.0,
            value,
            higher_is_better=False,
            metadata={"tokens": str(count)},
        )


@dataclass(frozen=True)
class ExecutionSummary:
    passed: int
    total: int

    def __post_init__(self) -> None:
        if self.total < 0 or self.passed < 0 or self.passed > self.total:
            raise ValueError("execution counts must satisfy 0 <= passed <= total")

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


class CodeExecutionProbe:
    """Delegate execution to a supplied sandbox runner.

    ``runner(model_or_adapter)`` must execute the isolated test set and return
    :class:`ExecutionSummary`. Shell/container execution remains outside this
    library's trust boundary.
    """

    def __init__(
        self,
        runner: Callable[[Any], ExecutionSummary],
        *,
        name: str = "execution_pass_rate",
    ) -> None:
        self.runner = runner
        self.name = name

    def run(self, base_candidate: Any, adapted_candidate: Any) -> ProbeResult:
        base = self.runner(base_candidate)
        candidate = self.runner(adapted_candidate)
        return ProbeResult(
            self.name,
            base.pass_rate,
            candidate.pass_rate,
            higher_is_better=True,
            metadata={
                "base_total": str(base.total),
                "candidate_total": str(candidate.total),
            },
        )
