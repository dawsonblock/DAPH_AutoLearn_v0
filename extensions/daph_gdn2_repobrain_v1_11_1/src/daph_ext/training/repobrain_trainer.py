from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from daph_ext.repobrain.layer import RepoLoRALinear, apply_generated_factors
from daph_ext.repobrain.materialize import materialize_group_delta
from daph_ext.repobrain.repo2lora import Repo2LoRALite


@dataclass(frozen=True)
class TrainStepMetrics:
    loss: float
    grad_norm: float
    valid_tokens: int
    delta_frobenius: dict[str, float]


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    raise TypeError("model output does not expose logits")


def _primary_device(module: nn.Module) -> torch.device:
    for p in module.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


class RepoBrainTrainer:
    """Single-device trainer for Repo2LoRALite with a frozen patched base model.

    Distributed qualification uses the dedicated FSDP/ZeRO scripts instead of
    relying on this primary-device placement helper.
    """

    def __init__(
        self,
        base_model: nn.Module,
        hypernetwork: Repo2LoRALite,
        wrappers: Mapping[str, RepoLoRALinear],
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        if lr <= 0 or grad_clip <= 0 or weight_decay < 0:
            raise ValueError("lr/grad_clip must be > 0 and weight_decay >= 0")
        self.base_model = base_model
        self.hypernetwork = hypernetwork
        self.wrappers = dict(wrappers)
        self.grad_clip = float(grad_clip)
        self.ignore_index = int(ignore_index)
        freeze_module(base_model)
        self.hypernetwork.train()
        for p in self.hypernetwork.parameters():
            p.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self.hypernetwork.parameters(), lr=lr, weight_decay=weight_decay)
        if any(p.requires_grad for p in self.base_model.parameters()):
            raise RuntimeError("base model must be frozen")

    def train_step(self, repo_embedding: Tensor, model_batch: Mapping[str, Tensor]) -> TrainStepMetrics:
        if "input_ids" not in model_batch:
            raise KeyError("model_batch requires input_ids")
        hyper_device = _primary_device(self.hypernetwork)
        base_device = _primary_device(self.base_model)
        repo_embedding = repo_embedding.to(hyper_device)
        moved = {k: (v.to(base_device) if isinstance(v, Tensor) else v) for k, v in model_batch.items()}

        self.optimizer.zero_grad(set_to_none=True)
        factors = self.hypernetwork(repo_embedding)
        apply_generated_factors(self.wrappers, factors, strict=True)
        inputs = {k: v for k, v in moved.items() if k != "labels"}
        labels = moved.get("labels", moved["input_ids"])
        logits = _extract_logits(self.base_model(**inputs))
        shift_logits = logits[..., :-1, :].float().contiguous()
        shift_labels = labels[..., 1:].contiguous().clone()
        if "labels" not in moved and isinstance(inputs.get("attention_mask"), Tensor):
            shift_mask = inputs["attention_mask"][..., 1:].to(dtype=torch.bool)
            shift_labels[~shift_mask] = self.ignore_index
        valid_tokens = int((shift_labels != self.ignore_index).sum().item())
        if valid_tokens == 0:
            raise ValueError("training batch contains no valid target tokens")
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction="mean",
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite RepoBrain training loss")
        loss.backward()
        grad_norm_t = torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), self.grad_clip)
        if not torch.isfinite(grad_norm_t):
            raise FloatingPointError("non-finite RepoBrain gradient norm")
        grad_norm = float(grad_norm_t.item())
        self.optimizer.step()

        fro: dict[str, float] = {}
        for name, f in factors.items():
            vals = [float(torch.linalg.vector_norm(materialize_group_delta(f, g).float()).item()) for g in range(f.groups)]
            fro[name] = sum(vals) / len(vals)
        return TrainStepMetrics(float(loss.detach().item()), grad_norm, valid_tokens, fro)
