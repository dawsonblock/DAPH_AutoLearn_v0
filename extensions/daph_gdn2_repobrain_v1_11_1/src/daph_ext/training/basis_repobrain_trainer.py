from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from daph_ext.repobrain.adapter_context import use_adapter_context
from daph_ext.repobrain.basis_repo2lora import BasisRepoBrain
from daph_ext.repobrain.layer import RepoLoRALinear
from .repobrain_trainer import _extract_logits, _primary_device, freeze_module


@dataclass(frozen=True)
class BasisTrainStepMetrics:
    total_loss: float
    task_loss: float
    coefficient_loss: float
    grad_norm: float
    coefficient_l2: float
    valid_tokens: int


class BasisRepoBrainTrainer:
    """Train repository -> basis-coordinate inference with optional LoRA teachers.

    This combines downstream task CE with direct supervision from coefficients
    obtained when fitting the adapter basis to independently trained per-repo LoRAs.
    """

    def __init__(
        self,
        base_model: nn.Module,
        repobrain: BasisRepoBrain,
        wrappers: Mapping[str, RepoLoRALinear],
        *,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        task_weight: float = 1.0,
        coefficient_weight: float = 1.0,
        ignore_index: int = -100,
    ) -> None:
        if lr <= 0 or grad_clip <= 0 or weight_decay < 0:
            raise ValueError("lr/grad_clip must be > 0 and weight_decay >= 0")
        if task_weight < 0 or coefficient_weight < 0 or (task_weight + coefficient_weight) <= 0:
            raise ValueError("loss weights must be non-negative with at least one > 0")
        self.base_model = base_model
        self.repobrain = repobrain
        self.wrappers = dict(wrappers)
        self.grad_clip = float(grad_clip)
        self.task_weight = float(task_weight)
        self.coefficient_weight = float(coefficient_weight)
        self.ignore_index = int(ignore_index)
        freeze_module(base_model)
        self.repobrain.train()
        trainable = [p for p in self.repobrain.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("BasisRepoBrain has no trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    def train_step(
        self,
        repo_embedding: Tensor,
        model_batch: Mapping[str, Tensor],
        *,
        teacher_coefficients: Tensor | None = None,
    ) -> BasisTrainStepMetrics:
        if "input_ids" not in model_batch:
            raise KeyError("model_batch requires input_ids")
        repodev = _primary_device(self.repobrain)
        basedev = _primary_device(self.base_model)
        repo_embedding = repo_embedding.to(repodev)
        moved = {k: (v.to(basedev) if isinstance(v, Tensor) else v) for k, v in model_batch.items()}
        if teacher_coefficients is not None:
            teacher_coefficients = teacher_coefficients.to(repodev)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.repobrain(repo_embedding)

        task_loss = torch.zeros((), device=repodev)
        valid_tokens = 0
        if self.task_weight > 0:
            inputs = {k: v for k, v in moved.items() if k != "labels"}
            labels = moved.get("labels", moved["input_ids"])
            with use_adapter_context(output.factors_by_module):
                logits = _extract_logits(self.base_model(**inputs))
            shift_logits = logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous().clone()
            if "labels" not in moved and isinstance(inputs.get("attention_mask"), Tensor):
                shift_mask = inputs["attention_mask"][..., 1:].to(dtype=torch.bool)
                shift_labels[~shift_mask] = self.ignore_index
            valid_tokens = int((shift_labels != self.ignore_index).sum().item())
            if valid_tokens == 0:
                raise ValueError("training batch contains no valid target tokens")
            task_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=self.ignore_index,
            ).to(repodev)

        coefficient_loss = torch.zeros((), device=repodev)
        if self.coefficient_weight > 0:
            if teacher_coefficients is None:
                raise ValueError("teacher_coefficients required when coefficient_weight > 0")
            if teacher_coefficients.shape != output.coefficients.shape:
                raise ValueError(
                    f"teacher_coefficients shape {tuple(teacher_coefficients.shape)} != "
                    f"prediction shape {tuple(output.coefficients.shape)}"
                )
            coefficient_loss = F.mse_loss(output.coefficients.float(), teacher_coefficients.float())

        total = self.task_weight * task_loss + self.coefficient_weight * coefficient_loss
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite BasisRepoBrain loss")
        total.backward()
        trainable = [p for p in self.repobrain.parameters() if p.requires_grad]
        grad_norm_t = torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)
        if not torch.isfinite(grad_norm_t):
            raise FloatingPointError("non-finite BasisRepoBrain gradient norm")
        self.optimizer.step()
        coeff_l2 = float(torch.linalg.vector_norm(output.coefficients.detach().float(), dim=-1).mean().item())
        return BasisTrainStepMetrics(
            total_loss=float(total.detach().item()),
            task_loss=float(task_loss.detach().item()),
            coefficient_loss=float(coefficient_loss.detach().item()),
            grad_norm=float(grad_norm_t.item()),
            coefficient_l2=coeff_l2,
            valid_tokens=valid_tokens,
        )
