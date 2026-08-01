from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from daph_ext.config import Repo2LoRAConfig
from daph_ext.distributed.placement import AdapterPlacementManager
from daph_ext.models.registry import get_architecture_adapter
from daph_ext.repobrain.layer import patch_repo_lora_linears
from daph_ext.repobrain.repo2lora import Repo2LoRALite


class CompositeRepoBrain(nn.Module):
    def __init__(self, base, hyper, wrappers):
        super().__init__()
        self.base = base
        self.hyper = hyper
        self._wrappers = wrappers
        self._placement = AdapterPlacementManager()

    def forward(self, repo_embedding, input_ids, labels):
        factors = self.hyper(repo_embedding)
        self._placement.apply(self._wrappers, factors)
        logits = self.base(input_ids=input_ids).logits.float()
        loss = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.shape[-1]),
            labels[..., 1:].reshape(-1),
        )
        return loss


def _assert_fsdp_compatible_parameters(module: nn.Module) -> None:
    """Fail before FSDP wrapping if any parameter is zero-dimensional.

    Current FSDP rejects scalar parameters. Catching this here gives a precise
    error without paying the cost of model wrapping/NCCL setup only to fail
    inside FSDP internals.
    """
    scalar = [name for name, parameter in module.named_parameters() if parameter.ndim == 0]
    if scalar:
        raise ValueError(
            "FSDP-incompatible scalar parameters detected: " + ", ".join(sorted(scalar))
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="torchrun FSDP qualification for end-to-end RepoBrain dynamic factors"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--repo-embedding-dim", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("install the 'real' extra") from exc
    if not torch.cuda.is_available():
        raise SystemExit("FSDP qualification requires CUDA")

    initialized_here = False
    try:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
            initialized_here = True

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        arch = get_architecture_adapter(base)
        num_layers = arch.num_layers(base)
        shapes = arch.module_shapes(base)
        targets = tuple(
            t for t in ("q_proj", "v_proj", "o_proj", "down_proj") if t in shapes
        )
        if not targets:
            raise RuntimeError("no RepoBrain targets found")

        wrappers = patch_repo_lora_linears(
            base,
            target_modules=targets,
            num_layers=num_layers,
            groups=min(4, num_layers),
        )
        for p0 in base.parameters():
            p0.requires_grad_(False)

        cfg = Repo2LoRAConfig(
            repo_embedding_dim=args.repo_embedding_dim,
            layer_groups=min(4, num_layers),
            target_modules=targets,
        )
        hyper = Repo2LoRALite(cfg, {k: shapes[k] for k in targets})
        composite = CompositeRepoBrain(base, hyper, wrappers).to(device)
        _assert_fsdp_compatible_parameters(composite)
        model = FSDP(composite, use_orig_params=True, device_id=device)

        repo_embedding = torch.randn(1, args.repo_embedding_dim, device=device)
        vocab = int(getattr(model.module.base.config, "vocab_size", 32000))
        input_ids = torch.randint(0, vocab, (1, args.seq_len), device=device)
        labels = input_ids.clone()
        loss = model(repo_embedding, input_ids, labels)
        loss.backward()
        trainable_grads = [
            p.grad for p in model.module.hyper.parameters() if p.requires_grad
        ]
        local_ok = torch.tensor(
            [
                int(
                    torch.isfinite(loss).item()
                    and any(
                        g is not None and torch.isfinite(g).all()
                        for g in trainable_grads
                    )
                )
            ],
            device=device,
        )
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        if not bool(local_ok.item()):
            raise RuntimeError("FSDP end-to-end RepoBrain qualification failed")

        if local_rank == 0:
            payload = {
                "data_origin": "empirical",
                "experiment_name": "fsdp_repobrain",
                "model": args.model,
                "world_size": dist.get_world_size(),
                "loss": float(loss.detach().cpu()),
                "passed": True,
            }
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            print("FSDP RepoBrain end-to-end PASS")
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
