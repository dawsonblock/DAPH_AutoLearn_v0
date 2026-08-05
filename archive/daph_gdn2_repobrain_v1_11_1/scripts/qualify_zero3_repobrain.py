from __future__ import annotations

import argparse
import torch
import json
from pathlib import Path
import torch.nn.functional as F
from torch import nn

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
        return F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.shape[-1]), labels[..., 1:].reshape(-1))


def main() -> None:
    p = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 end-to-end RepoBrain qualification")
    p.add_argument("--model", required=True)
    p.add_argument("--repo-embedding-dim", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    try:
        import deepspeed
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("install deepspeed and the 'real' extra") from exc

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    arch = get_architecture_adapter(base)
    num_layers = arch.num_layers(base)
    shapes = arch.module_shapes(base)
    targets = tuple(t for t in ("q_proj", "v_proj", "o_proj", "down_proj") if t in shapes)
    if not targets:
        raise RuntimeError("no RepoBrain targets found")
    wrappers = patch_repo_lora_linears(base, target_modules=targets, num_layers=num_layers, groups=min(4, num_layers))
    for p0 in base.parameters():
        p0.requires_grad_(False)
    cfg = Repo2LoRAConfig(repo_embedding_dim=args.repo_embedding_dim, layer_groups=min(4, num_layers), target_modules=targets)
    hyper = Repo2LoRALite(cfg, {k: shapes[k] for k in targets})
    composite = CompositeRepoBrain(base, hyper, wrappers)
    ds_cfg = {
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 3, "stage3_gather_16bit_weights_on_model_save": False},
        "bf16": {"enabled": True},
    }
    trainable = [p for p in composite.parameters() if p.requires_grad]
    engine, _, _, _ = deepspeed.initialize(model=composite, model_parameters=trainable, config=ds_cfg)
    device = engine.device
    vocab = int(getattr(engine.module.base.config, "vocab_size", 32000))
    repo_embedding = torch.randn(1, args.repo_embedding_dim, device=device)
    input_ids = torch.randint(0, vocab, (1, args.seq_len), device=device)
    labels = input_ids.clone()
    loss = engine(repo_embedding, input_ids, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite ZeRO-3 loss")
    engine.backward(loss)
    grads_present = any(p.grad is not None for p in engine.module.hyper.parameters() if p.requires_grad)
    if not grads_present:
        raise RuntimeError("no hypernetwork gradients observed under ZeRO-3")
    engine.step()
    if engine.global_rank == 0:
        payload={"data_origin":"empirical","experiment_name":"zero3_repobrain","model":args.model,"world_size":engine.world_size,"loss":float(loss.detach().cpu()),"passed":True}
        out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print("ZeRO-3 RepoBrain end-to-end PASS")


if __name__ == "__main__":
    main()
