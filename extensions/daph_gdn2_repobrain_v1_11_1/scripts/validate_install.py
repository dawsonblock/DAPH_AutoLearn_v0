from __future__ import annotations

from pathlib import Path

import torch

from daph_ext.config import DAPHIntegrationConfig
from daph_ext.mixers.gdn2_reference import gdn2_recurrence_reference
from daph_ext.mixers.router import alternating_gdn2_attention
from daph_ext.repobrain.repo2lora import ModuleShape, Repo2LoRALite


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = DAPHIntegrationConfig.from_yaml(root / "configs" / "daph_qwen06b_research.yaml")
    plan = alternating_gdn2_attention(cfg.model.num_layers, cfg.gdn2.every_n_layers)
    print("first 8 mixer kinds:", [x.value for x in plan.kinds[:8]])

    b, t, h, kdim, vdim = 1, 4, 2, 3, 5
    q = torch.randn(b, t, h, kdim) * 0.1
    k = torch.randn(b, t, h, kdim) * 0.1
    v = torch.randn(b, t, h, vdim) * 0.1
    erase = torch.sigmoid(torch.randn_like(k))
    write = torch.sigmoid(torch.randn_like(v))
    log_decay = -torch.nn.functional.softplus(torch.randn_like(k)) * 0.05
    y, state = gdn2_recurrence_reference(k, v, erase, write, log_decay, q=q)
    print("reference recurrence:", tuple(y.shape), tuple(state.shape), "finite=", bool(torch.isfinite(y).all()))

    module_shapes = {
        "q_proj": ModuleShape(1024, 1024),
        "v_proj": ModuleShape(1024, 1024),
        "o_proj": ModuleShape(1024, 1024),
        "down_proj": ModuleShape(3072, 1024),
    }
    gen = Repo2LoRALite(cfg.repobrain, module_shapes)
    repo = torch.randn(1, cfg.repobrain.repo_embedding_dim)
    factors = gen(repo)
    print("generated targets:", sorted(factors))
    print("q A/B:", tuple(factors["q_proj"].a.shape), tuple(factors["q_proj"].b.shape))
    print("validation OK")


if __name__ == "__main__":
    main()
