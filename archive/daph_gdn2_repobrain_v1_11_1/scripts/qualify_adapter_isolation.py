from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn

from daph_ext.repobrain.adapter_context import use_adapter_context
from daph_ext.repobrain.layer import RepoLoRALinear
from daph_ext.repobrain.repo2lora import LoRAFactors


def make_factors(scale: float, dim: int, rank: int = 2) -> LoRAFactors:
    a = torch.ones(1, rank, dim) * scale
    b = torch.ones(1, dim, rank) * scale
    return LoRAFactors(a=a, b=b, alpha=float(rank))

def main() -> None:
    p=argparse.ArgumentParser(description="Concurrent request-scoped RepoLoRA isolation qualification")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output", required=True)
    args=p.parse_args()
    if args.iterations <= 0 or args.workers <= 1:
        raise SystemExit("iterations must be >0 and workers >1")
    dim=8
    base=nn.Linear(dim, dim, bias=False); nn.init.eye_(base.weight)
    layer=RepoLoRALinear(base, 0, module_key="q_proj").eval()
    x=torch.arange(dim, dtype=torch.float32).unsqueeze(0)
    fa=make_factors(0.1, dim); fb=make_factors(0.2, dim)
    with use_adapter_context({"q_proj":fa}): ref_a=layer(x).detach()
    with use_adapter_context({"q_proj":fb}): ref_b=layer(x).detach()
    def one(i:int):
        f, ref = (fa,ref_a) if i%2==0 else (fb,ref_b)
        with use_adapter_context({"q_proj":f}):
            y=layer(x).detach()
        return bool(torch.equal(y, ref))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ok=list(ex.map(one, range(args.iterations)))
    passed=all(ok)
    payload={"data_origin":"empirical","experiment_name":"adapter_isolation",
             "iterations":args.iterations,"workers":args.workers,"passed":passed,
             "failures":len(ok)-sum(ok)}
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n")
    if not passed: raise SystemExit("adapter isolation FAILED")
    print("adapter isolation PASS")
if __name__=="__main__": main()
