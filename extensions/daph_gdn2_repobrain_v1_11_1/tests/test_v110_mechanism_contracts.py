import asyncio
import torch
import torch.nn as nn
from daph_ext.repobrain.adapter_context import use_adapter_context
from daph_ext.repobrain.layer import RepoLoRALinear
from daph_ext.repobrain.repo2lora import LoRAFactors

def _factors(batch=None, value=1.0):
    if batch is None:
        a=torch.full((1,2,4), value)
        b=torch.full((1,3,2), value)
    else:
        a=torch.full((batch,1,2,4), value)
        b=torch.full((batch,1,3,2), value)
    return LoRAFactors(a=a,b=b,alpha=2.0)

def test_batched_repoloRA_matches_serial_reference():
    torch.manual_seed(2)
    base=nn.Linear(4,3)
    layer=RepoLoRALinear(base,0,module_key="q_proj")
    x=torch.randn(3,5,4)
    bat=_factors(batch=3,value=0.2)
    with use_adapter_context({"q_proj":bat},scale=0.75):
        got=layer(x)
    refs=[]
    for i in range(3):
        single=LoRAFactors(a=bat.a[i],b=bat.b[i],alpha=bat.alpha)
        with use_adapter_context({"q_proj":single},scale=0.75):
            refs.append(layer(x[i:i+1])[0])
    assert torch.allclose(got,torch.stack(refs),atol=1e-6,rtol=1e-6)

def test_batched_repoloRA_rejects_factor_batch_mismatch():
    layer=RepoLoRALinear(nn.Linear(4,3),0,module_key="q_proj")
    x=torch.randn(2,4)
    with use_adapter_context({"q_proj":_factors(batch=3,value=0.1)}):
        try: layer(x)
        except ValueError as e: assert "factor batch size" in str(e)
        else: raise AssertionError("expected batch mismatch failure")

def test_adapter_context_isolated_across_async_tasks():
    layer=RepoLoRALinear(nn.Linear(4,3,bias=False),0,module_key="q_proj")
    x=torch.ones(1,4)
    async def one(v):
        with use_adapter_context({"q_proj":_factors(value=v)}):
            await asyncio.sleep(0)
            return layer(x).detach()
    async def run():
        return await asyncio.gather(one(0.1),one(0.3),one(0.7))
    outs=asyncio.run(run())
    assert not torch.allclose(outs[0],outs[1])
    assert not torch.allclose(outs[1],outs[2])
