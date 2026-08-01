from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from daph_ext.distributed.checkpoint import cache_state_dict, load_cache_state_dict, evolution_checkpoint, restore_evolution_checkpoint
from daph_ext.mixers.cache import HybridSequenceCache, CacheTensor
from daph_ext.repobrain.evolution import RepoEvolutionState


def main():
    ap=argparse.ArgumentParser(description='Checkpoint/cache restart recovery qualification')
    ap.add_argument('--output',required=True); ap.add_argument('--device',default='cpu'); ap.add_argument('--dtype',default='fp32',choices=['fp32','bf16','fp16'])
    a=ap.parse_args(); dt={'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}[a.dtype]
    device=torch.device(a.device)
    cache=HybridSequenceCache(); k=torch.randn(2,3,4); v=torch.randn(2,3,4); cache.set_attention(0,k,v)
    cache.set_recurrent(1,{'state':CacheTensor(torch.randn(2,2,3,4),batch_axis=0),'pos':torch.tensor([7],dtype=torch.long)})
    state=cache_state_dict(cache); restored=load_cache_state_dict(state,device=device,dtype=dt)
    cache_ok=(restored.attention_kv[0][0].dtype==dt and restored.recurrent_states[1]['state'].tensor.dtype==dt and restored.recurrent_states[1]['pos'].dtype==torch.long)
    evo=RepoEvolutionState(8,4); snap=torch.randn(8); h=evo.initialize(snap); payload=evolution_checkpoint(evo,h)
    evo2=RepoEvolutionState(8,4); h2=restore_evolution_checkpoint(evo2,payload,device=device,dtype=dt)
    params_ok=all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(evo.state_dict().values(),evo2.state_dict().values()))
    evolution_ok=params_ok and torch.allclose(h.to(dtype=dt).cpu(),h2.cpu())
    passed=bool(cache_ok and evolution_ok)
    out={'data_origin':'empirical','experiment_name':'restart_recovery','cache_schema_version':state['schema_version'],'cache_roundtrip':cache_ok,'evolution_roundtrip':evolution_ok,'passed':passed}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    if not passed: raise SystemExit('recovery qualification FAILED')
    print('recovery qualification PASS')
if __name__=='__main__': main()
