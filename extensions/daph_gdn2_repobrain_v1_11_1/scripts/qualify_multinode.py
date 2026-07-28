from __future__ import annotations
import argparse,json,os,socket
from pathlib import Path
import torch
import torch.distributed as dist

def main():
    ap=argparse.ArgumentParser(description='torchrun multi-node NCCL collective/continuity qualification')
    ap.add_argument('--output',required=True); ap.add_argument('--steps',type=int,default=100); a=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    dist.init_process_group('nccl'); rank=dist.get_rank(); world=dist.get_world_size(); local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); dev=torch.device('cuda',local)
    if world < 2: raise SystemExit('multi-node qualification requires world_size > 1')
    hosts=[None for _ in range(world)]; dist.all_gather_object(hosts,socket.gethostname()); node_count=len(set(hosts))
    if node_count < 2: raise SystemExit('qualification requires at least two distinct hosts')
    ok=True
    for i in range(a.steps):
        x=torch.tensor([rank+i],device=dev,dtype=torch.float32); dist.all_reduce(x); expected=sum(r+i for r in range(world)); ok &= bool(torch.equal(x.cpu(),torch.tensor([expected],dtype=torch.float32)))
    flag=torch.tensor([int(ok)],device=dev); dist.all_reduce(flag,op=dist.ReduceOp.MIN); passed=bool(flag.item())
    if rank==0:
        out={'data_origin':'empirical','experiment_name':'multinode_qualification','world_size':world,'node_count':node_count,'hosts':hosts,'steps':a.steps,'passed':passed}
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    dist.barrier(); dist.destroy_process_group()
    if not passed: raise SystemExit('multinode qualification FAILED')
if __name__=='__main__': main()
