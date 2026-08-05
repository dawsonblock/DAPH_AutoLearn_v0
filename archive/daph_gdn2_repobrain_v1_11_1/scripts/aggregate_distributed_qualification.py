from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p):
    x=json.loads(Path(p).read_text())
    if not isinstance(x,dict): raise SystemExit(f'{p} must contain object')
    return x

def main():
    ap=argparse.ArgumentParser(description='Aggregate FSDP and ZeRO-3 qualification artifacts')
    ap.add_argument('--fsdp',required=True); ap.add_argument('--zero3',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    f,z=load(a.fsdp),load(a.zero3); passed=f.get('passed') is True and z.get('passed') is True
    out={'data_origin':'empirical','experiment_name':'distributed_qualification','fsdp':f,'zero3':z,'passed':passed}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if not passed: raise SystemExit('distributed qualification FAILED')
    print('distributed qualification PASS')
if __name__=='__main__': main()
