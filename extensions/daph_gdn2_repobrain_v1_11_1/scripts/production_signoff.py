from __future__ import annotations
import argparse,json,math
from pathlib import Path

def _load(path: str) -> dict:
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict): raise ValueError(f'{path} must contain a JSON object')
    return data

def _finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))

def main():
    p=argparse.ArgumentParser(description='Fail-closed production qualification sign-off')
    for arg in ('gdn2-hardware','gdn2-cache','training-eval','calibration','sandbox','distributed','provenance','fisher-comparison','adapter-isolation','recovery','multinode'):
        p.add_argument('--'+arg,required=True,dest=arg.replace('-','_'))
    p.add_argument('--output',required=True); a=p.parse_args()
    artifacts={n:_load(getattr(a,n)) for n in ('gdn2_hardware','gdn2_cache','training_eval','calibration','sandbox','distributed','provenance','fisher_comparison','adapter_isolation','recovery','multinode')}
    cal=artifacts['calibration']; ci=cal.get('bootstrap_ci_95'); sand=artifacts['sandbox']; dista=artifacts['distributed']; iso=artifacts['adapter_isolation']; multi=artifacts['multinode']
    checks={
      'gdn2_hardware': artifacts['gdn2_hardware'].get('passed') is True,
      'gdn2_cache': artifacts['gdn2_cache'].get('passed') is True and isinstance(artifacts['gdn2_cache'].get('checkpoint_relative_l2'),dict),
      'training_eval': artifacts['training_eval'].get('passed') is True,
      'calibration': _finite(cal.get('max_kl_drift')) and cal.get('safe_count',0)>=30 and isinstance(ci,list) and len(ci)==2 and all(_finite(x) for x in ci) and ci[0] <= cal['max_kl_drift'] <= ci[1],
      'sandbox': sand.get('passed') is True and isinstance(sand.get('image_fingerprint'),str) and bool(sand.get('image_fingerprint')) and len(sand.get('required_cases',[]))>=10,
      'distributed': dista.get('passed') is True and isinstance(dista.get('fsdp'),dict) and dista['fsdp'].get('passed') is True and isinstance(dista.get('zero3'),dict) and dista['zero3'].get('passed') is True,
      'provenance': artifacts['provenance'].get('passed') is True,
      'fisher_comparison': artifacts['fisher_comparison'].get('fisher_pareto_superior_to_scaling') is True,
      'adapter_isolation': iso.get('passed') is True and iso.get('failures')==0 and iso.get('iterations',0)>=1000,
      'recovery': artifacts['recovery'].get('passed') is True and artifacts['recovery'].get('cache_schema_version')==2,
      'multinode': multi.get('passed') is True and multi.get('node_count',0)>=2 and multi.get('world_size',0)>=2,
    }
    passed=all(checks.values()); payload={'release_status':'PASS' if passed else 'FAIL','checks':checks}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if not passed: raise SystemExit('production sign-off FAILED: '+', '.join(k for k,v in checks.items() if not v))
    print('production sign-off PASS')
if __name__=='__main__': main()
