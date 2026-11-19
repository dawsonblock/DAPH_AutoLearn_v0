from __future__ import annotations
import argparse, json
from pathlib import Path

REQUIRED_ARMS={"base","rag","shared_lora","per_repo_lora","repo2lora","repo2lora_plus_rag"}
REQUIRED_META={"checkpoint","dataset_hash","seed"}

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--qualification",action="store_true")
    a=ap.parse_args()
    d=json.loads(Path(a.artifact).read_text())
    errors=[]
    if d.get("data_origin")!="empirical": errors.append("data_origin must be empirical")
    missing=REQUIRED_ARMS-set(d.get("arms",{}))
    if missing: errors.append(f"missing arms: {sorted(missing)}")
    meta=d.get("metadata",{})
    mm=REQUIRED_META-set(meta)
    if mm: errors.append(f"missing metadata: {sorted(mm)}")
    if not d.get("test_policy_frozen_before_evaluation",False): errors.append("test policy was not frozen before evaluation")
    s=d.get("split",{})
    mins=(100,20,20) if a.qualification else (20,5,5)
    vals=(int(s.get("train_repositories",0)),int(s.get("validation_repositories",0)),int(s.get("test_repositories",0)))
    for name,v,m in zip(("train","validation","test"),vals,mins):
        if v<m: errors.append(f"{name} repositories {v} < required {m}")
    out={"passed":not errors,"errors":errors}
    print(json.dumps(out,indent=2))
    return 0 if not errors else 1
if __name__=="__main__": raise SystemExit(main())
