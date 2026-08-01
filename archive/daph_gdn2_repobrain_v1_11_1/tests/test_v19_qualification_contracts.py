import json
from pathlib import Path
import subprocess,sys
import pytest
from daph_ext.experiments.splits import RepositoryIdentity, validate_repository_lineage


def test_lineage_rejects_cross_split_family():
    with pytest.raises(ValueError,match='family'):
        validate_repository_lineage([RepositoryIdentity('a','train',family='f'),RepositoryIdentity('b','test',family='f')])


def test_signoff_requires_both_distributed_backends(tmp_path: Path):
    good={'passed':True};
    artifacts={
      'gdn2_hardware':good,
      'gdn2_cache':{'passed':True,'checkpoint_relative_l2':{'8':0.0}},
      'training_eval':good,
      'calibration':{'max_kl_drift':0.1,'safe_count':30,'bootstrap_ci_95':[0.05,0.15]},
      'sandbox':{'passed':True,'image_fingerprint':'sha256:x','required_cases':list(range(10))},
      'distributed':{'passed':True,'fsdp':{'passed':True},'zero3':{'passed':False}},
      'provenance':good,'fisher_comparison':{'fisher_pareto_superior_to_scaling':True},
      'adapter_isolation':{'passed':True,'failures':0,'iterations':1000},
      'recovery':{'passed':True,'cache_schema_version':2},
      'multinode':{'passed':True,'node_count':2,'world_size':4},
    }
    paths={}
    for k,v in artifacts.items():
        p=tmp_path/f'{k}.json'; p.write_text(json.dumps(v)); paths[k]=p
    cmd=[sys.executable,'scripts/production_signoff.py']
    for k,p in paths.items(): cmd += ['--'+k.replace('_','-'),str(p)]
    cmd += ['--output',str(tmp_path/'out.json')]
    proc=subprocess.run(cmd,capture_output=True,text=True)
    assert proc.returncode != 0
    out=json.loads((tmp_path/'out.json').read_text())
    assert out['checks']['distributed'] is False


def test_recovery_script_roundtrip(tmp_path: Path):
    out=tmp_path/'recovery.json'
    p=subprocess.run([sys.executable,'scripts/qualify_recovery.py','--output',str(out)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads(out.read_text())
    assert data['passed'] is True and data['cache_schema_version']==2
