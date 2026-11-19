import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from torch import nn

from daph_ext.repobrain.adapter_context import use_adapter_context
from daph_ext.repobrain.layer import RepoLoRALinear
from daph_ext.repobrain.repo2lora import LoRAFactors
from daph_ext.mixers.cache import HybridSequenceCache, CacheTensor
from daph_ext.calibration.controller import CalibrationRecord, calibrate_kl_envelope
from daph_ext.experiments.splits import RepositoryIdentity, validate_repository_lineage


def _f(v):
    a=torch.ones(1,1,2)*v; b=torch.ones(1,2,1)*v
    return LoRAFactors(a=a,b=b,alpha=1.0)

def test_request_scoped_adapter_isolation():
    base=nn.Linear(2,2,bias=False); nn.init.eye_(base.weight)
    layer=RepoLoRALinear(base,0,module_key='q_proj'); x=torch.ones(1,2)
    fa,fb=_f(.1),_f(.2)
    with use_adapter_context({'q_proj':fa}): ra=layer(x).clone()
    with use_adapter_context({'q_proj':fb}): rb=layer(x).clone()
    def run(i):
        with use_adapter_context({'q_proj': fa if i%2==0 else fb}):
            return layer(x).clone()
    with ThreadPoolExecutor(max_workers=4) as ex: ys=list(ex.map(run,range(100)))
    assert all(torch.equal(y, ra if i%2==0 else rb) for i,y in enumerate(ys))

def test_typed_cache_reorder_only_declared_axis():
    c=HybridSequenceCache(); c.set_attention(0, torch.randn(2,1,1), torch.randn(2,1,1))
    c.set_recurrent(0, {'state': CacheTensor(torch.arange(6).reshape(3,2), batch_axis=1),
                        'meta': CacheTensor(torch.tensor([10,11]), batch_axis=None)})
    c.reorder_batch(torch.tensor([1,0]), batch_size=2)
    st=c.get_recurrent(0)
    assert st['state'].tensor.tolist()==[[1,0],[3,2],[5,4]]
    assert st['meta'].tensor.tolist()==[10,11]

def test_calibration_safe_useful_bootstrap():
    rs=[CalibrationRecord(.01+i*.001,0,0,.1,repo_id=f'r{i}') for i in range(30)]
    out=calibrate_kl_envelope(rs,min_safe_count=30,bootstrap_samples=100,seed=1)
    assert out.safe_count==30 and out.repository_count==30 and out.ci_low<=out.threshold<=out.ci_high

def test_lineage_leakage_rejected():
    rec=[RepositoryIdentity('a','train',fork_root='root'),RepositoryIdentity('b','test',fork_root='root')]
    import pytest
    with pytest.raises(ValueError): validate_repository_lineage(rec)
