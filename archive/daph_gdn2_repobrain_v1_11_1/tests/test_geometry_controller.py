import torch

from daph_ext.config import ControllerConfig
from daph_ext.geometry.metrics import geometry_metrics
from daph_ext.probes.base import ProbeResult
from daph_ext.probes.controller import AdapterAction, AdapterController


def test_geometry_metrics_identity():
    x = torch.tensor([1.0, -2.0, 3.0])
    g = geometry_metrics(x, x)
    assert abs(g.cosine - 1.0) < 1e-6
    assert abs(g.norm_ratio - 1.0) < 1e-6
    assert g.sign_conflict == 0.0


def test_geometry_alone_cannot_accept_or_project():
    controller = AdapterController(ControllerConfig())
    g = geometry_metrics(torch.ones(4), torch.ones(4))
    decision = controller.decide([], [g])
    assert decision.action == AdapterAction.REJECT


def test_execution_regression_rejects():
    controller = AdapterController(ControllerConfig(max_execution_regression=0.01))
    probes = [ProbeResult("execution_pass_rate", baseline=0.90, candidate=0.85, higher_is_better=True)]
    decision = controller.decide(probes)
    assert decision.action == AdapterAction.REJECT


def test_safe_adapter_accepts():
    controller = AdapterController(ControllerConfig())
    probes = [
        ProbeResult("repo_loss", baseline=1.0, candidate=0.8, higher_is_better=False),
        ProbeResult("execution_pass_rate", baseline=0.8, candidate=0.85, higher_is_better=True),
    ]
    decision = controller.decide(probes)
    assert decision.action == AdapterAction.ACCEPT
