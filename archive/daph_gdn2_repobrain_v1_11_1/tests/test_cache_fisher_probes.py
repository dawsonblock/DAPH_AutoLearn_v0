from types import SimpleNamespace

import torch
import torch.nn as nn

from daph_ext.geometry.fisher import empirical_fisher_diagonal
from daph_ext.mixers.cache import HybridSequenceCache
from daph_ext.probes.concrete import ExecutionSummary, CodeExecutionProbe, LogitKLDriftProbe, ValidationLossProbe


def test_hybrid_cache_reset_and_to():
    cache = HybridSequenceCache()
    cache.set_attention(0, torch.randn(1, 2), torch.randn(1, 2))
    cache.set_recurrent(1, torch.randn(1, 2, 3))
    cache.to(dtype=torch.float64)
    assert cache.get_attention(0)[0].dtype == torch.float64
    assert cache.get_recurrent(1).dtype == torch.float64
    cache.reset()
    assert not cache.attention_kv and not cache.recurrent_states


class TinyLM(nn.Module):
    def __init__(self, vocab=7):
        super().__init__()
        self.emb = nn.Embedding(vocab, 5)
        self.head = nn.Linear(5, vocab)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))


def test_loss_and_kl_probes_run():
    torch.manual_seed(0)
    base = TinyLM()
    candidate = TinyLM()
    candidate.load_state_dict(base.state_dict())
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    loss_result = ValidationLossProbe([batch]).run(base, candidate)
    kl_result = LogitKLDriftProbe([batch]).run(base, candidate)
    assert abs(loss_result.regression) < 1e-7
    assert kl_result.candidate < 1e-7


def test_execution_probe_runner_contract():
    scores = {"base": ExecutionSummary(8, 10), "candidate": ExecutionSummary(9, 10)}
    probe = CodeExecutionProbe(lambda key: scores[key])
    result = probe.run("base", "candidate")
    assert result.baseline == 0.8
    assert result.candidate == 0.9


def test_empirical_fisher_diagonal():
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    batches = [{"x": torch.randn(4, 3), "y": torch.randn(4, 1)}]

    def loss_fn(m, batch):
        return ((m(batch["x"]) - batch["y"]) ** 2).mean()

    fisher = empirical_fisher_diagonal(model, batches, loss_fn=loss_fn, per_sample=True)
    assert fisher.samples == 4
    assert set(fisher.values) == {"weight", "bias"}
    assert all(torch.isfinite(v).all() and (v >= 0).all() for v in fisher.values.values())
