"""Section 11 — TimingProtocol + measure_backend_latency tests."""

from __future__ import annotations

import pytest

from daph_learning.evaluation.timing import (
    LatencyMeasurement,
    TimingProtocol,
    measure_backend_latency,
)


def test_timing_protocol_defaults():
    p = TimingProtocol()
    assert p.warmup_runs == 2
    assert p.measured_runs == 5
    assert p.aggregation == "median"
    assert p.synchronize_cuda is True


def test_timing_protocol_validation():
    with pytest.raises(ValueError):
        TimingProtocol(warmup_runs=-1)
    with pytest.raises(ValueError):
        TimingProtocol(measured_runs=0)
    with pytest.raises(ValueError):
        TimingProtocol(aggregation="bogus")


def test_measure_backend_latency_returns_measurement():
    def fn():
        sum(range(1000))
    result = measure_backend_latency(fn, TimingProtocol(warmup_runs=1, measured_runs=3))
    assert isinstance(result, LatencyMeasurement)
    assert result.n_runs == 3
    assert result.median_ms > 0
    assert result.mean_ms > 0
    assert result.min_ms <= result.median_ms <= result.max_ms
    assert len(result.raw_ms) == 3


def test_measure_backend_latency_warmup_discarded():
    call_count = [0]
    def fn():
        call_count[0] += 1
    measure_backend_latency(fn, TimingProtocol(warmup_runs=3, measured_runs=2))
    assert call_count[0] == 5  # 3 warmup + 2 measured


def test_measure_backend_latency_mean_aggregation():
    def fn():
        pass
    result = measure_backend_latency(fn, TimingProtocol(
        warmup_runs=0, measured_runs=5, aggregation="mean"))
    assert result.aggregation == "mean"


def test_to_dict_serializable():
    def fn():
        pass
    result = measure_backend_latency(fn, TimingProtocol(warmup=0, measured_runs=2) if False else TimingProtocol(warmup_runs=0, measured_runs=2))
    d = result.to_dict()
    assert "median_ms" in d
    assert "raw_ms" in d
    assert isinstance(d["raw_ms"], list)


def test_timing_protocol_is_frozen():
    p = TimingProtocol()
    with pytest.raises((AttributeError, TypeError)):
        p.warmup_runs = 99  # type: ignore[misc]
