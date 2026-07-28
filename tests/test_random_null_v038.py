from __future__ import annotations

import pytest

from daph_learning.evaluation.nulls import empirical_null_test


def test_empirical_null_uses_finite_sample_correction() -> None:
    result = empirical_null_test(1.0, [0.0] * 5)
    assert result.exceedances == 0
    assert result.empirical_p_value == pytest.approx(1 / 6)
    assert result.empirical_p_value != 0


def test_empirical_null_protocol_threshold() -> None:
    small = empirical_null_test(1.0, [0.0] * 499)
    enough = empirical_null_test(1.0, [0.0] * 500)
    assert not small.protocol_eligible
    assert enough.protocol_eligible
    assert enough.observed_percentile == 100.0


def test_empirical_null_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        empirical_null_test(1.0, [0.0, float("nan")])
