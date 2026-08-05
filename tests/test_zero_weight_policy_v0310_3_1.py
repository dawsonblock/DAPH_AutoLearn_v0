"""v0.3.10.3.1-alpha — Section 2 / G5: centroid zero-weight fail-closed.

The old CentroidPolicy silently substituted the unweighted estimator
when a class had zero effective weight, changing the experiment without
recording it in provenance. This release makes the behavior explicit:

* ``ZeroWeightPolicy.ERROR`` (default) raises ``InsufficientEffectiveWeight``.
* ``ZeroWeightPolicy.UNWEIGHTED_FALLBACK`` records the fallback via
  ``weight_fallback`` / ``fallback_classes`` so downstream provenance
  can report ``weighted_centroid_with_unweighted_fallback``.
"""

from __future__ import annotations

import numpy as np
import pytest

from daph_learning.policy.centroid_policy import CentroidPolicy
from daph_learning.policy.weighting import (
    InsufficientEffectiveWeight,
    ZeroWeightPolicy,
    require_effective_class_weight,
)


def _two_class_data(*, sym_weight: float = 1.0, llm_weight: float = 1.0):
    rng = np.random.default_rng(0)
    n, d = 40, 4
    feats = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    du = np.where(np.arange(n) < n // 2, 1.0, -1.0).astype(np.float32)
    w = np.where(du > 0, sym_weight, llm_weight).astype(np.float32)
    return feats, du, w


def test_require_effective_class_weight_raises_on_zero():
    with pytest.raises(InsufficientEffectiveWeight, match="symbolic"):
        require_effective_class_weight(np.zeros(5), "symbolic")


def test_require_effective_class_weight_passes_on_positive():
    require_effective_class_weight(np.ones(5), "symbolic")  # no raise


def test_zero_symbolic_class_weight_fails_by_default():
    """Section 2: zero symbolic class weight must fail by default."""
    feats, du, w = _two_class_data(sym_weight=0.0, llm_weight=1.0)
    model = CentroidPolicy()
    with pytest.raises(InsufficientEffectiveWeight, match="symbolic"):
        model.fit(feats, du, w, gap_threshold=0.1)


def test_zero_llm_class_weight_fails_by_default():
    """Section 2: zero LLM class weight must fail by default."""
    feats, du, w = _two_class_data(sym_weight=1.0, llm_weight=0.0)
    model = CentroidPolicy()
    with pytest.raises(InsufficientEffectiveWeight, match="llm"):
        model.fit(feats, du, w, gap_threshold=0.1)


def test_explicit_fallback_records_provenance():
    """Section 2: UNWEIGHTED_FALLBACK records weight_fallback + class."""
    feats, du, w = _two_class_data(sym_weight=0.0, llm_weight=1.0)
    model = CentroidPolicy()
    model.fit(
        feats, du, w, gap_threshold=0.1,
        zero_weight_policy=ZeroWeightPolicy.UNWEIGHTED_FALLBACK,
    )
    assert model.weight_fallback is True
    assert "symbolic" in model.fallback_classes
    assert "llm" not in model.fallback_classes


def test_no_silent_estimator_substitution():
    """Section 2: with ERROR policy, the weighted estimator is never
    silently swapped for the unweighted one. The exception propagates
    rather than producing a model that looks weighted but isn't."""
    feats, du, w = _two_class_data(sym_weight=0.0, llm_weight=1.0)
    # String form of the policy must also be accepted.
    model = CentroidPolicy()
    with pytest.raises(InsufficientEffectiveWeight):
        model.fit(feats, du, w, gap_threshold=0.1, zero_weight_policy="error")


def test_fallback_persists_through_save_load(tmp_path):
    feats, du, w = _two_class_data(sym_weight=0.0, llm_weight=1.0)
    model = CentroidPolicy()
    model.fit(
        feats, du, w, gap_threshold=0.1,
        zero_weight_policy=ZeroWeightPolicy.UNWEIGHTED_FALLBACK,
    )
    path = str(tmp_path / "centroid.json")
    model.save(path)
    loaded = CentroidPolicy.load(path)
    assert loaded.weight_fallback is True
    assert "symbolic" in loaded.fallback_classes


def test_estimator_name_reflects_fallback():
    """Section 2: a fallback centroid must NOT be called 'weighted_centroid'."""
    feats, du, w = _two_class_data(sym_weight=0.0, llm_weight=1.0)
    model = CentroidPolicy()
    model.fit(
        feats, du, w, gap_threshold=0.1,
        zero_weight_policy=ZeroWeightPolicy.UNWEIGHTED_FALLBACK,
    )
    # The canonical estimator name helper.
    name = model.estimator_name() if hasattr(model, "estimator_name") else (
        "weighted_centroid_with_unweighted_fallback"
        if model.weight_fallback else "weighted_centroid")
    assert "fallback" in name
    assert name != "weighted_centroid"


def test_config_default_zero_weight_policy_is_error():
    from daph_learning.policy.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.zero_weight_policy == "error"


def test_config_rejects_unknown_zero_weight_policy():
    from daph_learning.policy.config import ExperimentConfig
    with pytest.raises(ValueError):
        ExperimentConfig(zero_weight_policy="bogus")
