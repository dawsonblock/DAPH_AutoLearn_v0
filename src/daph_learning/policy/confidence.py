"""v0.3.10 — explicit confidence model with provenance (Section 8).

Confidence is NOT a vague scalar. It is decomposed into components with
explicit provenance so that the learning system knows *why* an outcome is
trusted and can weight examples accordingly.

Components
----------
verifier   : confidence in the verifier's judgment (1.0 for deterministic
             symbolic checking unless execution is ambiguous).
measurement: confidence in the measurement process (latency, cost).
stability  : confidence that the outcome is stable across re-runs.
ood        : confidence that the task is in-distribution (1 - OOD score).

The combined confidence is the product of all components, which is
conservative: any single weak component drags the overall confidence down.

v0.3.10.3.1 — Section 5: paired confidence combine for gap weighting.
``c_i = combine(c_S, c_L)`` with product (default), min, geometric_mean.
Do not let an unsupported-but-known action erase the training example:
confidence is kept separate from quality (a known-unsupported backend
has low quality but HIGH measurement confidence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ConfidenceCombine(str, Enum):
    """v0.3.10.3.1 — Section 5: how to combine the two backends'
    confidence into the paired example confidence ``c_i``.

    PRODUCT         : ``c_i = c_S * c_L`` (default; conservative — either
                      weak confidence drags the pair down).
    MIN             : ``c_i = min(c_S, c_L)``.
    GEOMETRIC_MEAN  : ``c_i = sqrt(c_S * c_L)``.
    """

    PRODUCT = "product"
    MIN = "min"
    GEOMETRIC_MEAN = "geometric_mean"

    @classmethod
    def from_str(cls, value: str) -> "ConfidenceCombine":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(
            f"unknown confidence_combine {value!r}; expected one of "
            f"{[m.value for m in cls]}")


def combine_paired_confidence(
    c_symbolic: float,
    c_llm: float,
    mode: "ConfidenceCombine | str" = ConfidenceCombine.PRODUCT,
) -> float:
    """Combine the two backends' confidence into the paired example
    confidence ``c_i`` (Section 5).

    Parameters
    ----------
    c_symbolic, c_llm : float
        Per-backend confidence in ``[0, 1]``.
    mode : ConfidenceCombine | str
        Combination rule. Default ``PRODUCT``.
    """
    if isinstance(mode, str):
        mode = ConfidenceCombine.from_str(mode)
    for name, v in (("c_symbolic", c_symbolic), ("c_llm", c_llm)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if mode == ConfidenceCombine.PRODUCT:
        return float(c_symbolic * c_llm)
    if mode == ConfidenceCombine.MIN:
        return float(min(c_symbolic, c_llm))
    if mode == ConfidenceCombine.GEOMETRIC_MEAN:
        return float(math.sqrt(c_symbolic * c_llm))
    raise ValueError(f"unknown confidence_combine mode: {mode}")


@dataclass(frozen=True)
class OutcomeConfidence:
    """Explicit, multi-component outcome confidence (Section 8).

    All components must be in ``[0, 1]``. The combined confidence is the
    product of all components.
    """

    verifier: float
    measurement: float = 1.0
    stability: float = 1.0
    ood: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("verifier", self.verifier),
            ("measurement", self.measurement),
            ("stability", self.stability),
            ("ood", self.ood),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence component {name!r} must be in [0, 1], got {value}")

    def combined(self) -> float:
        """Conservative combined confidence (product of all components)."""
        result = 1.0
        for value in (self.verifier, self.measurement, self.stability, self.ood):
            result *= value
        return float(result)

    def to_dict(self) -> dict[str, float]:
        return {
            "verifier": self.verifier,
            "measurement": self.measurement,
            "stability": self.stability,
            "ood": self.ood,
            "combined": self.combined(),
        }


__all__ = ["ConfidenceCombine", "OutcomeConfidence", "combine_paired_confidence"]
