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
"""

from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["OutcomeConfidence"]
