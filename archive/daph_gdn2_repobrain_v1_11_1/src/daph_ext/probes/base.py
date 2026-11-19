from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ProbeResult:
    name: str
    baseline: float
    candidate: float
    higher_is_better: bool
    metadata: dict[str, float | str] = field(default_factory=dict)

    @property
    def regression(self) -> float:
        if self.higher_is_better:
            return self.baseline - self.candidate
        return self.candidate - self.baseline


class FunctionalProbe(Protocol):
    def run(self) -> ProbeResult: ...
