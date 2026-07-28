"""v0.3.10 — prioritized experience replay (Section 22).

Upgrade experience memory from passive provenance to replay-ready storage.
Each experience contains task_id, activation_hash, feature representation,
backend utilities, ΔU, chosen route, router probabilities, prediction
error, regret, confidence, task family, policy version, and timestamp.

Replay priority emphasizes surprising, costly errors::

    priority_i = |predicted_preference_i - realized_preference_i| × |ΔU_i|

or alternatively ``priority_i = regret_i``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


def replay_priority(
    predicted_symbolic_prob: float,
    realized_symbolic_target: float,
    delta_u: float,
) -> float:
    """Replay priority: ``|pred - realized| × |ΔU|`` (Section 22).

    Emphasizes surprising, costly errors. Do not over-sample already-
    solved trivial examples.
    """
    prediction_error = abs(predicted_symbolic_prob - realized_symbolic_target)
    return prediction_error * abs(delta_u)


@dataclass
class ReplayExperience:
    """A replay-ready experience record (Section 22)."""

    task_id: str
    activation_hash: str = ""
    feature_representation: np.ndarray | None = None
    u_symbolic: float = 0.0
    u_llm: float = 0.0
    delta_utility: float = 0.0
    chosen_route: str = "abstain"
    router_probabilities: dict[str, float] = field(default_factory=dict)
    prediction_error: float = 0.0
    regret: float = 0.0
    confidence: float = 1.0
    task_family: str = ""
    policy_version: str = ""
    timestamp: float = 0.0
    priority: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.feature_representation is not None:
            d["feature_representation"] = np.asarray(self.feature_representation).tolist()
        return d


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer (Section 22).

    Samples experiences proportional to their priority, emphasizing
    surprising, costly errors.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.buffer: list[ReplayExperience] = []
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, experience: ReplayExperience) -> None:
        """Add an experience, evicting the lowest-priority item if full."""
        if len(self.buffer) >= self.capacity:
            # Evict the lowest-priority experience.
            min_idx = int(np.argmin([e.priority for e in self.buffer]))
            self.buffer[min_idx] = experience
        else:
            self.buffer.append(experience)

    def sample(self, batch_size: int, beta: float = 1.0) -> list[ReplayExperience]:
        """Sample a batch proportional to priority.

        Parameters
        ----------
        batch_size : int
        beta : float
            Importance-sampling exponent (1.0 = full correction).
        """
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        priorities = np.array([max(e.priority, 1e-8) for e in self.buffer])
        probs = priorities ** beta
        probs = probs / probs.sum()
        idx = self._rng.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
        return [self.buffer[i] for i in idx]

    def mean_priority(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.mean([e.priority for e in self.buffer]))


__all__ = [
    "PrioritizedReplayBuffer",
    "ReplayExperience",
    "replay_priority",
]
