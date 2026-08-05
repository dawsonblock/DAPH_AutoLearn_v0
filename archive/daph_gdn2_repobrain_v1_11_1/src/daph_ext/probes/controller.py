from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from daph_ext.config import ControllerConfig
from daph_ext.geometry.metrics import GeometryMetrics
from daph_ext.probes.base import ProbeResult


class AdapterAction(str, Enum):
    ACCEPT = "accept"
    SCALE = "scale"
    GATE = "gate"
    PROJECT = "project"
    REJECT = "reject"
    EVALUATE_CORRECTIONS = "evaluate_corrections"


@dataclass(frozen=True)
class AdapterDecision:
    action: AdapterAction
    scale: float = 1.0
    reason: str = ""
    requires_revalidation: bool = False


class AdapterController:
    """Evidence-first controller. Geometry is advisory, never dispositive."""

    _EXECUTION_KEYS = ("execution", "exec", "pass_rate", "passrate", "unit_test")
    _KL_KEYS = ("kl", "drift", "divergence")
    _GENERIC_KEYS = ("generic", "capability", "retention")
    _REPO_KEYS = ("repo", "repository", "task")
    _LOSS_KEYS = ("loss", "perplexity", "ppl", "cross_entropy", "val_ce", "ce")

    def __init__(self, cfg: ControllerConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _matches(name: str, keys: tuple[str, ...]) -> bool:
        """Match aliases without letting short tokens collide accidentally.

        The old substring matcher classified ``kl_divergence`` as a loss because
        the short alias ``ce`` occurs at the end of ``divergence``. Short aliases
        now require token equality; longer phrases may match as substrings.
        """
        normalized = name.lower().replace("-", "_")
        tokens = set(normalized.split("_"))
        for key in keys:
            key = key.lower().replace("-", "_")
            if normalized == key or key in tokens:
                return True
            if len(key) >= 4 and key in normalized:
                return True
        return False

    def decide(
        self,
        probes: Iterable[ProbeResult],
        geometry: Iterable[GeometryMetrics] = (),
    ) -> AdapterDecision:
        probes = list(probes)
        _ = list(geometry)
        if not probes:
            return AdapterDecision(AdapterAction.REJECT, reason="no functional evidence")

        execution = [p for p in probes if self._matches(p.name, self._EXECUTION_KEYS)]
        generic = [p for p in probes if self._matches(p.name, self._GENERIC_KEYS)]
        repo = [p for p in probes if self._matches(p.name, self._REPO_KEYS)]
        losses = [p for p in probes if self._matches(p.name, self._LOSS_KEYS)]
        kl = [p for p in probes if self._matches(p.name, self._KL_KEYS)]

        if any(p.regression > self.cfg.max_execution_regression for p in execution):
            return AdapterDecision(AdapterAction.REJECT, reason="execution regression above gate")

        if any(p.regression > self.cfg.max_generic_regression for p in generic):
            return AdapterDecision(
                AdapterAction.EVALUATE_CORRECTIONS,
                reason="generic capability regression requires measured correction sweep",
                requires_revalidation=True,
            )

        if any(p.regression > self.cfg.max_validation_loss_regression for p in losses):
            return AdapterDecision(
                AdapterAction.EVALUATE_CORRECTIONS,
                reason="validation regression requires measured correction sweep",
                requires_revalidation=True,
            )

        if self.cfg.max_kl_drift is not None and any(
            p.regression > self.cfg.max_kl_drift for p in kl
        ):
            return AdapterDecision(
                AdapterAction.EVALUATE_CORRECTIONS,
                reason="calibrated KL envelope exceeded",
                requires_revalidation=True,
            )

        # Repo-specific evidence must demonstrate a strict benefit, not merely a tie.
        if repo and not any(p.regression < 0 for p in repo):
            return AdapterDecision(AdapterAction.REJECT, reason="no measured repository-task benefit")

        return AdapterDecision(AdapterAction.ACCEPT, reason="functional probes passed")
