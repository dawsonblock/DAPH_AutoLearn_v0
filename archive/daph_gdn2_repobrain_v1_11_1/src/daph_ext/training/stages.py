from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TrainingStage(IntEnum):
    GDN2_INTEGRATION = 1
    GDN2_NUMERICAL_QUALIFICATION = 2
    REPO2LORA_STATIC = 3
    FUNCTIONAL_CONTROLLER = 4
    REPO2LORA_EVOLUTION = 5
    STRUCTURAL_VISION = 6
    KERNEL_OPTIMIZATION = 7


@dataclass(frozen=True)
class StageGate:
    stage: TrainingStage
    required_artifacts: tuple[str, ...]
    pass_conditions: tuple[str, ...]


STAGE_GATES: tuple[StageGate, ...] = (
    StageGate(
        TrainingStage.GDN2_INTEGRATION,
        ("fixed layer plan", "forward/backward smoke test"),
        ("no sparse token compaction", "finite loss", "shape contracts pass"),
    ),
    StageGate(
        TrainingStage.GDN2_NUMERICAL_QUALIFICATION,
        ("reference outputs", "chunk outputs", "recurrent outputs"),
        ("relative L2 within dtype threshold", "final-state drift within threshold"),
    ),
    StageGate(
        TrainingStage.REPO2LORA_STATIC,
        ("static benchmark", "shared-LoRA baseline", "retrieval baseline"),
        ("held-out repository gain", "no base capability collapse"),
    ),
    StageGate(
        TrainingStage.FUNCTIONAL_CONTROLLER,
        ("geometry metrics", "loss/KL probes", "execution probes"),
        ("controller never corrects from geometry alone", "rejected adapters are truly harmful"),
    ),
    StageGate(
        TrainingStage.REPO2LORA_EVOLUTION,
        ("chronological split", "commit diff embeddings", "state replay"),
        ("beats static adapter after drift", "no future leakage"),
    ),
    StageGate(
        TrainingStage.STRUCTURAL_VISION,
        ("vision interface", "aligned token bridge"),
        ("measurable structural-task gain", "semantic task non-regression"),
    ),
    StageGate(
        TrainingStage.KERNEL_OPTIMIZATION,
        ("profiler trace",),
        ("triangular solve is material runtime bottleneck",),
    ),
)
