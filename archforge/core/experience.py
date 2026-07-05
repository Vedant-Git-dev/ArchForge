"""Experience record — one per pipeline run, persisted to JSONL."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .pipeline import PipelineDAG
from .task import Task


def new_experience_id() -> str:
    return f"exp-{uuid.uuid4().hex[:12]}"


@dataclass
class Diagnosis:
    """A structured explanation for why one axis is low. Phase 2 fills this in.

    Phase 1 keeps the schema in place but stores empty lists; nothing depends
    on it being populated yet.
    """

    axis: str  # "accuracy" | "speed" | "cost" | "structure"
    severity: float  # 0-1
    reason: str
    structural_root: str = ""  # categorial: "no_validator", "serial_bottleneck", ...

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "severity": self.severity,
            "reason": self.reason,
            "structural_root": self.structural_root,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Diagnosis":
        return cls(
            axis=data["axis"],
            severity=data["severity"],
            reason=data["reason"],
            structural_root=data.get("structural_root", ""),
        )


@dataclass
class OutputScores:
    """The five output-quality surfaces from plan.md Evaluator Surface 1.

    Phase 1 only fills accuracy + speed_normalized + cost_normalized.
    """

    accuracy: float = 0.0
    completeness: float = 0.0
    speed_normalized: float = 0.0
    cost_normalized: float = 0.0
    user_rating: float | None = None

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "speed_normalized": self.speed_normalized,
            "cost_normalized": self.cost_normalized,
            "user_rating": self.user_rating,
        }


@dataclass
class StructuralScores:
    """Phase 2 fills these. Phase 1 keeps the schema with zero defaults."""

    pipeline_length: int = 0
    critical_path_length: int = 0
    parallelism_ratio: float = 0.0
    redundant_agents: list[str] = field(default_factory=list)
    unused_outputs: list[str] = field(default_factory=list)
    dependency_depth: int = 0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pipeline_length": self.pipeline_length,
            "critical_path_length": self.critical_path_length,
            "parallelism_ratio": self.parallelism_ratio,
            "redundant_agents": list(self.redundant_agents),
            "unused_outputs": list(self.unused_outputs),
            "dependency_depth": self.dependency_depth,
            "score": self.score,
        }


@dataclass
class Experience:
    """One pipeline run against one task. Persisted as JSONL row."""

    id: str
    task: Task
    pipeline: PipelineDAG

    output: OutputScores = field(default_factory=OutputScores)
    structural: StructuralScores = field(default_factory=StructuralScores)

    composite_score: float = 0.0
    diagnoses: list[Diagnosis] = field(default_factory=list)

    interventions_applied: list[str] = field(default_factory=list)
    interventions_helped: dict[str, bool] = field(default_factory=dict)

    # Pipeline content hash for dedup / subgraph mining.
    # Mirrors plan.md's pipeline_hash field.
    pipeline_hash: str = ""

    # Cost / time tracking — derived but persisted for offline analysis.
    wall_time_seconds: float = 0.0
    token_estimate: int = 0

    # Final output of the pipeline (writer node) — saved for inspection.
    final_output: str = ""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generation: int = 0  # depth from the original pipeline (Phase 5+)

    def __post_init__(self) -> None:
        if not self.pipeline_hash:
            self.pipeline_hash = self.pipeline.content_hash()

    # ----- composite score -----

    # Per-spec defaults: Phase 6 may make these learnable per task-type.
    @staticmethod
    def default_weights() -> dict[str, float]:
        return {
            "accuracy": 0.5,
            "speed": 0.25,
            "cost": 0.25,
        }

    def compute_composite(self, weights: dict[str, float] | None = None) -> float:
        """Phase 1 weights: accuracy 0.5, speed 0.25, cost 0.25.

        Slow / expensive runs both reduce composite proportionally. If the
        pipeline emits nothing (accuracy=0), the score collapses.
        """
        w = weights if weights is not None else self.default_weights()
        return float(
            w["accuracy"] * self.output.accuracy
            + w["speed"] * self.output.speed_normalized
            + w["cost"] * self.output.cost_normalized
        )

    # ----- serialization -----

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "pipeline_hash": self.pipeline_hash,
            "output": self.output.to_dict(),
            "structural": self.structural.to_dict(),
            "composite_score": self.composite_score,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "interventions_applied": list(self.interventions_applied),
            "interventions_helped": dict(self.interventions_helped),
            "wall_time_seconds": self.wall_time_seconds,
            "token_estimate": self.token_estimate,
            "final_output": self.final_output,
            "timestamp": self.timestamp.isoformat(),
            "generation": self.generation,
            "task_type": self.task.type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experience":
        task = Task.from_dict(data["task"])
        pipeline = PipelineDAG.from_dict(data["pipeline"])
        output = OutputScores(**data.get("output", {}))
        structural = StructuralScores(**data.get("structural", {}))
        ts_raw = data.get("timestamp")
        ts = (
            datetime.fromisoformat(ts_raw)
            if isinstance(ts_raw, str)
            else datetime.now(timezone.utc)
        )
        return cls(
            id=data["id"],
            task=task,
            pipeline=pipeline,
            pipeline_hash=data.get("pipeline_hash", ""),
            output=output,
            structural=structural,
            composite_score=data.get("composite_score", 0.0),
            diagnoses=[Diagnosis.from_dict(d) for d in data.get("diagnoses", [])],
            interventions_applied=data.get("interventions_applied", []),
            interventions_helped=data.get("interventions_helped", {}),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
            token_estimate=data.get("token_estimate", 0),
            final_output=data.get("final_output", ""),
            timestamp=ts,
            generation=data.get("generation", 0),
        )

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
