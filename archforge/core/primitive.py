"""Primitive definition — the unit of composition for the Architect.

A Primitive is *what an agent is* (system prompt, schemas, optional provenance).
A pipeline Node is *how it is used* (id, position in DAG). Pool-level fields
like `created_from_n_experiences` only matter for evolved primitives (Phase 5+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Primitive:
    name: str
    level: int = 0  # 0 = base (human-defined), 1+ = evolved
    role: str = "analyze"  # "ingest" | "transform" | "analyze" | "validate" | "generate" | "compose"
    system_prompt: str = ""
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)

    # What kind of agent backs this primitive, and per-kind parameters.
    # "llm"          → reads system_prompt + params (max_tokens/temperature); ctx.llm required.
    # "deterministic" → a plain Python callable; system_prompt inert; ctx.llm ignored.
    # (future: "tool"/"http"/"subpipeline") — previewed by the Agent contract, not wired in v1.
    kind: str = "llm"
    params: dict = field(default_factory=dict)

    # Only meaningful for evolved primitives (Phase 5+).
    source_subgraph: list[str] | None = None
    fusing_prompt: str | None = None
    validation_score: float | None = None
    created_from_n_experiences: int | None = None
    created_at: datetime | None = None
    can_unwrap: bool = True

    # ----- serialization helpers (YAML on disk, dict in memory) -----

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "kind": self.kind,
            "params": self.params,
            "source_subgraph": self.source_subgraph,
            "fusing_prompt": self.fusing_prompt,
            "validation_score": self.validation_score,
            "created_from_n_experiences": self.created_from_n_experiences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "can_unwrap": self.can_unwrap,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Primitive":
        ts = data.get("created_at")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            name=data["name"],
            level=data.get("level", 0),
            role=data.get("role", "analyze"),
            system_prompt=data.get("system_prompt", ""),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            kind=data.get("kind", "llm"),
            params=data.get("params", {}),
            source_subgraph=data.get("source_subgraph"),
            fusing_prompt=data.get("fusing_prompt"),
            validation_score=data.get("validation_score"),
            created_from_n_experiences=data.get("created_from_n_experiences"),
            created_at=ts,
            can_unwrap=data.get("can_unwrap", True),
        )
