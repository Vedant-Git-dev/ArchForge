"""Task dataclass — input to the ArchForge pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_task_id() -> str:
    return f"task-{uuid.uuid4().hex[:12]}"


@dataclass
class Task:
    """A unit of work handed to the Architect.

    `embedding` is a list[float] (384 dims by default — MiniLM-L6-v2).
    It is filled in by the EmbeddingClient, never by callers.
    """

    id: str
    description: str
    type: str  # free-form label: "analysis", "generation", "extraction", ...
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        description: str,
        type: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> "Task":
        return cls(
            id=new_task_id(),
            description=description,
            type=type,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "metadata": self.metadata,
            # embedding is persisted alongside the task in the task index,
            # not in the JSONL store — keeps the file readable.
            "has_embedding": bool(self.embedding),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            description=data["description"],
            type=data["type"],
            metadata=data.get("metadata", {}),
            embedding=[],
        )
