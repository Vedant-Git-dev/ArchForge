"""Architect — Phase 1 implements retrieve-and-replay only.

The flow per plan:
    1. Embed the task.
    2. kNN search the experience index (task similarity).
    3. If a similar past experience exists with a usable pipeline,
       return its DAG as-is.
    4. Else, build the default linear pipeline from the 6 base primitives.

No mutations, no diagnosis, no interventions yet — that lands in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.experience import Experience
from ..core.pipeline import PipelineDAG
from ..core.task import Task
from ..store.experience_store import ExperienceStore, ScoredHit
from ..executor.embeddings import EmbeddingClient
from ..config import (
    DEFAULT_PIPELINE_AGENTS,
    DEFAULT_REPLAY_SIMILARITY_THRESHOLD,
)


@dataclass
class ArchitectureDecision:
    """What the Architect returns to the caller.

    Includes provenance (`triggered_from`, `matched_experience`) so the
    CLI and tests can assert that retrieval actually happened.
    """

    pipeline: PipelineDAG
    triggered_from: str  # "retrieval" | "default"
    matched_experience_id: str | None
    matched_pipeline_score: float | None  # composite score of the replayed pipeline
    candidates: list[ScoredHit] = field(default_factory=list)
    task_embedding_dim: int = 0


class Architect:
    """Decide which pipeline to run for a task.

    Phase 1: retrieve highest-composite-score experience above threshold.
    """

    def __init__(
        self,
        store: ExperienceStore,
        embedder: EmbeddingClient,
        *,
        replay_similarity_threshold: float = DEFAULT_REPLAY_SIMILARITY_THRESHOLD,
        top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.threshold = replay_similarity_threshold
        self.top_k = top_k

    # ----- entry point -----

    def compose(self, task: Task) -> ArchitectureDecision:
        """Return the PipelineDAG the executor should run for `task`."""

        # Embed the task. Description + type provides a stable signal even
        # for tasks with empty metadata.
        text = self._task_text(task)
        embedding = self.embedder.embed_one(text)
        task.embedding = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()

        # Make sure any prior experiences without embeddings are filled in
        # before we run the kNN search.
        self.store.recompute_embeddings(self.embedder)

        # Search by task similarity.
        hits: list[ScoredHit] = self.store.search_by_task(
            np.asarray(embedding, dtype=np.float32).reshape(-1), k=self.top_k
        )

        # Pick the best-scoring experience whose similarity clears the bar.
        replayable = [h for h in hits if h.score >= self.threshold and h.experience.pipeline.nodes]
        if replayable:
            best = max(replayable, key=lambda h: h.experience.composite_score)
            pipeline = PipelineDAG.from_dict(best.experience.pipeline.to_dict())
            pipeline.fingerprint = pipeline.compute_fingerprint()
            return ArchitectureDecision(
                pipeline=pipeline,
                triggered_from="retrieval",
                matched_experience_id=best.experience.id,
                matched_pipeline_score=best.experience.composite_score,
                candidates=hits,
                task_embedding_dim=task.embedding.__len__(),
            )

        # No usable hit — build the default pipeline.
        pipeline = PipelineDAG.linear(DEFAULT_PIPELINE_AGENTS)
        return ArchitectureDecision(
            pipeline=pipeline,
            triggered_from="default",
            matched_experience_id=None,
            matched_pipeline_score=None,
            candidates=hits,
            task_embedding_dim=task.embedding.__len__(),
        )

    # ----- helper -----

    @staticmethod
    def _task_text(task: Task) -> str:
        # Combine type with description so two tasks with the same wording
        # but different intents can hash to different embeddings.
        return f"{task.type}: {task.description}".strip()


# Import numpy lazily for the .tolist() projection — keeps the Architect
# import cheap for callers that only need defaults.

__all__ = ["Architect", "ArchitectureDecision", "DEFAULT_PIPELINE_AGENTS"]
