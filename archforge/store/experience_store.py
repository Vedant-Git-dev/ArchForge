"""Experience store — JSONL file of past runs + task-index sidecar.

Phase 1 reads everything into memory on init and writes both files on
every append. That keeps the implementation small but doesn't scale to
millions of experiences — Phase 6 can swap in sqlite/sql.py if needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.experience import Experience
from ..logging import get_logger
from .task_index import TaskIndex

log = get_logger("store")


@dataclass
class ScoredHit:
    """One element of a retrieval result: an Experience + cosine similarity."""

    experience: Experience
    score: float  # cosine similarity in [-1, 1]; Phase 1 expects positive
    rank: int


class ExperienceStore:
    """Persistent log of past runs, queryable via the task index."""

    def __init__(
        self,
        dirpath: Path,
        dim: int,
    ) -> None:
        """`dirpath` is the directory containing experiences.jsonl + index files."""
        self.dirpath = Path(dirpath)
        self.dim = dim
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.path = self.dirpath / "experiences.jsonl"
        self._experiences: list[Experience] = []
        self._id_to_idx: dict[str, int] = {}
        self.index = TaskIndex.load(self.dirpath, dim=dim)
        self._load_file()
        log.debug("ExperienceStore init dir=%s dim=%d loaded=%d", self.dirpath, dim, len(self))

    # ----- persistence -----

    def _load_file(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                exp = Experience.from_dict(data)
                # Defensive: ignore duplicates by id.
                if exp.id in self._id_to_idx:
                    continue
                self._id_to_idx[exp.id] = len(self._experiences)
                self._experiences.append(exp)
                # Rebuild index from stored embeddings if we have them.
                # The plan calls for embedding persistence, but to keep
                # Phase 1 simple, embeddings are recomputed lazily
                # from `task.description + task.type` if missing.
                # (See `recompute_embeddings` below.)

    def _append_file(self, exp: Experience) -> None:
        with self.path.open("a") as f:
            f.write(exp.to_jsonl())
            f.write("\n")

    # ----- mutation -----

    def append(self, exp: Experience) -> None:
        is_rewrite = exp.id in self._id_to_idx
        if is_rewrite:
            # Idempotent re-write of same experience: replace in-place.
            idx = self._id_to_idx[exp.id]
            self._experiences[idx] = exp
        else:
            self._id_to_idx[exp.id] = len(self._experiences)
            self._experiences.append(exp)
        self._append_file(exp)
        log.info(
            "append: experience id=%s composite=%.3f tokens=%d (rewrite=%s total=%d)",
            exp.id, exp.composite_score, exp.token_estimate, is_rewrite, len(self),
        )

    def recompute_embeddings(self, embedder) -> int:
        """Fill any missing task embeddings from descriptions.

        Returns the number of new embeddings computed. Caller passes an
        EmbeddingClient.
        """
        missing: list[tuple[int, str]] = []
        for i, exp in enumerate(self._experiences):
            if not exp.task.embedding:
                # task_text is the input the embedding model sees.
                # Including `type` so structurally-similar tasks from
                # different domains still match.
                text = f"{exp.task.type}: {exp.task.description}"
                missing.append((i, text))

        if not missing:
            return 0

        # Add to index in original append order so ids correlate with file order.
        for i, text in missing:
            vec = embedder.embed_one(text).reshape(-1)
            self._experiences[i].task.embedding = vec.tolist()
            self.index.add(self._experiences[i].id, vec)
        return len(missing)

    # ----- inspection -----

    def __len__(self) -> int:
        return len(self._experiences)

    def __iter__(self) -> Iterator[Experience]:
        yield from self._experiences

    def get(self, exp_id: str) -> Experience | None:
        idx = self._id_to_idx.get(exp_id)
        if idx is not None:
            return self._experiences[idx]
        return None

    def all(self) -> list[Experience]:
        return list(self._experiences)

    # ----- retrieval -----

    def search_by_task(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        *,
        min_score: float = -1.0,
    ) -> list[ScoredHit]:
        """Return up to `k` past experiences most similar to the query.

        Experiences in the index but missing task embeddings return `score=0`
        and are pushed to the end of the list — they have rank but no signal.
        """
        raw = self.index.search(query_embedding, k=k)
        out: list[ScoredHit] = []
        for rank, (eid, score) in enumerate(raw):
            if score < min_score:
                continue
            exp = self.get(eid)
            if exp is None:
                continue
            out.append(ScoredHit(experience=exp, score=score, rank=rank))
        return out

    # ----- flush -----

    def save_index(self) -> None:
        self.index.save(self.dirpath)
        log.info("save_index: persisted task index (entries=%d dir=%s)", len(self), self.dirpath)


__all__ = ["ExperienceStore", "ScoredHit"]
