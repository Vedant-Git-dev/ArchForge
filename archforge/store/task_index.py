"""Nearest-neighbour index over task embeddings.

Phase 1: a flat (brute-force) cosine kNN over numpy vectors. The store
holds task embeddings indexed by experience id. Phase 3 may swap this
for HNSW — Phase 1 keeps it simple.

The index loads from `embeddings.npy` + `experiences.jsonl` on init and
appends to both on every `add()`. Both files live under
`<data_dir>/experiences/` by default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TaskIndex:
    """Brute-force cosine kNN over stored task embeddings.

    Embeddings are assumed L2-normalised already (EmbeddingClient does this
    by default), so cosine similarity == dot product.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._dirty = False

    # ----- maintenance -----

    def add(self, id: str, embedding: np.ndarray) -> None:
        """Append one experience's embedding. `embedding` may be 1-D."""
        v = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dim {v.shape[0]} != index dim {self.dim}; "
                f"check you reconstructed the index with the right model."
            )
        self._ids.append(id)
        self._matrix = np.vstack([self._matrix, v])
        self._dirty = True

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> list[str]:
        return list(self._ids)

    # ----- search -----

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return the k nearest experience ids with cosine similarity (desc)."""
        if len(self._ids) == 0 or self._matrix.shape[0] == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Query dim {q.shape[0]} != index dim {self.dim}")
        # Cosine via dot product because vectors are L2-normalised.
        sims = self._matrix @ q  # shape (n,)
        k = max(1, min(k, len(self._ids)))
        # argpartition is O(n) instead of O(n log n); we re-sort the top slice.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_sorted = top_idx[np.argsort(-sims[top_idx])]
        return [(self._ids[i], float(sims[i])) for i in top_sorted]

    # ----- persistence -----

    def is_dirty(self) -> bool:
        return self._dirty

    def save(self, dirpath: Path) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        ids_path = dirpath / "index_ids.json"
        matrix_path = dirpath / "embeddings.npy"
        with ids_path.open("w") as f:
            import json

            json.dump(self._ids, f)
        np.save(matrix_path, self._matrix)
        self._dirty = False

    @classmethod
    def load(cls, dirpath: Path, dim: int) -> "TaskIndex":
        ids_path = dirpath / "index_ids.json"
        matrix_path = dirpath / "embeddings.npy"
        idx = cls(dim=dim)
        if ids_path.is_file() and matrix_path.is_file():
            import json

            with ids_path.open() as f:
                idx._ids = list(json.load(f))
            idx._matrix = np.load(matrix_path)
            if idx._matrix.dtype != np.float32:
                idx._matrix = idx._matrix.astype(np.float32)
            if idx._matrix.ndim == 1:
                # Empty arrays come back shape (0,) — reshape to (0, dim).
                idx._matrix = idx._matrix.reshape(-1, idx.dim)
        return idx


__all__ = ["TaskIndex"]
