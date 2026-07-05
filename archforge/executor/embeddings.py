"""Embedding client wrapping sentence-transformers MiniLM-L6-v2.

Phase 1: single client, single model, CPU-only. Phase 3 may split
task vs pipeline embedding models — out of scope here.

The model loads lazily on first use and stays in memory for the
process lifetime. This avoids re-paying the ~80 MB load cost on
every CLI invocation while keeping cold-start low.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_ENV


def _resolve_model_name() -> str:
    return os.environ.get(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


class EmbeddingClient:
    """Lazy-loading sentence-transformers wrapper.

    Embeddings are L2-normalised so cosine similarity == dot product.
    """

    def __init__(self, model_name: str | None = None, normalize: bool = True) -> None:
        self.model_name = model_name or _resolve_model_name()
        self.normalize = normalize
        self._model: Any | None = None  # set on first embed()
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  

        self._model = SentenceTransformer(self.model_name)
        self._dim = int(self._model.get_embedding_dimension())

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 matrix, normalised if configured."""
        self._ensure_loaded()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        return arr

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ─── Factory ────────────────────────────────────────────────────────────────


def get_default_embedding_client() -> EmbeddingClient:
    """Return the embedding client for the current environment.

    Production uses sentence-transformers (all-MiniLM-L6-v2 by default).
    There is no silent fallback — a missing `sentence-transformers` library
    surfaces as ImportError on first `.dim` / `.embed` call.
    """
    return EmbeddingClient()


__all__ = ["EmbeddingClient", "get_default_embedding_client"]
