"""Embedding client wrapping sentence-transformers MiniLM-L6-v2.

Phase 1: single client, single model, CPU-only. Phase 3 may split
task vs pipeline embedding models — out of scope here.

The model loads lazily on first use and stays in memory for the
process lifetime. This avoids re-paying the ~80 MB load cost on
every CLI invocation while keeping cold-start low.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

# Defaults — keep tunable via env so swapping the model doesn't require
# a code change. MiniLM-L6-v2 produces 384-dim vectors.
DEFAULT_MODEL = os.environ.get(
    "ARCHFORGE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class EmbeddingClient:
    """Lazy-loading sentence-transformers wrapper.

    Embeddings are L2-normalised so cosine similarity == dot product.
    """

    def __init__(self, model_name: str | None = None, normalize: bool = True) -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.normalize = normalize
        self._model: Any | None = None  # set on first embed()
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

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


# ─── Cached fallback (deterministic, offline) ───────────────────────────────


class HashingEmbeddingClient:
    """Deterministic, dependency-free embedding.

    Used when sentence-transformers isn't installed or for offline testing.
    Produces 256-dim vectors via token hashing — *not* semantically meaningful,
    but stable and zero-dep, so callers can still exercise the full pipeline.

    Pretty much only useful to prove the wiring works without a real model.
    Phase 1 keeps it as a fallback; production paths should use EmbeddingClient.
    """

    DIM = 256

    def __init__(self, normalize: bool = True) -> None:
        self.model_name = "hashing-fallback"
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self.DIM

    def _hash_token(self, token: str, seed: int) -> np.ndarray:
        h = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
        # 32 bytes -> 256 bits, mapped to +/-1 floats
        bits = np.frombuffer(h, dtype=np.uint8)
        # Expand to 256 dims: replicate the 32 bytes into 8 lanes
        out = np.zeros(self.DIM, dtype=np.float32)
        for i in range(8):
            out[i * 32 : (i + 1) * 32] = (bits.astype(np.float32) - 127.5) / 127.5
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for token in t.lower().split():
                out[i] += self._hash_token(token, 0)
                out[i] += self._hash_token(token, 1)
            if self.normalize:
                n = np.linalg.norm(out[i])
                if n > 0:
                    out[i] /= n
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ─── Factory ────────────────────────────────────────────────────────────────


def get_default_embedding_client(
    force_name: str | None = None,
) -> EmbeddingClient | HashingEmbeddingClient:
    """Return an embedding client for the current environment.

    Prefers sentence-transformers (all-MiniLM-L6-v2 by default).
    Falls back to HashingEmbeddingClient if the library isn't available
    — useful for tests that don't want the ~80 MB model download.

    Tests can also pass `force_name="hashing"` to skip the heavy model
    entirely.
    """
    name = force_name or os.environ.get("ARCHFORGE_EMBEDDING_MODEL", DEFAULT_MODEL)
    if name == "hashing":
        return HashingEmbeddingClient()
    try:
        return EmbeddingClient(model_name=name)
    except Exception:
        return HashingEmbeddingClient()
