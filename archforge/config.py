"""Centralized configuration for ArchForge.

Every tunable constant, env-var name, model id, and the per-component LLM
routing table lives here. The rest of the package imports from this module;
a few modules re-export values they own for backward compatibility (so the
public import paths used by tests do not change).

There is exactly one LLM provider (Groq, via the `groq` SDK). Each pipeline
component + the judge routes to a specific model id keyed by its own name.
Per-component overrides are read from `ARCHFORGE_LLM_<COMPONENT>` env vars.
"""

from __future__ import annotations

import os

# ─── Data directory ─────────────────────────────────────────────────────────

DATA_DIR_ENV = "ARCHFORGE_DATA_DIR"
DEFAULT_DATA_DIR = "data"


def data_dir() -> str:
    """Resolved data directory: env override or the default."""
    return os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR


# ─── Embeddings ────────────────────────────────────────────────────────────

EMBEDDING_MODEL_ENV = "ARCHFORGE_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Groq API ───────────────────────────────────────────────────────────────

GROQ_API_KEY_ENV = "GROQ_API_KEY"

# Per-component model routing. Keys are the actual component names that
# primitives/knobs identify themselves with; values are literal model-id
# strings on the Groq API. `default` is the fallback when a caller
# omits `kind`. Override any single component via
# `ARCHFORGE_LLM_<COMPONENT.toUpperCase>`.
#
# Light ingest/transform stages (reader, chunker) run on the fast, cheap
# llama-3.1-8b-instant; the analyser/validator/generator stages and the
# judge run on the stronger llama-3.3-70b-versatile. This mirrors the
# small/large split the Gemini config used, so quality-bearing components
# keep the heavier model. Every entry is overridable per-component via env.
DEFAULT_LLM_ROUTES: dict[str, str] = {
    "reader": "llama-3.1-8b-instant",
    "chunker": "llama-3.1-8b-instant",
    "classifier": "llama-3.3-70b-versatile",
    "summarizer": "llama-3.3-70b-versatile",
    "fact_checker": "llama-3.3-70b-versatile",
    "writer": "llama-3.3-70b-versatile",
    "judge": "llama-3.3-70b-versatile",
    "default": "llama-3.3-70b-versatile",
}


def load_llm_routes() -> dict[str, str]:
    """Return the component→model-id map with env overrides applied.

    Each component may be overridden by `ARCHFORGE_LLM_<COMPONENT>`. Only
    components present in `DEFAULT_LLM_ROUTES` are overridable.
    """
    routes = dict(DEFAULT_LLM_ROUTES)
    for component in DEFAULT_LLM_ROUTES:
        override = os.environ.get(f"ARCHFORGE_LLM_{component.upper()}")
        if override:
            routes[component] = override
    return routes


# ─── Architect ──────────────────────────────────────────────────────────────

# Default linear pipeline used when retrieval falls through.
DEFAULT_PIPELINE_AGENTS: list[str] = [
    "reader",
    "chunker",
    "classifier",
    "summarizer",
    "fact_checker",
    "writer",
]

# How similar a past task must be (cosine) before its pipeline is replayed.
DEFAULT_REPLAY_SIMILARITY_THRESHOLD = 0.5


# ─── Evaluator scoring floors ─────────────────────────────────────────────

# Speed: wall-clock seconds. at or below SLA → full credit; at or above
# penalty floor → zero credit; linear between.
SPEED_SLA_SECONDS = 5.0
SPEED_PENALTY_FLOOR = 60.0

# Cost: total tokens. at or below budget → full credit; at or above penalty
# floor → zero credit; linear between.
COST_BUDGET_TOKENS = 500
COST_PENALTY_FLOOR = 8000


# ─── Structural evaluation (Phase 2) ──────────────────────────────────────────

# Baseline structural-score penalties, per plan.md "Structural Evaluation".
# A 1.0 baseline is charged against the two unambiguously-bad defects; a
# learned ideal structural profile per task type (deviation from ideal) lands
# in Phase 6 weight learning. For now the score is the plan's hard-constraint
# penalties alone.
STRUCTURAL_UNUSED_PENALTY = 0.10   # per dead-output leaf (output no one reads)
STRUCTURAL_REDUNDANT_PENALTY = 0.15  # per structurally-duplicate agent


__all__ = [
    "DATA_DIR_ENV",
    "DEFAULT_DATA_DIR",
    "data_dir",
    "EMBEDDING_MODEL_ENV",
    "DEFAULT_EMBEDDING_MODEL",
    "GROQ_API_KEY_ENV",
    "DEFAULT_LLM_ROUTES",
    "load_llm_routes",
    "DEFAULT_PIPELINE_AGENTS",
    "DEFAULT_REPLAY_SIMILARITY_THRESHOLD",
    "SPEED_SLA_SECONDS",
    "SPEED_PENALTY_FLOOR",
    "COST_BUDGET_TOKENS",
    "COST_PENALTY_FLOOR",
    "STRUCTURAL_UNUSED_PENALTY",
    "STRUCTURAL_REDUNDANT_PENALTY",
]
