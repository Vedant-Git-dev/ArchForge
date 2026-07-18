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
    "classifier": "qwen/qwen3.6-27b",
    "summarizer": "qwen/qwen3.6-27b",
    "fact_checker": "openai/gpt-oss-120b",
    "writer": "openai/gpt-oss-20b",
    "judge": "openai/gpt-oss-120b",
    "default": "openai/gpt-oss-120b",
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


# ─── Diagnosis (Phase 2 — Surface 3) ─────────────────────────────────────────

# A diagnosis is emitted only when a metric trips its "low" floor. Severity
# scales with how far below the floor the value sits (see Diagnostician).
# These are the rule-based trip points for the structural-root categoriser —
# the keys the intervention library (next deliverable) matches a fix against.
DIAGNOSIS_ACCURACY_LOW = 0.6    # accuracy below this is "inaccurate"
DIAGNOSIS_SPEED_LOW = 0.4       # speed_normalized below this is "slow"
DIAGNOSIS_COST_LOW = 0.4        # cost_normalized below this is "expensive"
DIAGNOSIS_PARALLELISM_LOW = 0.05  # parallelism_ratio below this is "serial"
DIAGNOSIS_BOTTLENECK_MIN_PATH = 3  # critical_path (edges) at/above this is a bottleneck
DIAGNOSIS_DEEP_CHAIN_MIN = 7   # dependency_depth at/above this is a fragile deep chain

# The controlled vocabulary of structural_root categories. The LLM
# diagnostician is constrained to these (plus the "unknown:<brief>" escape
# for novel roots), and the intervention library — next Phase 2 deliverable —
# matches a fix by this key. Seeded from plan.md's intervention table; the
# plan notes the set is "not limited to these", hence the unknown escape.
STRUCTURAL_ROOTS: tuple[str, ...] = (
    "no_validator",        # no validation/verify step present
    "serial_bottleneck",  # long serial critical path, low parallelism
    "redundant_agents",   # two+ agents duplicate each other's work
    "unused_outputs",     # dead leaves — agents whose output no one reads
    "no_critique_loop",   # generate step with no critique→revision cycle
    "deep_chain",         # fragile long dependency chain
    "unnecessary_agents", # one or more agents don't earn their place for THIS
                          # task: either they don't contribute to the final
                          # output, or the step is overkill given the input
                          # size (e.g. chunking a short input; a classifier +
                          # fact_checker on a plain summary). The judge decides
                          # from the task, the input_word_count, and the
                          # per-node traces — NOT by catenating module names.
                          # The targeted node_ids go in target_nodes.
)


# ─── Intervention Library (Phase 2 — Reasoned Mutations) ─────────────────────

# The fixed set of pipeline edits the Architect can apply. Each value is the
# same verb as a Phase 2.1 PipelineDAG mutation primitive, so an Intervention
# is a declarative description of a primitive call the Architect dispatches.
MUTATION_TYPES: tuple[str, ...] = (
    "insert",       # splice a node in (→ insert_after / insert_before)
    "delete",       # → delete_node (with the all-to-all bypass)
    "parallelize",  # → parallelize (fan out a sibling alongside a node)
    "swap",         # → replace_node (swap an agent_type in place)
    "merge",        # → merge_chain (collapse a consecutive run to one node)
)

# Where in the pipeline an intervention lands. The Architect (Phase 2.3)
# resolves a slot to concrete node ids against a live pipeline + diagnosis:
#   before_generate    → the generate-role terminal (writer); insert BEFORE it
#                        so the inserted node's output flows INTO the writer.
#                        Backs no_validator — a fact_checker AFTER the writer
#                        would validate into a void (the engine extracts the
#                        writer's `output` regardless of position), so the
#                        validator must precede the producer of the final
#                        output. Matches the default pipeline's own shape:
#                        ... → summarizer → fact_checker → writer.
#   after_generate     → after the generate node. Backs no_critique_loop, but
#                        NOTE a critique→revision CYCLE isn't expressible as a
#                        simple insert in an acyclic DAG, AND the writer would
#                        still be extracted as terminal — so the critic's output
#                        would have to become the new terminal (an engine
#                        convention change). Shelved until 2.5 registers
#                        `critic` AND that convention is revisited; the seed
#                        documents the intent.
#   diagnosis_targets  → the diagnosis's own target_nodes (the evaluator
#                        already pinned the offending agents). Backs
#                        redundant_agents / unused_outputs / unnecessary_agents.
#   deep_chain_nodes   → the consecutive chain to collapse (the critical path
#                        or the diagnosis's target_nodes as a chain). Backs
#                        deep_chain.
#   bottleneck_node    → the serial node on the critical path to fan out.
#                        Backs serial_bottleneck.
TARGET_SLOTS: tuple[str, ...] = (
    "before_generate",
    "after_generate",
    "diagnosis_targets",
    "deep_chain_nodes",
    "bottleneck_node",
)


# ─── Roles vocabulary (Pipeline-agnostic resolution) ──────────────────────────
#
# Every primitive carries one semantic `role` on its Primitive definition. The
# Executor, Evaluator, and Architect key off a node's ROLE — not its primitive
# NAME — so that two pipelines with completely different primitive names but
# identical roles behave identically. This is the vocabulary that resolution
# (archforge.core.roles.RoleResolver) and the terminal-stage detection lean on,
# and it is the foundation for the pipeline-agnostic mutation logic.

ROLES: tuple[str, ...] = (
    "ingest",     # take raw input in (reader, fetcher, ...)
    "transform",  # reshape input without judging it (chunker, normalizer, ...)
    "analyze",    # interpret / classify / extract (classifier, summarizer, ...)
    "validate",  # verify claims / output against evidence (fact_checker, ...)
    "generate",   # produce the final deliverable (writer, composer, ...)
    "compose",    # merge / fan-in / route between branches (vote, fan_in, ...)
)

# Named members of ROLES, so callers import a symbol instead of string-typing a
# role and risking a typo the vocabulary can't catch.
ROLE_INGEST = "ingest"
ROLE_TRANSFORM = "transform"
ROLE_ANALYZE = "analyze"
ROLE_VALIDATE = "validate"
ROLE_GENERATE = "generate"
ROLE_COMPOSE = "compose"

# The role of the single node whose output the user receives. Resolution (the
# engine's final-output extraction and the structural evaluator's terminal
# leaf) used to hardcode the primitive NAME "writer"; this constant keys it on
# role instead. Any generate-role node is now a terminal candidate.
TERMINAL_ROLE = ROLE_GENERATE

# The role a resolver falls back to when a primitive name is unknown to it
# (a not-yet-registered evolved primitive, or a custom YAML drop-in the pool
# hasn't loaded). "analyze" is the neutral middle of the vocabulary — neither
# ingest/generate (which carry positional semantics) nor validate/compose
# (which carry gating semantics) — so a misclassified name does the least
# structural harm. Kept inert: the resolver exposes the role; nothing forces it.
DEFAULT_ROLE = ROLE_ANALYZE


# ─── Faithfulness evaluation (future Evaluator surface) ──────────────────────
#
# A planned evaluation surface: does the final output stay faithful to the
# evidence the pipeline gathered, vs inventing beyond it? This is distinct from
# `accuracy` (did the output satisfy the task?) and from `no_validator` (was a
# validate-role node present?) — a pipeline CAN have a validator and still
# produce an unfaithful answer. The knobs land here now so a later phase wires
# them in without a config restructure and without a STRUCTURAL_ROOTS
# relationship change.
#
# Deliberately inert by default: FAITHFULNESS_ENABLED is False and the weight is
# 0.0, so the existing (Phase 1-locked) composite and structural scoring are
# untouched — phase1.md pins the composite at 50% accuracy / 25% speed / 25%
# cost until Phase 6 makes weights learnable, and this respects that lock.
#
# FAITHFULNESS_ROOT is kept OUT of STRUCTURAL_ROOTS so a faithfulness diagnosis
# can never pivot the intervention matcher today (it would name a structural
# fix for a non-structural cause). It becomes a match key only when the
# faithfulness evaluator itself ships.
FAITHFULNESS_ENABLED = False
FAITHFULNESS_LOW = 0.6        # faithfulness normalized below this is "unfaithful"
FAITHFULNESS_ROOT = "unfaithful_output"
FAITHFULNESS_WEIGHT = 0.0     # blend weight into composite — 0.0 ⇒ off


# ─── Intervention learning (Reasoned Mutations — diagnosis-driven) ────────────
#
# NOTE: this system is diagnosis-driven, NOT exploration-driven. The Architect
# matches each diagnosis's `structural_root` to candidate interventions via
# InterventionLibrary.match_by_root and selects max(cands, key=(success_rate,
# id)). There is NO random-mutation exploration branch anywhere in the code
# (the only "random" in archforge/ is a comment about the pipeline id). So
# these knobs govern LEARNED-SELECTION and OUTCOME-RECORDING — there is
# explicitly no "explore/exploit threshold" or random-mutation gate among them.
#
# Landed now (inert / pre-wired) so Phase 2.4 (intervention success tracking)
# is pure-update-no-migration, mirroring how Phase 1 carried empty diagnosis
# placeholders into Phase 2.

INTERVENTION_LEARNING_ENABLED = True
# Gate: when True the Architect prefers the matched candidate with the highest
# learned success_rate; when False it applies seeds in fixed order. The current
# Architect always ranks by success_rate, so True is a no-op today and becomes
# the switch a "trust learned rate vs seed order" policy flips later.

INTERVENTION_SUCCESS_PRIOR = 0.5
# Starting success_rate for an unobserved (freshly seeded or newly discovered)
# intervention. A NEUTRAL prior — neither trusted nor distrusted — until a run
# records an outcome. Replaces the literal 0.5 that lived inline in
# interventions.py. NOT an "explore/exploit boundary"; there is no random
# mutation to bound.

INTERVENTION_MIN_SAMPLES = 0
# times_tried ≥ this before the learned rate is trusted OVER the prior. A
# freshly-seeded intervention sits at the prior until it has been observed
# enough to learn from. 0 ⇒ trust the rate from the first try.

INTERVENTION_HELP_MIN_DELTA = 0.0
# An intervention is recorded as "helped" when the post-run composite beats the
# pre-intervention baseline by ≥ this. This is an OUTCOME-RECORDING threshold
# (did the fix help?), the opposite axis from a random-mutation "exploration"
# knob. 0.0 ⇒ any improvement counts. Phase 2.4 consumes this when it bumps
# times_tried/times_helped and recomputes success_rate.


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
    "DIAGNOSIS_ACCURACY_LOW",
    "DIAGNOSIS_SPEED_LOW",
    "DIAGNOSIS_COST_LOW",
    "DIAGNOSIS_PARALLELISM_LOW",
    "DIAGNOSIS_BOTTLENECK_MIN_PATH",
    "DIAGNOSIS_DEEP_CHAIN_MIN",
    "STRUCTURAL_ROOTS",
    "MUTATION_TYPES",
    "TARGET_SLOTS",
    "ROLES",
    "ROLE_INGEST",
    "ROLE_TRANSFORM",
    "ROLE_ANALYZE",
    "ROLE_VALIDATE",
    "ROLE_GENERATE",
    "ROLE_COMPOSE",
    "TERMINAL_ROLE",
    "DEFAULT_ROLE",
    "FAITHFULNESS_ENABLED",
    "FAITHFULNESS_LOW",
    "FAITHFULNESS_ROOT",
    "FAITHFULNESS_WEIGHT",
    "INTERVENTION_LEARNING_ENABLED",
    "INTERVENTION_SUCCESS_PRIOR",
    "INTERVENTION_MIN_SAMPLES",
    "INTERVENTION_HELP_MIN_DELTA",
]
