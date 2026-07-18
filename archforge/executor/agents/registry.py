"""Primitive registry.

Every primitive — whether LLM-backed (bound through ``build_llm_agent``) or a
plain deterministic callable — is a ``(Primitive, AgentCallable)`` entry in
``AGENT_SPECS``. This is the general "agent = description + callable" contract
(spec §2) that replaced the class-based ``BUILTIN_CONSTRUCTORS`` in Phase 3.

YAML overrides (per ``data/primitives/``) are optional; if present, they extend
or replace in-code metadata at registry-init time. Full YAML-driven agents
(prompt templates driving a generic callable) are later work. Phase 5+ will
add evolved primitives at runtime.

To add a new primitive:
  - LLM primitive:       add a ``(Primitive, build_llm_agent(spec, shape))`` entry
                         to ``AGENT_SPECS`` (spec + shape fn, see ``reader.py``).
  - Deterministic agent: add a ``(Primitive, callable)`` entry to ``AGENT_SPECS``
                         (see ``regex_extractor.py``).
  - Metadata-only:       drop a ``<name>.yaml`` into ``data/primitives/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ...config import DATA_DIR_ENV
from ...core.primitive import Primitive
from ...core.roles import RoleResolver
from .base import Agent, AgentCallable
from .chunker import CHUNKER_AGENT, CHUNKER_SPEC
from .classifier import CLASSIFIER_AGENT, CLASSIFIER_SPEC
from .critic import CRITIC_AGENT, CRITIC_SPEC
from .fact_checker import FACT_CHECKER_AGENT, FACT_CHECKER_SPEC
from .reader import READER_AGENT, READER_SPEC
from .regex_extractor import REGEX_SPEC, regex_extractor
from .summarizer import SUMMARIZER_AGENT, SUMMARIZER_SPEC
from .writer import WRITER_AGENT, WRITER_SPEC


# ─── general agent-spec registry ────────────────────────────────────────────
#
# Maps a primitive name to its (Primitive, AgentCallable). This IS the agent
# contract (spec §2): a primitive is a description + a callable, not a class.
# LLM primitives bind through build_llm_agent; deterministic agents bind a
# plain callable directly (see regex_extractor).
AGENT_SPECS: dict[str, tuple[Primitive, AgentCallable]] = {
    "reader": (READER_SPEC, READER_AGENT),
    "chunker": (CHUNKER_SPEC, CHUNKER_AGENT),
    "classifier": (CLASSIFIER_SPEC, CLASSIFIER_AGENT),
    "summarizer": (SUMMARIZER_SPEC, SUMMARIZER_AGENT),
    "fact_checker": (FACT_CHECKER_SPEC, FACT_CHECKER_AGENT),
    "writer": (WRITER_SPEC, WRITER_AGENT),
    "critic": (CRITIC_SPEC, CRITIC_AGENT),
    "regex_extractor": (REGEX_SPEC, regex_extractor),
}


def default_data_dir() -> Path:
    """Resolve <project_root>/data unless overridden by env.

    The package sits at <project_root>/archforge/__init__.py so
    walking two parents up reaches the project root.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "data"


# ─── registry ───────────────────────────────────────────────────────────────


class PrimitivePool:
    """Catalogue of available primitives.

    Lookups are by name. Built-ins are loaded on first read of `primitives()`
    and can be extended with YAML files dropped in `data/primitives/`.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        self._primitives: dict[str, Primitive] = {}
        self._data_dir = data_dir or default_data_dir()
        self._loaded_yamls = False
        self._resolver: RoleResolver | None = None

    # ----- lookups -----

    def get(self, name: str) -> Agent:
        if not self._loaded_yamls:
            self._load_yamls()
        if name not in self._agents:
            # Late-bind directly from AGENT_SPECS. (YAML overrides, if any,
            # are loaded into self._primitives by _load_yamls but overwritten
            # here by the in-code spec — matching the original behavior.
            # Full YAML-driven agents are later work.)
            if name in AGENT_SPECS:
                primitive, call = AGENT_SPECS[name]
                self._agents[name] = Agent(primitive=primitive, call=call)
                self._primitives[name] = primitive
            else:
                raise KeyError(f"Unknown primitive: {name!r}")
        return self._agents[name]

    def primitive(self, name: str) -> Primitive:
        self.get(name)  # ensure loaded
        return self._primitives[name]

    def names(self) -> list[str]:
        if not self._loaded_yamls:
            self._load_yamls()
        return sorted(set(self._agents) | set(AGENT_SPECS))

    def primitives(self) -> dict[str, Primitive]:
        if not self._loaded_yamls:
            self._load_yamls()
        # Materialise every primitive so the dict is complete.
        for name in sorted(set(AGENT_SPECS)):
            self.get(name)
        return dict(self._primitives)

    def role_resolver(self) -> RoleResolver:
        """The {name → role} resolver for this pool's primitives.

        Memoized after first build (the primitive set is fixed once the YAMLs
        have loaded here; evolved primitives arriving at runtime in Phase 5+
        invalidate the cache only if the base set itself changes). The single
        producer of the role mapping the Executor / Evaluator / Architect
        consume — replaces the ad-hoc ``{name: p.role ...}`` rebuilds that
        used to live in main.py, designer, and interventions.
        """
        if self._resolver is None:
            self._resolver = RoleResolver.from_primitives(self.primitives())
        return self._resolver

    # ----- YAML overrides -----

    def _load_yamls(self) -> None:
        self._loaded_yamls = True
        prim_dir = self._data_dir / "primitives"
        if not prim_dir.is_dir():
            return
        for path in sorted(prim_dir.glob("*.yaml")):
            with path.open() as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict) or "name" not in data:
                continue
            primitive = Primitive.from_dict(data)
            self._primitives[primitive.name] = primitive
            # The agent class itself is still bound to the in-code
            # implementation; YAML only updates prompt/schema metadata.
            # For Phase 1 that's sufficient — base implementations
            # share the same code path. Full YAML-driven agents
            # come with Phase 4 template work.
            # Note: get() clobbers this entry with the in-code primitive on
            # first access, matching the original behavior (metadata overrides
            # are observed only via _primitives before a get()).


# Singleton used by simpler call sites.
_default_pool: PrimitivePool | None = None


def default_pool() -> PrimitivePool:
    global _default_pool
    if _default_pool is None:
        _default_pool = PrimitivePool()
    return _default_pool


def default_role_resolver() -> RoleResolver:
    """The role resolver for the singleton default pool.

    Mirrors ``default_pool()``: lazy, shared, memoized on the pool. The
    callers that previously rebuilt ``{name: p.role ...}`` from
    ``default_pool().primitives()`` use this instead.
    """
    return default_pool().role_resolver()


__all__ = ["PrimitivePool", "default_pool", "default_role_resolver", "default_data_dir"]
