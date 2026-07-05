"""Primitive registry.

Phase 1 ships 6 base primitives, defined in code. YAML overrides (per
`data/primitives/`) are optional; if present, they extend or replace
the in-code defaults at registry-init time. Phase 5+ will add evolved
primitives at runtime.

To add a new base primitive:
  1. Drop a `<name>.yaml` into data/primitives/ OR
  2. Add a constructor to BUILTIN_CONSTRUCTORS.
That's it — no import gymnastics needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import yaml

from ...core.primitive import Primitive
from .base import BaseAgent
from .chunker import ChunkerAgent
from .classifier import ClassifierAgent
from .fact_checker import FactCheckerAgent
from .reader import ReaderAgent
from .summarizer import SummarizerAgent
from .writer import WriterAgent


# ─── constructors for the 6 base primitives ────────────────────────────────

BUILTIN_CONSTRUCTORS: dict[str, Callable[[], BaseAgent]] = {
    "reader": lambda: ReaderAgent(),
    "chunker": lambda: ChunkerAgent(),
    "classifier": lambda: ClassifierAgent(),
    "summarizer": lambda: SummarizerAgent(),
    "fact_checker": lambda: FactCheckerAgent(),
    "writer": lambda: WriterAgent(),
}


def default_data_dir() -> Path:
    """Resolve <project_root>/data unless overridden by env.

    The package sits at <project_root>/archforge/__init__.py so
    walking two parents up reaches the project root.
    """
    override = os.environ.get("ARCHFORGE_DATA_DIR")
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
        self._agents: dict[str, BaseAgent] = {}
        self._primitives: dict[str, Primitive] = {}
        self._data_dir = data_dir or default_data_dir()
        self._loaded_yamls = False

    # ----- lookups -----

    def get(self, name: str) -> BaseAgent:
        if not self._loaded_yamls:
            self._load_yamls()
        if name not in self._agents:
            # Late-bind built-ins so YAML overrides win.
            if name in BUILTIN_CONSTRUCTORS:
                agent = BUILTIN_CONSTRUCTORS[name]()
                self._agents[name] = agent
                self._primitives[name] = agent.primitive
            else:
                raise KeyError(f"Unknown primitive: {name!r}")
        return self._agents[name]

    def primitive(self, name: str) -> Primitive:
        self.get(name)  # ensure loaded
        return self._primitives[name]

    def names(self) -> list[str]:
        if not self._loaded_yamls:
            self._load_yamls()
        return sorted(set(self._agents) | set(BUILTIN_CONSTRUCTORS))

    def primitives(self) -> dict[str, Primitive]:
        if not self._loaded_yamls:
            self._load_yamls()
        # Materialise every built-in so the dict is complete.
        for name in list(BUILTIN_CONSTRUCTORS):
            self.get(name)
        return dict(self._primitives)

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


# Singleton used by simpler call sites.
_default_pool: PrimitivePool | None = None


def default_pool() -> PrimitivePool:
    global _default_pool
    if _default_pool is None:
        _default_pool = PrimitivePool()
    return _default_pool


__all__ = ["PrimitivePool", "default_pool", "default_data_dir"]
