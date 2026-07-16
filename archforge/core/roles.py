"""Role resolver — map primitive names to semantic roles.

A Primitive carries a `role` (ingest|transform|analyze|validate|generate|compose).
But a pipeline's ``AgentNode`` stores only the primitive NAME (``agent_type``); the
role lives on the pool's primitive definitions, not on the node. ``RoleResolver``
is the bridge: built once from the pool (the source of truth for primitive→role),
then queried by name.

Keying the terminal generation stage and node-location on ROLE — not NAME — is what
makes two pipelines with completely different primitive names but identical roles
behave identically (see ``config.ROLES`` / ``config.TERMINAL_ROLE``). It is also
the foundation the Executor, Evaluator, and Architect build on instead of each
rebuilding a `{name: role}` dict by hand.

Core-layer only: depends on ``core.primitive`` + ``config`` (never on the executor
package), so the resolution abstraction does not pull the agent registry into the
data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from ..config import DEFAULT_ROLE, ROLES
from ..logging import get_logger

if TYPE_CHECKING:
    from .primitive import Primitive

log = get_logger("core.roles")


@dataclass(frozen=True)
class RoleResolver:
    """Immutable name→role mapping.

    Built from a pool's primitives (the source of truth). The only public
    surface is queries: ``role_of`` / ``role_of_node`` / ``as_dict``. Roles
    outside ``config.ROLES`` are validated LENIENTLY — mapped to
    ``DEFAULT_ROLE`` and logged at debug, never raised — so a custom YAML
    drop-in with a typo'd role can't crash a run. An unknown NAME (not in the
    pool) simply returns ``None`` from ``role_of``.

    Frozen: the mapping is wrapped in a ``MappingProxyType`` and the dataclass
    is frozen, so neither the resolver nor its contents are mutable after build.
    """

    _roles: MappingProxyType  # name → role (validated, read-only)

    def __post_init__(self) -> None:
        # Validate leniently: keep vocabulary roles as-is; anything novel maps
        # to DEFAULT_ROLE. TERMINAL_ROLE is a ROLES member, so a mis-mapped
        # primitive can never accidentally satisfy the terminal-stage test —
        # it simply becomes "analyze" and is ignored by generate-keyed logic.
        cleaned: dict[str, str] = {}
        for name, role in self._roles.items():
            if role in ROLES:
                cleaned[name] = role
            else:
                log.debug(
                    "RoleResolver: primitive %r has unknown role %r → %r",
                    name, role, DEFAULT_ROLE,
                )
                cleaned[name] = DEFAULT_ROLE
        object.__setattr__(self, "_roles", MappingProxyType(cleaned))

    # ----- constructors -----

    @classmethod
    def from_primitives(cls, primitives: Mapping[str, "Primitive"]) -> "RoleResolver":
        """Build from a {name: Primitive} mapping (reads ``p.role``)."""
        mapping = {name: p.role for name, p in primitives.items()}
        return cls(MappingProxyType(mapping))

    @classmethod
    def from_pool(cls, pool) -> "RoleResolver":
        """Build from anything exposing ``.primitives() -> {name: Primitive}``.

        Duck-typed on purpose: the real ``PrimitivePool`` AND the lightweight
        ``_StubPool`` used by the intervention tests (which defines
        ``primitives()`` but no ``role_resolver()`` of its own) both satisfy
        this, so the resolver composes with the same pool protocol the test
        stubs already speak.
        """
        return cls.from_primitives(pool.primitives())

    # ----- queries -----

    def role_of(self, name: str) -> str | None:
        """Role of a primitive name, or ``None`` if the name is unknown."""
        return self._roles.get(name)

    def role_of_node(self, node) -> str | None:
        """Role of an ``AgentNode`` (duck-typed: reads ``node.agent_type``).

        Returns ``None`` for a node whose primitive name isn't in the pool —
        which is exactly the signal "this node plays no known role here".
        """
        return self._roles.get(node.agent_type)

    def as_dict(self) -> dict[str, str]:
        """A plain ``{name: role}`` copy.

        Projection of the resolver for callers that take ``roles: dict[str,str]``
        (``OutputEvaluator`` / ``Diagnostician``). The resolver remains the
        single source of truth; this is a snapshot, not a shared reference.
        """
        return dict(self._roles)

    def __len__(self) -> int:
        return len(self._roles)


__all__ = ["RoleResolver"]
