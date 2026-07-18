"""Intervention Library — Phase 2 sub-phase 2.2 (Reasoned Mutations).

A structured, machine-matchable mapping from a diagnosis `structural_root`
to a pipeline mutation, per plan.md's "Reasoned Mutations" subsystem.

An Intervention is a *declarative description of a Phase 2.1 primitive call*:
  - `diagnosis_pattern` : which STRUCTURAL_ROOT this fixes (the match key)
  - `mutation_type`      : a MUTATION_TYPES verb (insert|delete|parallelize|
                           swap|merge) — the same verb as a PipelineDAG
                           mutation primitive
  - `target_slot`        : a TARGET_SLOTS abstraction the Architect (Phase 2.3)
                           resolves to concrete node ids against a live pipeline
  - `agent_to_insert`    : the primitive to insert/swap BY CONCRETE NAME, or
                           None when the mutation inserts nothing (delete/merge)
                           or derives the type at resolution (parallelize → the
                           target node's own agent_type) or names a ROLE instead
                           (see `agent_role`)
  - `agent_role`         : for insert, the ROLE of the primitive to insert,
                           resolved from the active pool at dispatch (Task 6).
                           Mutually exclusive with `agent_to_insert`: an
                           intervention names the primitive EITHER by a
                           concrete name (the future `critic` seed) OR by role
                           (the `no_validator` seed → ROLE_VALIDATE). Naming by
                           role is what makes the same intervention work across
                           arbitrary primitive vocabularies — it never bakes in
                           a primitive name like `fact_checker`.

So the Architect's mutation step (2.3) becomes a dispatch table:
  diagnosis → match_by_root → resolve target_slot → call the 2.1 primitive.
The library itself is data, not behaviour.

Learned state (`success_rate`/`times_tried`/`times_helped`/`last_updated`)
ships at a 0.5 prior now and round-trips through JSON, so Phase 2.4 (success
tracking) is purely the update call after a run — no schema migration. This
mirrors Phase 1 carrying empty `diagnoses`/`structural` placeholders into
Phase 2.

Applicability is split (see ``is_structurally_eligible``):
  - structural eligibility (pool + topology; diagnosis-free) lives HERE —
    fully unit-testable offline.
  - diagnosis-aware eligibility (the diagnosis's own `target_nodes`;
    idempotency against the live pipeline) lives in 2.3, which holds the
    diagnosis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator

from ..config import (
    DIAGNOSIS_BOTTLENECK_MIN_PATH,
    DIAGNOSIS_DEEP_CHAIN_MIN,
    INTERVENTION_SUCCESS_PRIOR,
    MUTATION_TYPES,
    ROLE_GENERATE,
    ROLE_VALIDATE,
    ROLES,
    STRUCTURAL_ROOTS,
    TARGET_SLOTS,
)
from ..core.pipeline import PipelineDAG
from ..core.roles import RoleResolver
from ..executor.agents.registry import PrimitivePool
from ..logging import get_logger

log = get_logger("architect.interventions")

# mutation_types that MUST carry a concrete agent_to_insert, and those that
# MUST NOT. `parallelize` derives its sibling from the target node's own
# agent_type at resolution time, so it takes None here.
_INSERTS_AGENT = {"insert", "swap"}
_NO_INSERTED_AGENT = {"delete", "merge", "parallelize"}


@dataclass
class Intervention:
    """One diagnosis→mutation rule + its learned track record."""

    id: str
    diagnosis_pattern: str  # a STRUCTURAL_ROOTS value (the match key)
    mutation_type: str      # a MUTATION_TYPES verb (a Phase 2.1 primitive)
    target_slot: str        # a TARGET_SLOTS abstraction (resolved in 2.3)
    agent_to_insert: str | None = None  # concrete primitive name (OR use agent_role)
    agent_role: str | None = None       # role to resolve from the pool (Task 6)

    # Learned state — Phase 2.4 wires the updates. Carried now at a neutral
    # prior (config.INTERVENTION_SUCCESS_PRIOR) so the dataclass round-trips
    # through JSON from day one. A NEUTRAL prior — neither trusted nor
    # distrusted — until a run records an outcome. This system is
    # diagnosis-driven (the Architect selects the matched candidate by
    # success_rate), NOT exploration-driven: there is no random-mutation
    # explore/exploit branch, so the prior is not an "explore/explore
    # boundary" — it is simply where an unobserved fix starts.
    success_rate: float = INTERVENTION_SUCCESS_PRIOR
    times_tried: int = 0
    times_helped: int = 0
    last_updated: datetime | None = None

    def __post_init__(self) -> None:
        # Closed-vocabulary, fail-fast at construction (catches seed typos
        # and corrupted seeds on load). Structural validity ≠ pool
        # eligibility: agent_to_insert is a free primitive name, validated
        # against the pool at runtime by is_structurally_eligible.
        if self.diagnosis_pattern not in STRUCTURAL_ROOTS:
            raise ValueError(
                f"Intervention {self.id!r}: diagnosis_pattern "
                f"{self.diagnosis_pattern!r} not in STRUCTURAL_ROOTS")
        if self.mutation_type not in MUTATION_TYPES:
            raise ValueError(
                f"Intervention {self.id!r}: mutation_type "
                f"{self.mutation_type!r} not in MUTATION_TYPES")
        if self.target_slot not in TARGET_SLOTS:
            raise ValueError(
                f"Intervention {self.id!r}: target_slot "
                f"{self.target_slot!r} not in TARGET_SLOTS")
        # An insert/swap names the primitive to insert/swap EITHER by a
        # concrete name (agent_to_insert) OR by role (agent_role) — never
        # both, never neither. Role-based naming (Task 6) is what makes a
        # seed like no_validator work across arbitrary primitive
        # vocabularies: it says "the validate-primitive the pool offers",
        # not "fact_checker".
        if self.mutation_type in _INSERTS_AGENT:
            if not self.agent_to_insert and not self.agent_role:
                raise ValueError(
                    f"Intervention {self.id!r}: mutation_type {self.mutation_type!r} "
                    "requires a concrete agent_to_insert OR an agent_role")
            if self.agent_to_insert and self.agent_role:
                raise ValueError(
                    f"Intervention {self.id!r}: specify agent_to_insert OR "
                    f"agent_role, not both")
        if self.mutation_type in _NO_INSERTED_AGENT and (
                self.agent_to_insert or self.agent_role):
            raise ValueError(
                f"Intervention {self.id!r}: mutation_type {self.mutation_type!r} "
                "must not set agent_to_insert (inserts nothing / derives it)")
        if self.agent_role is not None and self.agent_role not in ROLES:
            raise ValueError(
                f"Intervention {self.id!r}: agent_role {self.agent_role!r} "
                "not in ROLES")

    # ----- serialization -----

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "diagnosis_pattern": self.diagnosis_pattern,
            "mutation_type": self.mutation_type,
            "target_slot": self.target_slot,
            "agent_to_insert": self.agent_to_insert,
            "agent_role": self.agent_role,
            "success_rate": self.success_rate,
            "times_tried": self.times_tried,
            "times_helped": self.times_helped,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Intervention":
        ts = data.get("last_updated")
        last = datetime.fromisoformat(ts) if isinstance(ts, str) else None
        return cls(
            id=data["id"],
            diagnosis_pattern=data["diagnosis_pattern"],
            mutation_type=data["mutation_type"],
            target_slot=data["target_slot"],
            agent_to_insert=data.get("agent_to_insert"),
            agent_role=data.get("agent_role"),
            success_rate=data.get("success_rate", INTERVENTION_SUCCESS_PRIOR),
            times_tried=data.get("times_tried", 0),
            times_helped=data.get("times_helped", 0),
            last_updated=last,
        )


# ─── seeded library ──────────────────────────────────────────────────────────


def default_interventions() -> list[Intervention]:
    """One seed per *live* STRUCTURAL_ROOT (7 for 7), per plan.md's
    intervention table — updated for the vocabulary as it actually ships:

    - ``over_chunking`` (plan table) is DEAD: cost-overwork is diagnosed as
      `unnecessary_agents` (see the combined judge prompt — "NOT a separate
      root"), so it has no seed here.
    - `unnecessary_agents` (live, absent from the plan table) gets a DELETE
      seed targeting the nodes the diagnosis itself named.
    - `no_critique_loop` seeds ``critic`` — now a REGISTERED base primitive
      (``generate`` role, after the writer, verify-and-revise terminal). The
      `no_validator` diagnosis points at a missing VERIFY of the final answer;
      the critic IS that verify-and-revise step (it checks the writer's draft
      against the source and emits the grounded output as the terminal).
      ``is_structurally_eligible``'s pool gate
      (``agent_to_insert in pool.names()``) now passes.
    """
    return [
        Intervention(
            id="iv-no_validator-insert-fact_checker",
            diagnosis_pattern="no_validator",
            mutation_type="insert",
            # BEFORE the writer, not after: the validator's verdict must flow
            # INTO the producer of the final output. A validator after the
            # writer validates into a void — the engine extracts the writer's
            # `output` regardless of position, so a post-writer validator's
            # verdicts would be discarded. This matches the default pipeline's
            # own shape (... → summarizer → fact_checker → writer).
            target_slot="before_generate",
            # ROLE, not a primitive name (Task 6): the seed names the validate
            # ROLE and the Architect resolves whichever validate primitive the
            # ACTIVE pool offers at dispatch (fact_checker in the default pool;
            # a custom pipeline's "verifier" / "checker" / any name). The id
            # still says "fact_checker" as a human-scannable historical label,
            # but NOTHING in the architecture depends on that name — the same
            # intervention inserts a validator across arbitrary vocabularies.
            agent_role=ROLE_VALIDATE,
        ),
        Intervention(
            id="iv-serial_bottleneck-parallelize",
            diagnosis_pattern="serial_bottleneck",
            mutation_type="parallelize",
            target_slot="bottleneck_node",
            agent_to_insert=None,  # sibling = the bottleneck node's own agent_type
        ),
        Intervention(
            id="iv-redundant_agents-delete",
            diagnosis_pattern="redundant_agents",
            mutation_type="delete",
            target_slot="diagnosis_targets",
            agent_to_insert=None,
        ),
        Intervention(
            id="iv-unused_outputs-delete",
            diagnosis_pattern="unused_outputs",
            mutation_type="delete",
            target_slot="diagnosis_targets",
            agent_to_insert=None,
        ),
        Intervention(
            id="iv-no_critique_loop-insert-critic",
            diagnosis_pattern="no_critique_loop",
            mutation_type="insert",
            target_slot="after_generate",
            # AFTER the writer: the critic is a verify-and-revise TERMINAL
            # (generate-role leaf → _extract_final_output picks its `output`
            # for free, no engine change). It checks the writer's draft against
            # the forwarded source and emits the grounded answer — unlike a
            # verdict-only validator after the writer, its output is NOT dead
            # (it IS the terminal, so no unused_outputs self-trigger).
            agent_to_insert="critic",
        ),
        Intervention(
            id="iv-deep_chain-merge",
            diagnosis_pattern="deep_chain",
            mutation_type="merge",
            target_slot="deep_chain_nodes",
            agent_to_insert=None,
        ),
        Intervention(
            id="iv-unnecessary_agents-delete",
            diagnosis_pattern="unnecessary_agents",
            mutation_type="delete",
            target_slot="diagnosis_targets",
            agent_to_insert=None,
        ),
    ]


class InterventionLibrary:
    """Indexed set of interventions, keyed by the structural_root they fix."""

    def __init__(self, interventions: Iterable[Intervention] | None = None) -> None:
        self._by_root: dict[str, list[Intervention]] = defaultdict(list)
        self._by_id: dict[str, Intervention] = {}
        for iv in (interventions if interventions is not None else default_interventions()):
            self.register(iv)

    def register(self, iv: Intervention) -> None:
        if iv.id in self._by_id:
            raise ValueError(f"Intervention id {iv.id!r} already registered")
        self._by_id[iv.id] = iv
        self._by_root[iv.diagnosis_pattern].append(iv)
        log.debug("register: %s → %s %s", iv.id, iv.mutation_type, iv.target_slot)

    def match_by_root(self, root: str) -> list[Intervention]:
        """All interventions whose diagnosis_pattern == `root`. Empty if none.

        This is the Architect's (2.3) lookup: given a diagnosis's
        structural_root, return the candidate fixes. Selection among
        candidates (by success_rate) is the Architect's job.
        """
        return list(self._by_root.get(root, []))

    def get(self, iv_id: str) -> Intervention | None:
        return self._by_id.get(iv_id)

    def all(self) -> list[Intervention]:
        return list(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Intervention]:
        return iter(self._by_id.values())

    # Phase 2.4 will add: record_outcome(iv_id, helped: bool) -> None
    # which bumps times_tried/times_helped and recomputes success_rate.


# ─── structural eligibility (diagnosis-free) ────────────────────────────────


def is_structurally_eligible(
    iv: Intervention, pipeline: PipelineDAG, pool: PrimitivePool,
) -> bool:
    """Can `iv` typecheck against reality, WITHOUT a diagnosis?

    Two diagnosis-free gates:
      (1) pool — the primitive to insert is resolvable from the pool: either
          a concrete `agent_to_insert` is a registered primitive, OR an
          `agent_role` is offered by ≥1 primitive in the pool (Task 6 — a
          role-based seed like no_validator names the validate ROLE, and the
          gate confirms the pool HAS a validate-role primitive to insert).
      (2) topology — a diagnosis-free `target_slot` resolves to ≥1 real node
          in this pipeline. The thresholds reuse the Diagnostician's own
          DIAGNOSIS_* floors, so an intervention for root X is structurally
          eligible exactly when the topology is in the regime where root X's
          diagnosis could fire. The generate/validate topology checks key on
          ROLE (via a resolver built from the pool), not primitive name, so an
          intervention is eligible for a pipeline whose generate primitive is
          named anything at all.

    Diagnosis-aware slots (`diagnosis_targets`) cannot be resolved without the
    diagnosis; this gate checks only the pool half and returns True for them.
    The diagnosis-aware gate (the diagnosis's own `target_nodes` + idempotency
    against the live pipeline) is the Architect's, Phase 2.3.

    A cyclic pipeline is never eligible — `depth()`/`critical_path()` would
    loop, and the engine refuses to execute such a pipeline anyway.
    """
    if pipeline.has_cycle():
        return False

    resolver = RoleResolver.from_pool(pool)

    # (1) pool gate — concrete name registered OR a named role is offered.
    if iv.agent_to_insert is not None and iv.agent_to_insert not in pool.names():
        return False
    if iv.agent_role is not None and not any(
            r == iv.agent_role for r in resolver.as_dict().values()):
        return False

    # (2) topology gate, diagnosis-free slots only
    if iv.target_slot in ("before_generate", "after_generate"):
        # Both insert slots resolve to the generate-role node; the eligibility
        # check is identical — a generate node must exist to insert relative
        # to. WHICH side (before vs after) is a correctness property of the
        # seed, not the gate: no_validator uses before_generate (verdict must
        # flow into the generator); no_critique_loop uses after_generate (and
        # is shelved for cycle/terminal reasons — see the seed comment).
        return pipeline.has_role(ROLE_GENERATE, resolver)
    if iv.target_slot == "deep_chain_nodes":
        return pipeline.depth() >= DIAGNOSIS_DEEP_CHAIN_MIN
    if iv.target_slot == "bottleneck_node":
        crit_edges = max(0, len(pipeline.critical_path()) - 1)
        return crit_edges >= DIAGNOSIS_BOTTLENECK_MIN_PATH
    if iv.target_slot == "diagnosis_targets":
        return True  # diagnosis-aware half lives in the Architect (2.3)
    return False


__all__ = [
    "Intervention",
    "InterventionLibrary",
    "default_interventions",
    "is_structurally_eligible",
]
