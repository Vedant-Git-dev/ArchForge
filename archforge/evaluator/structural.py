"""Structural quality evaluator (Phase 2).

Computes the pipeline's structural metrics — Surface 2 of plan.md's
Evaluator: pipeline length, critical path, parallelism ratio, redundant
agents, unused outputs, dependency depth — plus a baseline structural score.

The metrics are derived purely from the `PipelineDAG` topology: no LLM,
no execution, no runtime outputs. This module is what populates the
`Experience.structural` field that Phase 1 left zeroed — see the Phase 1
record, "Phase 2 entry points (already reserved in the schema)".

Phase 2 deliberately does NOT fold the structural score into the composite.
phase1.md locks the composite formula (50% accuracy / 25% speed / 25% cost)
until Phase 6 makes the per-task-type weights learnable. The metrics are
computed and stored now so they exist for retrieval ranking, diagnosis +
interventions (later Phase 2 work), and weight learning (Phase 6).

Structural scoring baseline (plan.md "Structural Evaluation"):

    score = max(0, 1.0 - 0.10*unused_outputs - 0.15*redundant_agents)

The "deviation from a learned ideal profile" term is Phase 6; for now the
score reflects only the hard-constraint penalties against a neutral 1.0.
"""

from __future__ import annotations

from ..config import (
    STRUCTURAL_REDUNDANT_PENALTY,
    STRUCTURAL_UNUSED_PENALTY,
)
from ..core.experience import StructuralScores
from ..core.pipeline import AgentNode, PipelineDAG
from ..logging import get_logger

log = get_logger("evaluator.structural")


def _terminal_node(pipeline: PipelineDAG) -> AgentNode | None:
    """The single node whose output the user receives.

    Mirrors the engine's `_extract_final_output` convention: prefer a
    `writer` leaf; otherwise the leaf that is last in topo order (the
    de-facto final stage of a linear replay). Other leaves feed no one
    and are counted as unused outputs.
    """
    leaves = pipeline.leaves()
    if not leaves:
        return None
    for leaf in leaves:
        if leaf.agent_type == "writer":
            return leaf
    order = pipeline.topo_order()
    order_index = {n.id: i for i, n in enumerate(order)}
    return max(leaves, key=lambda n: order_index[n.id])


def _unused_outputs(pipeline: PipelineDAG) -> list[str]:
    """Leaf node ids whose output no downstream agent consumes.

    A well-formed pipeline has exactly one terminal leaf; every other
    leaf produces dead output. These are the plan's "unused_outputs"
    (structural dead code) — distinct from runtime unused *values*, which
    a later phase can detect from execution traces.
    """
    terminal = _terminal_node(pipeline)
    if terminal is None:
        return []
    return [leaf.id for leaf in pipeline.leaves() if leaf.id != terminal.id]


def _redundant_agents(pipeline: PipelineDAG) -> list[str]:
    """Deletable duplicates among structurally-identical agents.

    Two nodes are "twins" when they share agent_type, predecessor set,
    and successor set — they occupy the same structural slot, so one is
    redundant. This is a conservative structural proxy for the plan's
    ">80% output Jaccard" redundancy, which needs runtime outputs and is
    left to a later phase. Exact structural twins are unambiguous.

    For a twin group of size k, k-1 are deletable duplicates (one survives).
    The returned ids are the deletable ones, so `len(...) == penalty count`.
    """
    groups: dict[tuple, list[str]] = {}
    for n in pipeline.nodes:
        preds = tuple(sorted(p.id for p in pipeline.predecessors(n.id)))
        succs = tuple(sorted(s.id for s in pipeline.successors(n.id)))
        sig = (n.agent_type, preds, succs)
        groups.setdefault(sig, []).append(n.id)

    redundant: list[str] = []
    for ids in groups.values():
        if len(ids) > 1:
            redundant.extend(ids[1:])  # one survives per group
    return sorted(redundant)


class StructuralEvaluator:
    """Compute `StructuralScores` from a pipeline's topology.

    Stateless caller — instantiate once per session, like `OutputEvaluator`.
    No LLM, no network; safe on the offline / no-API-key demo path.
    """

    def evaluate(self, pipeline: PipelineDAG) -> StructuralScores:
        n = len(pipeline.nodes)
        if n == 0 or pipeline.has_cycle():
            # Empty has nothing to measure; a cyclic pipeline never reaches
            # here in practice (the engine refuses to execute it), but we
            # return the zero default defensively rather than raising.
            log.debug("evaluate: empty/cyclic pipeline (nodes=%d) → zero scores", n)
            return StructuralScores()

        critical_path = pipeline.critical_path()
        critical_path_length = max(0, len(critical_path) - 1)  # edges
        dependency_depth = pipeline.depth()  # nodes on the longest path

        unused = _unused_outputs(pipeline)
        redundant = _redundant_agents(pipeline)

        # Parallelism ratio = fraction of nodes NOT on the critical path.
        # A linear pipeline has every node serial, so ratio 0.0; a fan-out
        # moves nodes off the critical path, so the ratio rises.
        parallelism_ratio = (n - dependency_depth) / n if n else 0.0

        score = max(
            0.0,
            1.0
            - STRUCTURAL_UNUSED_PENALTY * len(unused)
            - STRUCTURAL_REDUNDANT_PENALTY * len(redundant),
        )

        log.info(
            "evaluate: pipeline id=%s len=%d crit_path=%d depth=%d parallelism=%.3f"
            " unused=%d redundant=%d score=%.3f",
            pipeline.id, n, critical_path_length, dependency_depth, parallelism_ratio,
            len(unused), len(redundant), score,
        )

        return StructuralScores(
            pipeline_length=n,
            critical_path_length=critical_path_length,
            parallelism_ratio=round(parallelism_ratio, 6),
            redundant_agents=redundant,
            unused_outputs=unused,
            dependency_depth=dependency_depth,
            score=round(score, 6),
        )


__all__ = ["StructuralEvaluator"]
