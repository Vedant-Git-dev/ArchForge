"""Architect — Phase 2: retrieve-and-replay + diagnosed interventions.

The flow per plan:
    1. Embed the task.
    2. kNN search the experience index (task similarity).
    3. If a similar past experience exists with a usable pipeline,
       replay its DAG — THEN read that experience's stored diagnoses and
       apply the matched interventions (Phase 2.3). Diagnoses come from the
       *retrieved* experience (the current task hasn't run yet — evaluation
       is post-execution). This is the plan's "reads diagnoses from past
       low-scoring runs → matched interventions" loop.
    4. Else, build the default linear pipeline from the 6 base primitives.

Mutation dispatch (Phase 2.3): each diagnosis → ``InterventionLibrary.match_by_root``
→ best seed by success_rate → resolve ``target_slot`` to concrete node ids
against the live pipeline → call a Phase 2.1 ``PipelineDAG`` primitive. Edits
apply in a fixed order — deletes → inserts → reshape — re-checking eligibility
against the mutated pipeline between steps, so a delete can't move an insert's
target and an insert is reflected in the reshape's critical path. Idempotency
gates re-application: an insert whose role is already present is skipped, and a
delete target that another mutation already removed is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.experience import Diagnosis, Experience
from ..core.pipeline import PipelineDAG
from ..core.task import Task
from ..executor.agents.registry import PrimitivePool, default_pool
from ..executor.embeddings import EmbeddingClient
from ..store.experience_store import ExperienceStore, ScoredHit
from ..config import (
    DEFAULT_PIPELINE_AGENTS,
    DEFAULT_REPLAY_SIMILARITY_THRESHOLD,
)
from .interventions import Intervention, InterventionLibrary, is_structurally_eligible
from ..logging import get_logger

log = get_logger("architect")


def _role_map(pool: PrimitivePool) -> dict[str, str]:
    return {name: p.role for name, p in pool.primitives().items()}


@dataclass
class ArchitectureDecision:
    """What the Architect returns to the caller.

    Includes provenance (`triggered_from`, `matched_experience`) so the
    CLI and tests can assert that retrieval actually happened, plus the
    intervention outcome (`mutated`, `interventions_applied`) so a second
    similar run can be observed to apply fixes.
    """

    pipeline: PipelineDAG
    triggered_from: str  # "retrieval" | "default"
    matched_experience_id: str | None
    matched_pipeline_score: float | None  # composite score of the replayed pipeline
    candidates: list[ScoredHit] = field(default_factory=list)
    task_embedding_dim: int = 0
    mutated: bool = False  # Phase 2.3: did ≥1 intervention mutate the pipeline?
    interventions_applied: list[str] = field(default_factory=list)  # intervention ids


class Architect:
    """Decide which pipeline to run for a task.

    Phase 1: retrieve highest-composite-score experience above threshold.
    Phase 2.3: then apply diagnosed interventions to the replayed pipeline.
    """

    def __init__(
        self,
        store: ExperienceStore,
        embedder: EmbeddingClient,
        *,
        library: InterventionLibrary | None = None,
        pool: PrimitivePool | None = None,
        replay_similarity_threshold: float = DEFAULT_REPLAY_SIMILARITY_THRESHOLD,
        top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.library = library if library is not None else InterventionLibrary()
        self.pool = pool if pool is not None else default_pool()
        self.threshold = replay_similarity_threshold
        self.top_k = top_k

    # ----- entry point -----

    def compose(self, task: Task) -> ArchitectureDecision:
        """Return the PipelineDAG the executor should run for `task`."""
        log.info("compose start: task id=%s type=%r", task.id, task.type)

        # Embed the task. Description + type provides a stable signal even
        # for tasks with empty metadata.
        text = self._task_text(task)
        log.debug("compose: embedding task text (len=%d)", len(text))
        embedding = self.embedder.embed_one(text)
        task.embedding = np.asarray(embedding, dtype=np.float32).reshape(-1).tolist()
        log.debug("compose: task embedded (dim=%d)", len(task.embedding))

        # Make sure any prior experiences without embeddings are filled in
        # before we run the kNN search.
        n_recomputed = self.store.recompute_embeddings(self.embedder)
        if n_recomputed:
            log.info("compose: recomputed %d missing task embeddings", n_recomputed)

        # Search by task similarity.
        log.info("compose: kNN search (top_k=%d threshold=%.2f)", self.top_k, self.threshold)
        hits: list[ScoredHit] = self.store.search_by_task(
            np.asarray(embedding, dtype=np.float32).reshape(-1), k=self.top_k
        )
        if hits:
            top = hits[0]
            log.debug(
                "compose: nearest neighbour id=%s cosine=%.3f rank=%d",
                top.experience.id, top.score, top.rank,
            )
        else:
            log.debug("compose: no neighbours in index (store empty)")

        # Pick the best-scoring experience whose similarity clears the bar.
        replayable = [h for h in hits if h.score >= self.threshold and h.experience.pipeline.nodes]
        if replayable:
            best = max(replayable, key=lambda h: h.experience.composite_score)
            pipeline = PipelineDAG.from_dict(best.experience.pipeline.to_dict())
            pipeline.fingerprint = pipeline.compute_fingerprint()
            log.info(
                "compose: retrieval HIT — replaying experience id=%s cosine=%.3f prior_score=%.3f",
                best.experience.id, best.score, best.experience.composite_score,
            )

            # Phase 2.3: apply diagnosed interventions to the replayed pipeline.
            # The diagnoses are the *matched experience's* (the current task
            # hasn't been evaluated yet). Empty diagnoses (clean past run, or
            # a judge parse-failure that produced none) → no mutation.
            pipeline, applied = self._apply_interventions(
                pipeline, best.experience.diagnoses, self.pool
            )
            if applied:
                log.info(
                    "compose: applied %d intervention(s): %s → nodes=%s",
                    len(applied), applied, [n.agent_type for n in pipeline.nodes],
                )

            return ArchitectureDecision(
                pipeline=pipeline,
                triggered_from="retrieval",
                matched_experience_id=best.experience.id,
                matched_pipeline_score=best.experience.composite_score,
                candidates=hits,
                task_embedding_dim=task.embedding.__len__(),
                mutated=bool(applied),
                interventions_applied=applied,
            )

        # No usable hit — build the default pipeline. No diagnoses to act on.
        pipeline = PipelineDAG.linear(DEFAULT_PIPELINE_AGENTS)
        log.info(
            "compose: retrieval MISS (candidates=%d, replayable=0) — default pipeline %s",
            len(hits), [n.agent_type for n in pipeline.nodes],
        )
        return ArchitectureDecision(
            pipeline=pipeline,
            triggered_from="default",
            matched_experience_id=None,
            matched_pipeline_score=None,
            candidates=hits,
            task_embedding_dim=task.embedding.__len__(),
            mutated=False,
            interventions_applied=[],
        )

    # ----- Phase 2.3: diagnose → match → mutate -----

    def _apply_interventions(
        self,
        pipeline: PipelineDAG,
        diagnoses: list[Diagnosis],
        pool: PrimitivePool,
    ) -> tuple[PipelineDAG, list[str]]:
        """Match each diagnosis to a seeded intervention and apply its mutation.

        Returns the (possibly mutated) pipeline and the ids of interventions
        that actually mutated it. One intervention per diagnosis root (the
        best by success_rate among the seeds); the fixed apply order is
        deletes → inserts → reshape, re-validating against the live pipeline
        at every step.
        """
        if not diagnoses:
            return pipeline, []

        # Match: one diagnosis → one best intervention (by success_rate, then id).
        matched: list[tuple[Diagnosis, Intervention]] = []
        for diag in diagnoses:
            cands = self.library.match_by_root(diag.structural_root)
            if not cands:
                continue  # no seed for this root (e.g. unknown:... escape)
            best = max(cands, key=lambda iv: (iv.success_rate, iv.id))
            matched.append((diag, best))
        if not matched:
            return pipeline, []

        roles = _role_map(pool)
        # Apply order: deletes first (so an insert's before_generate target
        # hasn't moved), inserts next (added to the critical path before the
        # reshape measures it), reshape last (recomputes critical path on the
        # post-edit pipeline). is_structurally_eligible is re-run inside
        # _dispatch on the current pipeline each time.
        DELETE = {"delete"}
        INSERT = {"insert"}
        RESHAPE = {"parallelize", "merge", "swap"}

        def group(types: set[str]) -> list[tuple[Diagnosis, Intervention]]:
            return [(d, iv) for d, iv in matched if iv.mutation_type in types]

        applied_ids: list[str] = []
        p = pipeline
        for phase_types in (DELETE, INSERT, RESHAPE):
            for diag, iv in group(phase_types):
                if iv.id in applied_ids:
                    continue  # one application per intervention per run
                new_p, applied = self._dispatch(iv, diag, p, pool, roles)
                if applied:
                    applied_ids.append(iv.id)
                    p = new_p  # _dispatch guarantees non-None when applied
        return p, applied_ids

    def _dispatch(
        self,
        iv: Intervention,
        diag: Diagnosis,
        pipeline: PipelineDAG,
        pool: PrimitivePool,
        roles: dict[str, str],
    ) -> tuple[PipelineDAG | None, bool]:
        """Resolve `iv`'s slot against (pipeline, diag) and apply the mutation.

        Returns (new_pipeline_or_None, applied). `applied=True` ⇒ a mutation
        happened and `new_pipeline` is the resulting DAG; `applied=False` ⇒
        the intervention was skipped (ineligible, idempotent, or ungrounded)
        and the pipeline is unchanged (`new_pipeline=None`).
        """
        if pipeline.has_cycle():
            return None, False
        if not is_structurally_eligible(iv, pipeline, pool):
            return None, False
        mt = iv.mutation_type
        if mt == "delete":
            return self._dispatch_delete(iv, diag, pipeline)
        if mt == "insert":
            return self._dispatch_insert(iv, diag, pipeline, roles)
        if mt == "parallelize":
            return self._dispatch_parallelize(iv, diag, pipeline)
        if mt == "merge":
            return self._dispatch_merge(iv, diag, pipeline)
        if mt == "swap":
            return self._dispatch_swap(iv, diag, pipeline)
        return None, False

    # ----- per-mutation resolvers -----

    def _dispatch_delete(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
    ) -> tuple[PipelineDAG | None, bool]:
        # `diagnosis_targets` resolves to the diagnosis's own target_nodes.
        # A delete with no pinned targets is ungrounded → skip (don't guess).
        targets = list(diag.target_nodes)
        if not targets:
            log.debug("dispatch delete %s: no target_nodes → skip", iv.id)
            return None, False
        any_deleted = False
        p = pipeline
        for t in targets:
            if p.node_by_id(t) is not None:
                p = p.delete_node(t)
                any_deleted = True
        return (p, True) if any_deleted else (None, False)

    def _dispatch_insert(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
        roles: dict[str, str],
    ) -> tuple[PipelineDAG | None, bool]:
        slot = iv.target_slot  # before_generate | after_generate
        # Idempotency: don't insert a validate-role node when one's present.
        # (Backs the no_validator seed; a broader "role-already-present" guard
        # generalizes once more insert seeds land.)
        if iv.agent_to_insert and roles.get(iv.agent_to_insert) == "validate":
            if any(roles.get(n.agent_type) == "validate" for n in pipeline.nodes):
                log.debug(
                    "dispatch insert %s: a validate-role node already present → skip",
                    iv.id)
                return None, False
        gen_nodes = [n for n in pipeline.nodes if roles.get(n.agent_type) == "generate"]
        if not gen_nodes:
            log.debug("dispatch insert %s: no generate-role node → skip", iv.id)
            return None, False
        p = pipeline
        for gen in gen_nodes:
            if slot == "before_generate":
                p = p.insert_before(gen.id, iv.agent_to_insert)
            else:  # after_generate
                p = p.insert_after(gen.id, iv.agent_to_insert)
        return p, True

    def _dispatch_parallelize(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
    ) -> tuple[PipelineDAG | None, bool]:
        # `bottleneck_node` resolves to the middle of the critical path. The
        # sibling takes the bottleneck node's OWN agent_type (the seed sets no
        # agent_to_insert); a dedicated vote/merge aggregator is a Phase 2.5
        # primitive — the engine's multi-predecessor merge handles fan-in
        # structurally today.
        path = pipeline.critical_path()
        if len(path) < 2:
            log.debug("dispatch parallelize %s: critical path too short → skip", iv.id)
            return None, False
        bottleneck = path[len(path) // 2]
        p = pipeline.parallelize(bottleneck.id, bottleneck.agent_type, n=1)
        return p, True

    def _dispatch_merge(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
    ) -> tuple[PipelineDAG | None, bool]:
        # `deep_chain_nodes` resolves to the diagnosis's target_nodes when they
        # form a consecutive chain. The critical-path fallback (collapse the
        # whole chain) is deferred — without compatibility checking it would
        # fuse incompatible agents (reader+writer) into one node, which the
        # Phase 5 fused-prompt merge is meant to handle, not Phase 2.3. So: no
        # pinpointed chain → skip (the diagnosis still informs retrieval/score;
        # the mutation is best-effort).
        targets = [t for t in diag.target_nodes if pipeline.node_by_id(t) is not None]
        if len(targets) < 2:
            log.debug("dispatch merge %s: <2 pinpointed chain nodes → skip", iv.id)
            return None, False
        try:
            p = pipeline.merge_chain(targets)
        except ValueError as e:
            log.debug("dispatch merge %s: %s→ skip", iv.id, e)
            return None, False
        return p, True

    def _dispatch_swap(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
    ) -> tuple[PipelineDAG | None, bool]:
        # No swap seed ships today (over_chunking's "swap chunker for a variant"
        # would be one, once a distinct variant exists). Implemented for
        # completeness: swap each pinned target to iv.agent_to_insert in place.
        targets = list(diag.target_nodes)
        if not targets or not iv.agent_to_insert:
            return None, False
        any_swapped = False
        p = pipeline
        for t in targets:
            if p.node_by_id(t) is not None:
                p = p.replace_node(t, iv.agent_to_insert)
                any_swapped = True
        return (p, True) if any_swapped else (None, False)

    # ----- helper -----

    @staticmethod
    def _task_text(task: Task) -> str:
        # Combine type with description so two tasks with the same wording
        # but different intents can hash to different embeddings.
        return f"{task.type}: {task.description}".strip()


__all__ = ["Architect", "ArchitectureDecision", "DEFAULT_PIPELINE_AGENTS"]
