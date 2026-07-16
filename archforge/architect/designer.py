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
from ..core.roles import RoleResolver
from ..core.task import Task
from ..executor.agents.registry import PrimitivePool, default_pool
from ..executor.embeddings import EmbeddingClient
from ..store.experience_store import ExperienceStore, ScoredHit
from ..config import (
    DEFAULT_PIPELINE_AGENTS,
    DEFAULT_REPLAY_SIMILARITY_THRESHOLD,
    ROLE_GENERATE,
    ROLE_VALIDATE,
)
from .interventions import Intervention, InterventionLibrary, is_structurally_eligible
from ..logging import get_logger

log = get_logger("architect")


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

        resolver = RoleResolver.from_pool(pool)
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
                new_p, applied = self._dispatch(iv, diag, p, pool, resolver)
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
        resolver: RoleResolver,
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
            return self._dispatch_delete(iv, diag, pipeline, resolver)
        if mt == "insert":
            return self._dispatch_insert(iv, diag, pipeline, resolver)
        if mt == "parallelize":
            return self._dispatch_parallelize(iv, diag, pipeline)
        if mt == "merge":
            return self._dispatch_merge(iv, diag, pipeline, resolver)
        if mt == "swap":
            return self._dispatch_swap(iv, diag, pipeline, resolver)
        return None, False

    # ----- per-mutation resolvers -----

    @staticmethod
    def _keeps_generator(
        before: PipelineDAG, after: PipelineDAG, resolver: RoleResolver,
    ) -> bool:
        """Generator-removal invariant (Task 5): a mutation must NOT eliminate
        the last generate-role node. The optimizer must not reward-hack by
        dropping the generator to cut cost. True iff `after` still has a
        generate-role node, OR `before` had none to begin with (a non-generator
        mutation on a generator-less pipeline is fine — there was nothing to
        protect). Role-keyed via the resolver so the guard holds across
        arbitrary primitive vocabularies, keeping the agent-agnostic PipelineDAG
        mutation primitives clean.
        """
        if not before.has_role(ROLE_GENERATE, resolver):
            return True
        return after.has_role(ROLE_GENERATE, resolver)

    @staticmethod
    def _resolve_role_primitive(role: str | None, resolver: RoleResolver) -> str | None:
        """Pick a concrete primitive name of `role` from the pool's resolver,
        deterministically. Sorted by name so the same pool yields the same
        insertion (reproducible). A role-based intervention (Task 6) doesn't
        care WHICH validate primitive a pool offers, only that it offers one;
        this makes the concrete choice. None if the pool has none of `role`.
        """
        if role is None:
            return None
        names = sorted(n for n, r in resolver.as_dict().items() if r == role)
        return names[0] if names else None

    def _dispatch_delete(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
        resolver: RoleResolver,
    ) -> tuple[PipelineDAG | None, bool]:
        # `diagnosis_targets` resolves to the diagnosis's own target_nodes.
        # A delete with no pinned targets is ungrounded → skip (don't guess).
        targets = list(diag.target_nodes)
        if not targets:
            log.debug("dispatch delete %s: no target_nodes → skip", iv.id)
            return None, False
        # Generator invariant (Task 5): per-target. Try the delete; if it would
        # eliminate the last generate-role node, skip THAT target while still
        # removing the rest of a multi-target delete (legitimate non-generator
        # deletions are preserved). Re-checked against the live (post-prior-
        # deletes) pipeline each iteration, so a target set that would TOGETHER
        # remove every generator has its last one retained automatically —
        # order-independent.
        any_deleted = False
        p = pipeline
        for t in targets:
            if p.node_by_id(t) is None:
                continue
            cand = p.delete_node(t)
            if not self._keeps_generator(p, cand, resolver):
                log.debug(
                    "dispatch delete %s: target %s is the only generator → skip",
                    iv.id, t)
                continue
            p = cand
            any_deleted = True
        return (p, True) if any_deleted else (None, False)

    def _dispatch_insert(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
        resolver: RoleResolver,
    ) -> tuple[PipelineDAG | None, bool]:
        slot = iv.target_slot  # before_generate | after_generate
        # Resolve the concrete primitive to insert. A role-based intervention
        # (Task 6) carries NO hardcoded name — it names the ROLE and the
        # Architect picks whichever primitive of that role the active pool
        # offers, so the same intervention works across arbitrary vocabularies
        # (no_validator → the pool's validate primitive, whatever it's called).
        # A concrete-name intervention (e.g. the future critic seed) inserts
        # as-is.
        insert_name = iv.agent_to_insert
        if insert_name is None:
            insert_name = self._resolve_role_primitive(iv.agent_role, resolver)
            if insert_name is None:
                log.debug(
                    "dispatch insert %s: no %s-role primitive in pool → skip",
                    iv.id, iv.agent_role)
                return None, False
            insert_role = iv.agent_role
        else:
            insert_role = resolver.role_of(insert_name)
        # Idempotency: don't insert a validate-role node when one's present.
        # Keyed on ROLE — derived from `agent_role` for the role-based seed or
        # from the concrete name's resolved role — so a `fact_checker` OR a
        # pool's differently-named validate primitive both gate. Only
        # ROLE_VALIDATE triggers the skip: a critic seed (analyze role) still
        # inserts even though an analyze node is already present (having an
        # analyzer doesn't preclude a critique loop).
        if insert_role == ROLE_VALIDATE and pipeline.has_role(ROLE_VALIDATE, resolver):
            log.debug(
                "dispatch insert %s: a validate-role node already present → skip",
                iv.id)
            return None, False
        gen_nodes = pipeline.nodes_by_role(ROLE_GENERATE, resolver)
        if not gen_nodes:
            log.debug("dispatch insert %s: no generate-role node → skip", iv.id)
            return None, False
        p = pipeline
        for gen in gen_nodes:
            if slot == "before_generate":
                p = p.insert_before(gen.id, insert_name)
            else:  # after_generate
                p = p.insert_after(gen.id, insert_name)
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
        resolver: RoleResolver,
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
        # Generator invariant (Task 5): a merge collapses the chain to the
        # first node's agent_type. If the only generate node was inside the
        # chain and the first node isn't a generator, the merge would remove
        # the last generator → refuse the WHOLE merge (the chain is the unit;
        # a partial merge isn't a simple-chain collapse). No live deep_chain
        # seed pins the generator, but the invariant holds uniformly.
        if not self._keeps_generator(pipeline, p, resolver):
            log.debug("dispatch merge %s: would remove the only generator → skip", iv.id)
            return None, False
        return p, True

    def _dispatch_swap(
        self, iv: Intervention, diag: Diagnosis, pipeline: PipelineDAG,
        resolver: RoleResolver,
    ) -> tuple[PipelineDAG | None, bool]:
        # No swap seed ships today (over_chunking's "swap chunker for a variant"
        # would be one, once a distinct variant exists). Implemented for
        # completeness: swap each pinned target to iv.agent_to_insert in place.
        # swap resolves a concrete name (agent_to_insert) — a role-based swap is
        # a future generalization; a swap carrying agent_role instead no-ops.
        targets = list(diag.target_nodes)
        if not targets or not iv.agent_to_insert:
            return None, False
        any_swapped = False
        p = pipeline
        for t in targets:
            if p.node_by_id(t) is None:
                continue
            cand = p.replace_node(t, iv.agent_to_insert)
            # Generator invariant (Task 5): replacing the only generator's
            # agent_type with a non-generator would eliminate the last
            # generator → skip THAT target (per-target, like delete).
            if not self._keeps_generator(p, cand, resolver):
                log.debug(
                    "dispatch swap %s: target %s is the only generator → skip",
                    iv.id, t)
                continue
            p = cand
            any_swapped = True
        return (p, True) if any_swapped else (None, False)

    # ----- helper -----

    @staticmethod
    def _task_text(task: Task) -> str:
        # Combine type with description so two tasks with the same wording
        # but different intents can hash to different embeddings.
        return f"{task.type}: {task.description}".strip()


__all__ = ["Architect", "ArchitectureDecision", "DEFAULT_PIPELINE_AGENTS"]
