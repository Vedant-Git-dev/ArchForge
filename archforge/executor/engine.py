"""DAG execution engine.

Phase 1: runs nodes in topological order, sequentially. Multi-predecessor
nodes merge their predecessors' outputs into a single dict.

Phase 2 will layer parallelism on top without changing the call site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..core.pipeline import AgentNode, PipelineDAG
from ..core.task import Task
from ..logging import get_logger
from .agents.base import AgentContext, AgentResult
from .agents.registry import PrimitivePool, default_pool
from .llm import LLMClient

log = get_logger("engine")


@dataclass
class NodeTrace:
    """Per-node timing and result for inspection / debugging."""

    node_id: str
    agent_type: str
    started_at: float
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    output: dict[str, Any]
    text: str = ""
    ok: bool = True           # agent-kind-blind soft-fail channel (spec §6)
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class PipelineResult:
    final_output: str  # the user-facing output (writer's `output` field)
    node_results: dict[str, AgentResult] = field(default_factory=dict)
    traces: list[NodeTrace] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


def _build_node_input(
    node: AgentNode,
    pred_outputs: dict[str, dict[str, Any]],
    task: Task,
    outer_input: str,
) -> dict[str, Any]:
    """Compose the JSON payload for a node from its predecessors and task.

    Linear pipeline: previous node's output is the only predecessor, so the
    payload is `{task, input, **prev_output}`. For fan-in nodes, multiple
    predecessors are merged under a `predecessors` key — primitives that
    need direct fields can still pull them from there later.
    """
    base: dict[str, Any] = {
        "task": task.description,
        "task_type": task.type,
        "input": outer_input,
        "context": task.metadata,
    }
    if not pred_outputs:
        return base
    if len(pred_outputs) == 1:
        # Single predecessor — surface its fields directly so primitives like
        # chunker can read .text without indirection.
        (_nid, fields), = pred_outputs.items()
        return {**base, **fields}
    # Multiple predecessors → wrap so the primitive can introspect.
    return {**base, "predecessors": pred_outputs}


def _extract_final_output(
    node_outputs: dict[str, AgentResult],
    pipeline: PipelineDAG,
    resolver,
) -> str:
    """Pick the user-facing output of the pipeline, keyed on `TERMINAL_ROLE`.

    Replaces the hardcoded ``agent_type == "writer"`` terminal detection with
    role-keyed resolution so a pipeline whose generate primitive is named
    anything else (e.g. ``composer``) extracts just as well. Candidate order:

      1. a generate-role LEAF, topological-last among several — the terminal
         stage whose output the user receives;
      2. any generate-role node (a generate stage that is not a leaf still
         produces the deliverable — mirrors the original full-node scan);
      3. the topological-last leaf overall (the original writer→last-leaf
         fallback, generalized to any terminal role).

    Reads the matched node result's ``output`` field (falling back to
    ``text``). `resolver` is duck-typed (exposes ``role_of_node``); the engine
    builds one from its pool at init. Acyclic by contract — `Engine.run`
    refuses a cyclic pipeline before it reaches here.
    """
    from ..config import TERMINAL_ROLE

    order = pipeline.topo_order()
    idx = {n.id: i for i, n in enumerate(order)}

    def extract(node: AgentNode) -> str | None:
        r = node_outputs.get(node.id)
        if r is None:
            return None
        o = r.output
        if o.get("output"):
            return str(o["output"])
        if o.get("text"):
            return str(o["text"])
        return None

    def topo_last(group):
        for n in sorted(group, key=lambda nd: idx[nd.id], reverse=True):
            v = extract(n)
            if v is not None:
                return v
        return None

    v = topo_last(pipeline.leaves_by_role(TERMINAL_ROLE, resolver))
    if v is not None:
        return v
    v = topo_last(pipeline.nodes_by_role(TERMINAL_ROLE, resolver))
    if v is not None:
        return v
    v = topo_last(pipeline.leaves())
    return v if v is not None else ""


class Engine:
    """Run a PipelineDAG.

    Stateless w.r.t. the pipeline — only depends on the LLM client and
    PrimitivePool. Multiple engines can share the same pool/llm.
    """

    def __init__(
        self,
        llm: LLMClient,
        pool: PrimitivePool | None = None,
    ) -> None:
        self.llm = llm
        # The capability carrier threaded into every agent run (spec §2).
        # Sync-only for v1; future slots (tools/scheduler/engine) land here.
        self._ctx = AgentContext(llm=self.llm)
        self.pool = pool or default_pool()
        # Role-keyed terminal-stage resolution (replaces the hardcoded
        # "writer" name check in final-output extraction). Built from the
        # same pool the engine resolves primitives through, so a node is
        # identified as the terminal by its role, not its primitive name.
        self._resolver = self.pool.role_resolver()

    def run(
        self,
        pipeline: PipelineDAG,
        task: Task,
        outer_input: str = "",
    ) -> PipelineResult:
        if pipeline.has_cycle():
            log.error("run: refusing cyclic pipeline id=%s", pipeline.id)
            raise ValueError(f"Pipeline {pipeline.id} has a cycle; refusing to execute")

        order = pipeline.topo_order()
        log.info(
            "run: pipeline id=%s nodes=%d order=%s",
            pipeline.id, len(order), [n.agent_type for n in order],
        )
        wall_start = time.perf_counter()

        node_outputs: dict[str, AgentResult] = {}
        predecessor_outputs: dict[str, dict[str, dict[str, Any]]] = {n.id: {} for n in pipeline.nodes}

        traces: list[NodeTrace] = []
        total_prompt = 0
        total_completion = 0

        for i, node in enumerate(order):
            # Assemble predecessors' outputs for THIS node only.
            preds = pipeline.predecessors(node.id)
            pred_outputs_for_node: dict[str, dict[str, Any]] = {}
            for pred in preds:
                if pred.id in node_outputs:
                    pred_outputs_for_node[pred.id] = node_outputs[pred.id].output

            payload = _build_node_input(node, pred_outputs_for_node, task, outer_input)

            agent = self.pool.get(node.agent_type)
            log.info(
                "run: node %d/%d start id=%s agent=%s preds=%s",
                i + 1, len(order), node.id, node.agent_type, [p.id for p in preds],
            )
            t0 = time.perf_counter()
            result = agent(payload, ctx=self._ctx)
            dt = time.perf_counter() - t0

            node_outputs[node.id] = result
            total_prompt += result.prompt_tokens
            total_completion += result.completion_tokens

            log.info(
                "run: node %d/%d done  id=%s agent=%s %.0fms tokens=%d (p=%d c=%d) model=%s",
                i + 1, len(order), node.id, node.agent_type, dt * 1000, result.total_tokens,
                result.prompt_tokens, result.completion_tokens, result.model,
            )
            log.debug(
                "run: node id=%s output_keys=%s",
                node.id, sorted(result.output.keys()),
            )

            traces.append(NodeTrace(
                node_id=node.id,
                agent_type=node.agent_type,
                started_at=t0,
                duration_seconds=dt,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                output=result.output,
                text=result.text,
                ok=result.ok,
                error=result.error,
            ))

        wall_dt = time.perf_counter() - wall_start
        final_output = _extract_final_output(node_outputs, pipeline, self._resolver)
        log.info(
            "run: pipeline done id=%s wall=%.3fs tokens=%d final_len=%d",
            pipeline.id, wall_dt, total_prompt + total_completion, len(final_output),
        )

        return PipelineResult(
            final_output=final_output,
            node_results=node_outputs,
            traces=traces,
            wall_time_seconds=wall_dt,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
        )


__all__ = ["Engine", "PipelineResult", "NodeTrace"]
