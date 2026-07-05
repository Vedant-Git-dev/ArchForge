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
from .agents.base import AgentResult
from .agents.registry import PrimitivePool, default_pool
from .llm import LLMClient


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


def _extract_final_output(node_outputs: dict[str, AgentResult], pipeline: PipelineDAG) -> str:
    """Pick the user-facing output of the pipeline.

    Convention: the writer-node's `output` field. If the pipeline has none,
    fall back to a JSON dump of the last leaf's full output.
    """
    for node in pipeline.nodes:
        if node.agent_type == "writer" and node.id in node_outputs:
            writer_out = node_outputs[node.id].output
            # Writer system prompt guarantees an `output` string field.
            return str(writer_out.get("output", writer_out.get("text", "")))
    # Fallback: serialise the last leaf.
    leaves = pipeline.leaves()
    if leaves:
        last_result = node_outputs.get(leaves[-1].id)
        if last_result is not None:
            if last_result.output.get("output"):
                return str(last_result.output["output"])
            if last_result.output.get("text"):
                return str(last_result.output["text"])
    return ""


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
        self.pool = pool or default_pool()

    def run(
        self,
        pipeline: PipelineDAG,
        task: Task,
        outer_input: str = "",
    ) -> PipelineResult:
        if pipeline.has_cycle():
            raise ValueError(f"Pipeline {pipeline.id} has a cycle; refusing to execute")

        order = pipeline.topo_order()
        wall_start = time.perf_counter()

        node_outputs: dict[str, AgentResult] = {}
        predecessor_outputs: dict[str, dict[str, dict[str, Any]]] = {n.id: {} for n in pipeline.nodes}

        traces: list[NodeTrace] = []
        total_prompt = 0
        total_completion = 0

        for node in order:
            # Assemble predecessors' outputs for THIS node only.
            preds = pipeline.predecessors(node.id)
            pred_outputs_for_node: dict[str, dict[str, Any]] = {}
            for pred in preds:
                if pred.id in node_outputs:
                    pred_outputs_for_node[pred.id] = node_outputs[pred.id].output

            payload = _build_node_input(node, pred_outputs_for_node, task, outer_input)

            agent = self.pool.get(node.agent_type)
            t0 = time.perf_counter()
            result = agent.run(payload, self.llm)
            dt = time.perf_counter() - t0

            node_outputs[node.id] = result
            total_prompt += result.prompt_tokens
            total_completion += result.completion_tokens

            traces.append(NodeTrace(
                node_id=node.id,
                agent_type=node.agent_type,
                started_at=t0,
                duration_seconds=dt,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                output=result.output,
                text=result.text,
            ))

        wall_dt = time.perf_counter() - wall_start
        final_output = _extract_final_output(node_outputs, pipeline)

        return PipelineResult(
            final_output=final_output,
            node_results=node_outputs,
            traces=traces,
            wall_time_seconds=wall_dt,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
        )


__all__ = ["Engine", "PipelineResult", "NodeTrace"]
