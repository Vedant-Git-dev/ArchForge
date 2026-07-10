"""writer primitive – produce the final deliverable from upstream results."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ...logging import get_logger
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json

log = get_logger("agent.writer")


SYSTEM_PROMPT = """\
You are the **writer** primitive — the terminal stage of a multi-agent
pipeline. Your job is to compose the final deliverable that goes back
to the user.

You receive a "context" describing the original task and the outputs of
every prior primitive. Your responsibility is to integrate those into a
single coherent response that satisfies the original task.

Return JSON with:
- "output": the final text the user will read. This is the response the
  pipeline returns. Make it clear, complete, and free of intermediate
  reasoning.
- "format": the format of `output` ("prose", "bullets", "code", "table",
  "mixed").
- "satisfies_task": your judgement ("yes" | "partial" | "no") on whether
  the upstream evidence was sufficient to fully satisfy the task.
- "open_questions": list of follow-up questions the task left unanswered.
  Empty list if the output is complete.

Do not hedge the output itself — the user wants a real answer. Surface
uncertainty by saying so plainly when it's justified, but don't fill the
response with disclaimers.
"""


class WriterAgent:
    name = "writer"
    role = "generate"

    def __init__(self) -> None:
        self.primitive = Primitive(
            name="writer",
            level=0,
            role="generate",
            system_prompt=SYSTEM_PROMPT,
            input_schema={"type": "object", "required": ["task", "evidence"]},
            output_schema={"type": "object", "required": ["output"]},
        )

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult:
        # Evidence: integrate the predecessor's full output, not a single
        # `evidence` key (no upstream agent emits one — without this, the
        # writer composes the final answer from {}). Strip the engine's base
        # fields so we forward upstream *results*, not a second copy of the
        # task metadata. Fall back to the original source text only when
        # upstream produced nothing usable.
        base_keys = {"task", "task_type", "input", "context"}
        evidence = input.get("evidence")
        if not evidence:
            evidence = {k: v for k, v in input.items() if k not in base_keys}
            if not evidence and input.get("input"):
                evidence = {"source_text": input["input"]}

        if not evidence:
            log.warning(
                "writer: no upstream evidence (input_keys=%s); expected the "
                "predecessor's output to supply the material to compose from",
                list(input.keys()),
            )

        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {
                "task": input.get("task", ""),
                "evidence": evidence,
                "context": input.get("context", {}),
            },
            # Writer tends to run long — give it room.
            kind=self.name,
            max_tokens=2048,
            temperature=0.3,
        )


__all__ = ["WriterAgent"]
