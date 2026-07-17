"""writer primitive – produce the final deliverable from upstream results.

Phase 3 shape: a Primitive spec + a shape fn bound through ``build_llm_agent``;
behavior is identical to the old ``WriterAgent.run()`` (the evidence-derivation
logic and the missing-evidence warning carry verbatim into the shape fn). The
old hardcoded ``max_tokens=2048``/``temperature=0.3`` now live in ``params``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from ...logging import get_logger
from .base import AgentCallable, build_llm_agent

log = get_logger("agent.writer")

WRITER_PROMPT = """\
You are the **writer** primitive — the terminal stage of a multi-agent
pipeline. Your job is to compose the final deliverable that goes back
to the user.

You receive a "context" describing the original task and the outputs of
every prior primitive. Your responsibility is to integrate those into a
single coherent response that satisfies the original task.

Return Proper JSON (with no extra text, strips) with the following fields:
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
response with disclaimers. Answer ONLY from the evidence you were given. If the evidence is insufficient,
acknowledge that and say what additional information would be needed to
fully answer the task. Do not hallucinate or invent information.
"""

WRITER_SPEC = Primitive(
    name="writer",
    level=0,
    role="generate",
    kind="llm",
    system_prompt=WRITER_PROMPT,
    input_schema={"type": "object", "required": ["task", "evidence"]},
    output_schema={"type": "object", "required": ["output"]},
    params={"max_tokens": 2048, "temperature": 0.3},
)


def _writer_shape(input: Mapping[str, Any]) -> Mapping[str, Any]:
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

    return {
        "task": input.get("task", ""),
        "evidence": evidence,
        "context": input.get("context", {}),
    }


WRITER_AGENT: AgentCallable = build_llm_agent(WRITER_SPEC, _writer_shape)


__all__ = ["WRITER_SPEC", "WRITER_AGENT"]
