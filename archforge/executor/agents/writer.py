"""writer primitive – produce the final deliverable from upstream results.

Phase 3 shape: a Primitive spec + a custom callable. Behaviour is identical
to the old ``WriterAgent.run()`` evidence-derivation logic (carried verbatim
into the shape), with ONE additive change: it now **forwards ``source_text``
as provenance** in its output so the post-writer ``critic`` (the verify-and-
revise terminal) has the source to check the answer against. Without it, the
critic after the writer would mirror the run-2/3 severance bug (answer in,
but no doc to ground it).

``source_text`` is injected **deterministically** (post-LLM), never by asking
the model to echo the document back — that would be token-wasteful and lossy
(the model paraphrasing the source is worse than forwarding the bytes). The
old hardcoded ``max_tokens=2048``/``temperature=0.3`` live in ``params``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from ...logging import get_logger
from .base import AgentCallable, AgentResult, call_llm_json

log = get_logger("agent.writer")

WRITER_PROMPT = """\
You are the **writer** primitive — the terminal stage of a multi-agent
pipeline. Your job is to compose the final deliverable that goes back
to the user.

You receive a "context" describing the original task and the outputs of
every prior primitive (including any ``verified_facts`` from the fact_checker,
which you SHOULD compose the answer from — those are the vetted facts).
Your responsibility is to integrate those into a single coherent response
that satisfies the original task.

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
response with disclaimers. Answer ONLY from the evidence you were given. If
the evidence is insufficient, acknowledge that and say what additional
information would be needed to fully answer the task. Do not hallucinate or
invent information.

Do NOT echo back any source document you were given — provenance is forwarded
separately by the runtime. Compose from the upstream evidence fields only.
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


def _writer_evidence(input: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return (evidence_for_llm, source_text) for the call.

    Evidence: integrate the predecessor's full output, not a single
    `evidence` key (no upstream agent emits one — without this, the
    writer composes the final answer from {}). Strip the engine's base
    fields so we forward upstream *results*, not a second copy of the
    task metadata. Fall back to the original source text only when
    upstream produced nothing usable.

    `source_text` is surfaced (separate from evidence) so the callable can
    forward it as provenance for the post-writer critic.
    """
    base_keys = {"task", "task_type", "input", "context", "text", "source_text"}
    evidence = input.get("evidence")
    if not evidence:
        evidence = {k: v for k, v in input.items() if k not in base_keys}
        if not evidence and input.get("input"):
            evidence = {"source_text": input["input"]}

    source_text = input.get("source_text") or input.get("text") or ""
    if not isinstance(source_text, str):
        source_text = str(source_text) if source_text is not None else ""

    return evidence, source_text


def writer(input: Mapping[str, Any], ctx=None) -> AgentResult:
    """Compose the final deliverable; forward `source_text` as provenance."""
    if ctx is None or getattr(ctx, "llm", None) is None:
        raise RuntimeError(
            f"llm-kind agent 'writer' needs ctx.llm; got ctx={ctx!r}"
        )
    evidence, source_text = _writer_evidence(input)
    if not evidence:
        log.warning(
            "writer: no upstream evidence (input_keys=%s); expected the "
            "predecessor's output to supply the material to compose from",
            list(input.keys()),
        )
    payload = {
        "task": input.get("task", ""),
        "evidence": evidence,
        "context": input.get("context", {}),
    }
    res = call_llm_json(
        ctx.llm, WRITER_PROMPT, payload,
        kind="writer", max_tokens=2048, temperature=0.3,
    )
    # Forward the source as provenance for the post-writer critic — injected
    # deterministically, not via the LLM (authoritative bytes, no echo/paraphrase).
    res.output["source_text"] = source_text
    return res


WRITER_AGENT: AgentCallable = writer


__all__ = ["WRITER_SPEC", "WRITER_AGENT", "writer"]
