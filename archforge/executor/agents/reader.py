"""reader primitive – ingest arbitrary input and normalise into structured form.

Phase 3 shape: a Primitive spec + a shape fn bound through ``build_llm_agent``.
Behavior is identical to the old ``ReaderAgent.run()`` — the spec holds the
prompt/schemas, the shape fn carries the (verbatim) re-keying, and the factory
threads ``ctx.llm`` through ``call_llm_json``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, build_llm_agent

READER_PROMPT = """\
You are the **reader** primitive in a multi-agent pipeline.

Your job is to take raw input and produce a structured, normalised
representation the rest of the pipeline can rely on. You do not interpret,
classify, summarise, or judge — your output is the substrate other
primitives operate on.

Return JSON with these fields:
- "text":  the cleaned input as a single string.
- "tokens_estimate": your rough token-count estimate for downstream budget checks.
- "source": a short label for where this input came from (e.g. "user",
   "file:foo.txt", or "inline").
- "notes": optional list of short remarks about the input (e.g. length,
   structure, presence of empty sections). Empty list if nothing notable.

Be terse. Do not add opinions or summaries.
"""

READER_SPEC = Primitive(
    name="reader",
    level=0,
    role="ingest",
    kind="llm",
    system_prompt=READER_PROMPT,
    input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
    output_schema={"type": "object", "required": ["text"]},
    params={},
)


def _reader_shape(input: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"input": input.get("input", ""), "context": input.get("context", {})}


READER_AGENT: AgentCallable = build_llm_agent(READER_SPEC, _reader_shape)


__all__ = ["READER_SPEC", "READER_AGENT"]
