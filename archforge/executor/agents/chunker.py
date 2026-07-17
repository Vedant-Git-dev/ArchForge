"""chunker primitive – split long input into manageable pieces.

Phase 3 shape: a Primitive spec + a shape fn bound through ``build_llm_agent``;
behavior is identical to the old ``ChunkerAgent.run()``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, build_llm_agent

CHUNKER_PROMPT = """\
You are the **chunker** primitive in a multi-agent pipeline.

Your job is to decide how to split a long input into smaller, semantically
coherent pieces for downstream processing. You do not analyse or summarise
the content — you only segment it.

Choose a chunking strategy based on input shape:
- Prose / essay → paragraph or section boundaries.
- Code / config → function, class, or block boundaries.
- Tabular / list → row or item boundaries.
- Mixed → prefer the most local semantic boundary, never split mid-sentence
  when prose is present.
- Very short input → return a single chunk containing the whole input.

Return JSON with:
- "chunks": list of {"id": <int>, "text": <str>, "kind": <str>} entries.
  `kind` is a short label ("paragraph", "function", "row", "block", "single").
- "strategy": one-line explanation of the strategy you picked.
- "warnings": empty list unless input was already too small to chunk or
  contained clear structural problems.

Aim for 3-12 chunks when the input warrants splitting. Order is significant.
"""

CHUNKER_SPEC = Primitive(
    name="chunker",
    level=0,
    role="transform",
    kind="llm",
    system_prompt=CHUNKER_PROMPT,
    input_schema={"type": "object", "required": ["text"]},
    output_schema={"type": "object", "required": ["chunks"]},
    params={},
)


def _chunker_shape(input: Mapping[str, Any]) -> Mapping[str, Any]:
    text = input.get("text") or input.get("raw_text") or ""
    return {"input": text, "context": input.get("context", {})}


CHUNKER_AGENT: AgentCallable = build_llm_agent(CHUNKER_SPEC, _chunker_shape)


__all__ = ["CHUNKER_SPEC", "CHUNKER_AGENT"]
