"""chunker primitive – split long input into manageable pieces."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json


SYSTEM_PROMPT = """\
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


class ChunkerAgent:
    name = "chunker"
    role = "transform"

    def __init__(self) -> None:
        self.primitive = Primitive(
            name="chunker",
            level=0,
            role="transform",
            system_prompt=SYSTEM_PROMPT,
            input_schema={"type": "object", "required": ["text"]},
            output_schema={"type": "object", "required": ["chunks"]},
        )

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult:
        text = input.get("text") or input.get("raw_text") or ""
        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {"input": text, "context": input.get("context", {})},
            kind=self.name,
        )


__all__ = ["ChunkerAgent"]
