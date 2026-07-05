"""reader primitive – ingest arbitrary input and normalise into structured form."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json


SYSTEM_PROMPT = """\
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


class ReaderAgent:
    name = "reader"
    role = "ingest"

    def __init__(self) -> None:
        self.primitive = Primitive(
            name="reader",
            level=0,
            role="ingest",
            system_prompt=SYSTEM_PROMPT,
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            output_schema={"type": "object", "required": ["text"]},
        )

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult:
        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {"input": input.get("input", ""), "context": input.get("context", {})},
        )


__all__ = ["ReaderAgent"]
