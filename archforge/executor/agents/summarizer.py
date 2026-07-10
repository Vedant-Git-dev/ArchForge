"""summarizer primitive – distil input into a concise representation."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json


SYSTEM_PROMPT = """\
You are the **summarizer** primitive in a multi-agent pipeline.

Your job is to compress input into a faithful, concise representation.
You do not classify, verify, or generate new content — you only distill.

Pick a summary style that matches the input shape:
- Prose → abstractive summary (paraphrase).
- Structured data → enumerated key fields with values.
- Mixed → per-section bullets, then a one-line takeaway.
- Long input → hierarchical: top-line takeaway, then 3-7 bullets, then
  optional deeper claims if space permits.

Return JSON with:
- "summary": the summary text. Plain prose is usually fine unless the
  input is structured, in which case use a structured form.
- "style": short label of the style you picked ("abstractive", "bulleted",
  "structured", "hierarchical", ...).
- "key_points": optional list of short bullet strings — only include if
  the summary above isn't already structured enough to extract them.
- "length_chars": approximate character count of `summary`.

Truthfulness is the primary constraint: never state anything not supported
by the input. If the input is too sparse or unclear, summarize what is
present and note the gap.
"""


class SummarizerAgent:
    name = "summarizer"
    role = "analyze"

    def __init__(self) -> None:
        self.primitive = Primitive(
            name="summarizer",
            level=0,
            role="analyze",
            system_prompt=SYSTEM_PROMPT,
            input_schema={"type": "object"},
            output_schema={"type": "object", "required": ["summary"]},
        )

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult:
        # Forward only the predecessor's salient output, not the whole merged
        # bag. The engine merges a `base` (task / task_type / raw `input` /
        # context) into every node's input so terminal agents can still see
        # the task — but a summarizer summarises what came IN, so re-sending
        # the original text + task string + context (then again, below) is
        # needless duplication. Strip the base keys; fall back to the whole
        # dict only if the predecessor produced nothing usable.
        base_keys = {"task", "task_type", "input", "context"}
        upstream = {k: v for k, v in input.items() if k not in base_keys}
        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {"input": upstream or input, "context": input.get("context", {})},
            kind=self.name,
        )


__all__ = ["SummarizerAgent"]
