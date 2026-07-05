"""classifier primitive – label items with relevant categories."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json


SYSTEM_PROMPT = """\
You are the **classifier** primitive in a multi-agent pipeline.

Your job is to assign labels to each input item. You do not summarise,
verify, or generate content. You classify.

Decide category set dynamically from the input — do not assume a fixed
taxonomy. Common axes:
- For text: topic, genre, intent, sentiment, complexity.
- For tasks: type ("analysis", "generation", "extraction", "review", ...).
- For code: language, purpose, risk level.

Return JSON with:
- "categories": list of {"item_index": <int>, "labels": [{"name": <str>,
   "confidence": <float 0-1>}], "rationale": <short str>}.
  One entry per item classified.
- "summary": a one-line note explaining the taxonomy you chose and why.
- "ambiguous_items": indices of any items you couldn't confidently classify.
  Empty list if all are clear.

Be specific over generic. Avoid single-word labels when a phrase is clearer.
"""


class ClassifierAgent:
    name = "classifier"
    role = "analyze"

    def __init__(self) -> None:
        self.primitive = Primitive(
            name="classifier",
            level=0,
            role="analyze",
            system_prompt=SYSTEM_PROMPT,
            input_schema={"type": "object", "required": ["items"]},
            output_schema={"type": "object", "required": ["categories"]},
        )

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult:
        items = input.get("items") or input.get("chunks") or [input.get("text", "")]
        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {"items": items, "context": input.get("context", {})},
            kind=self.name,
        )


__all__ = ["ClassifierAgent"]
