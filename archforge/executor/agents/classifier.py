"""classifier primitive – label items with relevant categories.

Phase 3 shape: a Primitive spec + a shape fn bound through ``build_llm_agent``;
behavior is identical to the old ``ClassifierAgent.run()``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, build_llm_agent

CLASSIFIER_PROMPT = """\
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

CLASSIFIER_SPEC = Primitive(
    name="classifier",
    level=0,
    role="analyze",
    kind="llm",
    system_prompt=CLASSIFIER_PROMPT,
    input_schema={"type": "object", "required": ["items"]},
    output_schema={"type": "object", "required": ["categories"]},
    params={},
)


def _classifier_shape(input: Mapping[str, Any]) -> Mapping[str, Any]:
    items = input.get("items") or input.get("chunks") or [input.get("text", "")]
    return {"items": items, "context": input.get("context", {})}


CLASSIFIER_AGENT: AgentCallable = build_llm_agent(CLASSIFIER_SPEC, _classifier_shape)


__all__ = ["CLASSIFIER_SPEC", "CLASSIFIER_AGENT"]
