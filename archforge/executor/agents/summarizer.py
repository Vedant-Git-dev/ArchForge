"""summarizer primitive – distil input into a concise representation.

Phase 3 shape: a Primitive spec + a shape fn bound through ``build_llm_agent``;
behavior is identical to the old ``SummarizerAgent.run()``.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, build_llm_agent

SUMMARIZER_PROMPT = """\
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

SUMMARIZER_SPEC = Primitive(
    name="summarizer",
    level=0,
    role="analyze",
    kind="llm",
    system_prompt=SUMMARIZER_PROMPT,
    input_schema={"type": "object"},
    output_schema={"type": "object", "required": ["summary"]},
    params={},
)


def _summarizer_shape(input: Mapping[str, Any]) -> Mapping[str, Any]:
    # Forward only the predecessor's salient output, not the whole merged
    # bag. The engine merges a `base` (task / task_type / raw `input` /
    # context) into every node's input so terminal agents can still see
    # the task — but a summarizer summarises what came IN, so re-sending
    # the original text + task string + context (then again, below) is
    # needless duplication. Strip the base keys; fall back to the whole
    # dict only if the predecessor produced nothing usable.
    base_keys = {"task", "task_type", "input", "context"}
    upstream = {k: v for k, v in input.items() if k not in base_keys}
    return {"input": upstream or input, "context": input.get("context", {})}


SUMMARIZER_AGENT: AgentCallable = build_llm_agent(SUMMARIZER_SPEC, _summarizer_shape)


__all__ = ["SUMMARIZER_SPEC", "SUMMARIZER_AGENT"]
