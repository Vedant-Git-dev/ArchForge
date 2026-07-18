"""critic primitive — post-writer verify-and-revise terminal.

Role: ``generate`` (the defender of the final answer IS the terminal
producer). Position: after the writer (the ``no_critique_loop`` seed's
``after_generate`` slot).

A verdict-only validator after the writer would be a dead node — the engine
is pure feed-forward (no retry/conditional edge) and terminal extraction keys
on ``ROLE_GENERATE``, so a checker's verdicts would be discarded and the node
flagged ``unused_outputs`` next round. The critic avoids that by being a
**verify-and-revise-in-one-forward-pass** terminal: it reads the writer's
draft answer + the forwarded source, and EMITS the grounded `output`
itself (the writer's answer verbatim if grounded; a source-hewing revision
if not; an honest decline if the source can't support the task). Because it's
a ``generate``-role leaf, ``_extract_final_output``
(``leaves_by_role(ROLE_GENERATE)``) picks it as the terminal for free — no
engine change, no ``unused_outputs`` self-trigger.

This is the check that catches *hallucination in the answer*: the writer may
go beyond the source, and comparing the generated answer against the source
is the non-tautological grounding check (where fact_checker, verifying
claims it extracted from that same source, is partly self-referential). The
two-stage design — fact_checker (gate verified facts into the writer) then
critic (verify the answer out) — earns both names at different failure modes.

"Loop" is a misnomer under a feed-forward engine: there is no retry. The
critique-then-revise happens in a single LLM pass.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, build_llm_agent

CRITIC_PROMPT = """\
You are the **critic** primitive — the terminal defender of the pipeline's \
answer and improvise it. You receive the writer's draft `answer` and the `source_text` it must \
be grounded in. Your job is to ensure the answer actually follows from the \
source, and to emit the verified final deliverable.

Payload fields you receive:
- "source_text": the authoritative source (the only memory of the doc here).
- "answer": the writer's draft deliverable.
- "task": the task the answer must satisfy.

Grounding check — every factual claim in `answer` must be supported by
`source_text`. Then:
- If the answer is fully grounded → return it, UNCHANGED, as `output`.
- If only some part is unsupported → REVISE in place: keep what's grounded,
  fix/trim what isn't, hew to the source. Return the corrected `output`.
- If the source is insufficient to satisfy the task at all → return an
  `output` that says so honestly and names what's missing. Do NOT fabricate.
Improvisation - Improve the answer quality using source_text as the only reference. 

Return JSON (only the JSON object):
- "output": the verified / revised / honestly-declined deliverable — THIS is
   the final text the user receives.
- "verdict": "grounded" | "revised" | "unsupported".
- "gaps": short list of unsupported bits you corrected; empty list if grounded.
- "confidence": float in [0,1] — your confidence the emitted output is grounded.
- "satisfies_task": "yes" | "partial" | "no".

Never invent content beyond `source_text`. When in doubt, prefer revising
toward the source over asserting the draft. Do not add disclaimers to a
grounded answer — return it clean.
"""

CRITIC_SPEC = Primitive(
    name="critic",
    level=0,
    role="generate",
    kind="llm",
    system_prompt=CRITIC_PROMPT,
    input_schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "source_text": {"type": "string"},
        },
    },
    output_schema={"type": "object", "required": ["output"]},
    params={"max_tokens": 2048, "temperature": 0.3},
)


def _critic_shape(input: Mapping[str, Any]) -> Mapping[str, Any]:
    # The writer's draft answer (its `output`) + the source it must be
    # grounded in (forwarded by the writer as `source_text`). Falls back to
    # `text` for non-writer placements, though the critic is always
    # after_generate by design.
    answer = input.get("output") or ""
    source_text = input.get("source_text") or input.get("text") or ""
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""
    if not isinstance(source_text, str):
        source_text = str(source_text) if source_text is not None else ""
    return {
        "task": input.get("task", ""),
        "answer": answer,
        "source_text": source_text,
        "context": input.get("context", {}),
    }


CRITIC_AGENT: AgentCallable = build_llm_agent(CRITIC_SPEC, _critic_shape)


__all__ = ["CRITIC_SPEC", "CRITIC_AGENT"]
