"""chunker primitive — deterministic text segmentation.

The chunker was an LLM primitive whose whole job was "split a long input
into smaller, semantically coherent pieces" — splitting on boundaries never
needed a model. It is now ``kind="deterministic"``: a pure-stdlib transform
that segments text by paragraph, then by sentence groups, falling back to a
single chunk for short input.

Output keeps the old contract (``{"chunks", "strategy", "warnings"}``) so
downstream agents (``classifier``/``summarizer`` read ``chunks`` unchanged)
need no edits. Deterministic → zero-token ``AgentResult``, never raises
(empty/garbage input → a single warning chunk, not an exception).
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, AgentResult

# Split thresholds (chars). Tuned to the old "aim for 3-12 chunks" guidance.
_MIN_CHUNK_INPUT = 200      # below this → one single chunk (not worth splitting)
_LONG_INPUT = 1500          # a single paragraph this long → split by sentence groups
_TARGET_CHUNK = 1000        # sentence-group target size (never splits mid-sentence)
_MAX_CHUNKS = 60            # pathological-input guardrail (a warning if hit)

# Blank-line paragraph boundary: two+ newlines, optionally whitespace between.
_PARA_SPLIT = re.compile(r"\n\s*\n+")
# Sentence boundary: . ! or ? followed by whitespace. Lookbehind keeps the
# terminator attached to the preceding sentence.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

CHUNKER_SPEC = Primitive(
    name="chunker",
    level=0,
    role="transform",
    kind="deterministic",
    system_prompt="",  # inert for the deterministic kind
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "raw_text": {"type": "string"},
        },
    },
    output_schema={"type": "object", "required": ["chunks"]},
    params={},
)


def _chunk_paragraphs(text: str) -> list[str]:
    """Split on blank lines; drop empties. Returns 1+ paragraph strings."""
    return [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]


def _chunk_sentence_groups(text: str) -> list[str]:
    """Sentence-aware splitting: group sentences up to ``_TARGET_CHUNK`` chars,
    never splitting mid-sentence. Preserves order."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        # No sentence terminators found — fall back to fixed-size char windows.
        return [text[i : i + _TARGET_CHUNK] for i in range(0, len(text), _TARGET_CHUNK)] or [text]
    groups: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > _TARGET_CHUNK:
            groups.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip() if buf else sent
    if buf:
        groups.append(buf)
    return groups


def chunker(input: Mapping[str, Any], ctx=None) -> AgentResult:
    """Deterministic chunker.

    ``ctx`` is the ``AgentContext``; this agent ignores ``ctx.llm`` (the
    zero-token contract). Never raises — empty/garbage input yields a single
    warning chunk, not an exception (soft-fail channel, spec §6).
    """
    t0 = time.perf_counter()
    text = input.get("text") or input.get("raw_text") or input.get("input") or ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:  # noqa: BLE001 — never raise from a deterministic agent
            text = ""

    warnings: list[str] = []

    # Empty / whitespace-only input → single empty chunk + a warning (never raises).
    if not text.strip():
        warnings.append("empty input; returned a single empty chunk")
        out: dict[str, Any] = {
            "chunks": [{"id": 1, "text": "", "kind": "single"}],
            "strategy": "single chunk (empty input)",
            "warnings": warnings,
        }
        return AgentResult(
            output=out, text="", prompt_tokens=0, completion_tokens=0, model="",
            latency_ms=(time.perf_counter() - t0) * 1000.0, cost=0.0, ok=True, error=None,
        )

    # Strategy priority: paragraphs → sentence groups → single.
    paragraphs = _chunk_paragraphs(text)
    if len(paragraphs) > 1:
        chunks = paragraphs
        kind = "paragraph"
        strategy = f"paragraph split, {len(chunks)} chunks"
    elif len(text) > _LONG_INPUT:
        chunks = _chunk_sentence_groups(text)
        kind = "sentence_group"
        strategy = f"sentence-group split, {len(chunks)} chunks (target {_TARGET_CHUNK} chars)"
    elif len(text) < _MIN_CHUNK_INPUT:
        chunks = [text]
        kind = "single"
        strategy = "single chunk (input below split threshold)"
    else:
        # One paragraph, medium length — return it whole.
        chunks = paragraphs or [text]
        kind = "single"
        strategy = "single chunk (one paragraph, below long-split threshold)"

    # Pathological-input guardrail: cap the count, preserve order, warn.
    if len(chunks) > _MAX_CHUNKS:
        chunks = chunks[:_MAX_CHUNKS]
        warnings.append(f"chunk count capped at {_MAX_CHUNKS}")

    chunk_objs = [
        {"id": i + 1, "text": c, "kind": kind}
        for i, c in enumerate(chunks)
    ]

    out = {
        "chunks": chunk_objs,
        "strategy": strategy,
        "warnings": warnings,
    }

    return AgentResult(
        output=out,
        text="",
        prompt_tokens=0,
        completion_tokens=0,
        model="",
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        cost=0.0,
        ok=True,
        error=None,
    )


CHUNKER_AGENT: AgentCallable = chunker


__all__ = ["CHUNKER_SPEC", "CHUNKER_AGENT", "chunker"]
