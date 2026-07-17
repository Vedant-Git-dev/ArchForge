"""regex_extractor — a deterministic (non-LLM) primitive.

The first non-LLM agent. Proves the ``Agent`` contract generalizes beyond
LLMs: it runs with no ``ctx.llm`` in scope, returns a zero-token
``AgentResult``, and never raises — soft-fail goes through ``ok``/``error``
(spec §6). Registered as ``role="analyze"`` so it coexists with the
``classifier``/``summarizer`` analyze primitives and stays out of the
validate-role name-sort the no_validator seed relies on (spec §7).
"""

from __future__ import annotations

import re
import time
from typing import Any

from ...core.primitive import Primitive
from .base import AgentResult

# Built-in pattern registry. Callers may add named patterns via input["patterns"]
# (each {"name": ..., "pattern": ...}); bad patterns land in output["pattern_errors"].
_PATTERNS: dict[str, str] = {
    "emails": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-.]+)+",
    "urls": r"https?://[^\s)]+",
    "dates": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b",
    "phones": r"\+?\d[\d\s\-()]{6,}\d",
}


REGEX_SPEC = Primitive(
    name="regex_extractor",
    level=0,
    role="analyze",
    kind="deterministic",
    system_prompt="",  # inert for the deterministic kind
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "patterns": {"type": "array", "items": {"type": "string"}},
        },
    },
    output_schema={"type": "object", "required": ["matches"]},
    params={},
)


def regex_extractor(input: dict[str, Any], ctx=None) -> AgentResult:
    """Deterministic regex extractor.

    ``ctx`` is the ``AgentContext``; this agent ignores ``ctx.llm`` (it reads
    nothing from the context — the zero-token contract proof). Never raises:
    a malformed caller pattern is recorded in ``output["pattern_errors"]``
    with ``ok`` staying ``True`` (soft-fail channel, spec §6).
    """
    t0 = time.perf_counter()
    text = input.get("text") or input.get("input") or ""
    if not isinstance(text, str):
        text = str(text)

    active: dict[str, str] = dict(_PATTERNS)
    pattern_errors: list[dict[str, str]] = []
    extra = input.get("patterns")
    if isinstance(extra, list):
        for p in extra:
            if isinstance(p, dict):
                name = p.get("name")
                pat = p.get("pattern")
            else:
                name = pat = None
                try:
                    pat = str(p)
                except Exception:  # noqa: BLE001 — never raise from a deterministic agent
                    pat = ""
            if not name or not pat:
                pattern_errors.append({"pattern": repr(p), "error": "missing name or pattern"})
                continue
            try:
                re.compile(pat)  # validate before adding
                active[str(name)] = pat
            except re.error as exc:
                pattern_errors.append({"pattern_name": str(name), "error": f"bad regex: {exc}"})

    matches: dict[str, list[str]] = {}
    for name, pat in active.items():
        try:
            found = re.findall(pat, text)
        except re.error as exc:
            pattern_errors.append({"pattern_name": name, "error": f"bad regex: {exc}"})
            continue
        flat: list[str] = []
        for m in found:
            if isinstance(m, tuple):
                # Grouped patterns return tuples; take the first non-empty group.
                s = next((x for x in m if x), "")
            else:
                s = m
            if s:
                flat.append(str(s))
        if flat:
            matches[name] = flat

    out: dict[str, Any] = {"matches": matches}
    if pattern_errors:
        out["pattern_errors"] = pattern_errors

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


__all__ = ["regex_extractor", "REGEX_SPEC"]
