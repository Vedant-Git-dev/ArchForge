"""Base agent contract + AgentResult + a tiny primitive helper.

All primitives in this package follow the same shape:
    1. Take a JSON-serialisable input dict from upstream.
    2. Send it to the LLM with a fixed system_prompt.
    3. Parse the response back into a dict.
    4. Return AgentResult(output=..., text=..., tokens=...).

The Executor doesn't care WHAT a primitive does, only that it follows
this contract. Phase 2+ may add streaming, retries, validation hooks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...core.primitive import Primitive
from ..llm import LLMClient, LLMResult


@dataclass
class AgentResult:
    """What an agent returns to the executor after one tick."""

    output: dict[str, Any]  # structured output passed downstream
    text: str = ""  # raw LLM text (preserved for inspection / eval)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseAgent(Protocol):
    """The contract every primitive implements.

    Implementations expose `name` and `role` (read from Primitive),
    and a `run(input_dict, llm_client) -> AgentResult` method.
    """

    name: str
    role: str
    primitive: Primitive

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult: ...


# ─── shared helpers ─────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response.

    Models frequently wrap JSON in markdown fences or add prose. We try:
      1. Direct json.loads on the whole string.
      2. ```json ... ``` fence extraction.
      3. First {...} block in the text.

    On any failure, fall back to {"raw_text": text}. This keeps primitives
    forgiving — a malformed LLM response shouldn't crash the whole pipeline.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass

    # First {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"raw_text": text}


def call_llm_json(
    llm: LLMClient,
    system_prompt: str,
    payload: dict[str, Any],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> AgentResult:
    """The shared heart of every primitive.

    Sends payload as JSON, parses the response as JSON, returns both.
    Centralising this means primitive definitions stay short — each one
    is just (Primitive + system prompt + the run() mapping for input).
    """
    user_msg = json.dumps(payload, ensure_ascii=False)
    result: LLMResult = llm.chat(
        system=system_prompt,
        user=user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    parsed = _extract_json(result.text)
    if "raw_text" in parsed and "text" not in parsed:
        # Make the raw text recoverable downstream
        parsed.setdefault("text", result.text)
    return AgentResult(
        output=parsed,
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        model=result.model,
    )
