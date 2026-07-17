"""Agent contract — an agent is a callable + its description.

Every primitive — LLM-backed or a plain deterministic callable — is an
``Agent``: a ``Primitive`` (description) bound to an ``AgentCallable``. The
Executor calls ``agent(payload, ctx)`` and doesn't care what the agent IS,
only that it returns an ``AgentResult``. LLM primitives bind through
``build_llm_agent``; deterministic agents bind a plain function directly.

``call_llm_json`` is the shared heart of every LLM primitive: serialize a
payload to JSON, call ``ctx.llm``, parse the response back into a dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ...core.primitive import Primitive
from ...logging import get_logger
from ..llm import LLMClient, LLMResult

log = get_logger("agent")


@dataclass
class AgentResult:
    """What an agent returns to the executor after one tick."""

    output: dict[str, Any]  # structured output passed downstream
    text: str = ""  # raw LLM text (preserved for inspection / eval)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    # Additive — safe defaults so existing constructors (call_llm_json) keep working.
    latency_ms: float = 0.0      # universal: every agent has a duration (deterministic included)
    cost: float = 0.0            # self-reported; 0 for deterministic and when no price table exists
    ok: bool = True              # did the agent succeed? (soft-fail channel, spec §6)
    error: str | None = None     # failure reason when ok=False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ─── general agent contract (an agent is a callable + its description) ───────


@dataclass
class AgentContext:
    """Capability carrier threaded through an agent's run.

    Sync-only for v1. The only live slot is ``llm``; future slots (tools,
    scheduler, engine) are previewed here so the async scheduler (plan.md
    Phase 2.5) lands into this carrier without re-opening the agent contract.
    """

    llm: LLMClient | None = None


# An agent IS a callable from (input, ctx) -> AgentResult. No class hierarchy.
AgentCallable = Callable[[Mapping[str, Any], AgentContext], AgentResult]


@dataclass
class Agent:
    """Universal binder: a Primitive (description) + a callable (behavior).

    Every primitive — LLM-backed or a plain Python callable — is an ``Agent``.
    Execute by calling it directly: ``agent(payload, ctx)``. The transitional
    ``.run(input, llm)`` shim keeps the old test call sites
    (``pool.get(name).run(payload, StubLLM(...))``) working unchanged; new code
    calls ``agent(payload, ctx)`` directly.
    """

    primitive: Primitive
    call: AgentCallable

    @property
    def name(self) -> str:
        return self.primitive.name

    @property
    def role(self) -> str:
        return self.primitive.role

    def __call__(self, input, ctx: AgentContext | None = None) -> AgentResult:
        return self.call(dict(input), ctx)

    def run(self, input, llm: LLMClient | None = None) -> AgentResult:
        """Transitional shim for the old ``agent.run(input, llm)`` contract."""
        return self.call(dict(input), AgentContext(llm=llm))


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
    kind: str = "default",
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> AgentResult:
    """The shared heart of every primitive.

    Sends payload as JSON, parses the response as JSON, returns both.
    Centralising this means primitive definitions stay short — each one
    is just (Primitive + system prompt + the run() mapping for input).

    `kind` identifies which component is calling, so the router can pick
    the per-component model from `config.DEFAULT_LLM_ROUTES`.
    """
    user_msg = json.dumps(payload, ensure_ascii=False)
    # Sentinel for payload bloat / accidental duplication: the engine's
    # per-node merge can re-inject upstream fields an agent doesn't need.
    # `user_chars` is the size the model actually receives.
    log.debug(
        "call_llm_json: kind=%s payload_keys=%s user_chars=%d",
        kind, list(payload.keys()), len(user_msg),
    )
    result: LLMResult = llm.chat(
        system=system_prompt,
        user=user_msg,
        kind=kind,
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


def build_llm_agent(
    primitive: Primitive,
    shape: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> AgentCallable:
    """Build an LLM-backed ``AgentCallable`` from a Primitive spec + a shape fn.

    The shape fn maps the engine's node-input payload to the JSON payload the
    LLM receives — it carries each primitive's re-keying/derivation logic
    (and its own ``log.warning`` calls) verbatim from the old class ``run()``.
    ``primitive.params`` carries ``max_tokens``/``temperature`` (defaults
    1024/0.2, matching ``call_llm_json``). Raises ``RuntimeError`` at call-time
    if ``ctx``/``ctx.llm`` is missing — an LLM agent cannot run without a client.
    """
    prompt = primitive.system_prompt
    kind = primitive.name
    max_tokens = primitive.params.get("max_tokens", 1024)
    temperature = primitive.params.get("temperature", 0.2)

    def _run(input, ctx):
        if ctx is None or ctx.llm is None:
            raise RuntimeError(
                f"llm-kind agent {kind!r} needs ctx.llm; got ctx={ctx!r}"
            )
        return call_llm_json(
            ctx.llm, prompt, dict(shape(input)),
            kind=kind, max_tokens=max_tokens, temperature=temperature,
        )

    return _run
