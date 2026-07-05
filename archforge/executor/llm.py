"""LLM interface + Groq implementation.

The system is backend-pluggable: primitives never call Groq directly.
They go through LLMClient.chat(), which lets tests inject a FakeLLM with
deterministic responses.

Groq is OpenAI-compatible; the official `groq` SDK semantics are used here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    """Minimum interface primitives depend on."""

    model: str

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        ...


# ─── Groq implementation ───────────────────────────────────────────────────

DEFAULT_GROQ_MODEL = os.environ.get("ARCHFORGE_GROQ_MODEL", "llama-3.1-8b-instant")

# Pricing (USD per 1M tokens) — used by the evaluator for cost normalisation.
# Defaults to llama-3.1-8b-instant pricing on Groq as of plan inception.
GROQ_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
}


def groq_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = GROQ_PRICING_USD_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (
        pricing["input"] * prompt_tokens / 1_000_000
        + pricing["output"] * completion_tokens / 1_000_000
    )


class GroqLLMClient:
    """Real Groq client. Requires `GROQ_API_KEY` in the environment."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_GROQ_MODEL
        # Import lazily so the package is importable without groq installed
        # (e.g. in CI environments without the dep).
        from groq import Groq  # type: ignore

        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        # Strip kwargs we don't pass through; useful for tests that want to
        # override temperature but not pass it to Groq.
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        resp = self._client.chat.completions.create(**params)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=self.model,
        )


# ─── Fake / deterministic implementation ───────────────────────────────────


class FakeLLMClient:
    """Returns deterministic outputs based on a configured script of responses.

    Used by tests. Each call advances a counter; the script values cycle.

    For agents that want the call to echo something useful, scripts can be
    functions that take (system, user) and return text — that lets the test
    assert what each primitive saw.
    """

    def __init__(
        self,
        scripted_responses: list[str | None] | None = None,
        model: str = "fake-llm",
    ) -> None:
        """`scripted_responses[i]` is returned on call i, cycling if exhausted.

        Passing None for an entry returns a generic JSON stub. Useful for
        agents that need non-empty output but where the content doesn't matter
        for the test.
        """
        self.model = model
        self._script = scripted_responses or []
        self._call_count = 0
        self.call_log: list[dict[str, Any]] = []  # inspection

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        self.call_log.append({"system": system, "user": user, "kwargs": kwargs})
        idx = self._call_count
        self._call_count += 1
        if self._script:
            text = self._script[idx % len(self._script)]
            if text is None:
                text = json.dumps({"status": "ok"})
        else:
            text = json.dumps({"status": "ok"})
        # Crude token estimate: 1 token per ~4 chars total input+output.
        approx = (len(system) + len(user) + len(text)) // 4
        return LLMResult(
            text=text,
            prompt_tokens=approx // 2,
            completion_tokens=approx - approx // 2,
            model=self.model,
        )


# ─── Factory ────────────────────────────────────────────────────────────────


def get_default_llm_client() -> LLMClient:
    """Return a working LLMClient for the current environment.

    - If GROQ_API_KEY is set → real Groq.
    - Else → FakeLLM with generic JSON stubs (so primitives still produce
      something useful for offline exploration).
    """
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLMClient()
    return FakeLLMClient()
