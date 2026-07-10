"""LLM interface + Groq implementation.

The system is backend-pluggable through the `LLMClient` Protocol: primitives
never call the Groq SDK directly — they go through a client's `chat()`,
which lets tests inject a stub with deterministic responses.

One provider is supported in production: Groq via the `groq` SDK. Each
pipeline component + the judge routes to a specific model id keyed by its
own name. Per-component overrides are read from `ARCHFORGE_LLM_<COMPONENT>`
env vars. Groq is OpenAI-compatible and honours a separate `system` role
natively, so no prompt-folding hack is required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import GROQ_API_KEY_ENV, load_llm_routes


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


# ─── Groq implementation ────────────────────────────────────────────────────


class GroqLLMClient:
    """Real Groq client. Requires `GROQ_API_KEY` in the environment."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None) -> None:
        # Lazily import so the package is importable without groq installed
        # (e.g. static analysis). A missing key/SDK surfaces only when a real
        # call is attempted.
        from groq import Groq

        self.model = model  # overridden per-call by the router
        self._client = Groq(api_key=api_key or os.environ.get(GROQ_API_KEY_ENV))

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        model = kwargs.pop("model", None) or self.model
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 1024)

        # Groq is OpenAI-compatible: the `system` role is honoured natively,
        # so no prompt-folding is needed (unlike Gemma served on the Gemini API).
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )


# ─── Per-component router ────────────────────────────────────────────────────


class RouterLLMClient:
    """Dispatch each `chat()` to a specific Groq model based on `kind`.

    `kind` is set by primitives (their own name) and by the judge ("judge").
    The route table maps component → model id (see `config.load_llm_routes`).
    A single underlying Groq client is reused across all kinds; only the
    per-call `model` differs.
    """

    def __init__(self, routes: dict[str, str], groq_client: LLMClient) -> None:
        self._routes = routes
        self._groq = groq_client

    @property
    def model(self) -> str:
        return "router"

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        kind = kwargs.pop("kind", "default")
        model_id = self._routes.get(kind, self._routes["default"])
        return self._groq.chat(system, user, model=model_id, **kwargs)


# ─── Factory ─────────────────────────────────────────────────────────────────


def get_default_llm_client() -> LLMClient:
    """Return a router wired to the per-component Groq models.

    Raises RuntimeError if `GROQ_API_KEY` is not set — there is no silent
    fallback. Set the key (e.g. via a `.env` loaded at startup) before calling.
    """
    if not os.environ.get(GROQ_API_KEY_ENV):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment (or a "
            ".env file loaded at startup) before running ArchForge."
        )
    routes = load_llm_routes()
    return RouterLLMClient(routes=routes, groq_client=GroqLLMClient())


__all__ = [
    "LLMResult",
    "LLMClient",
    "GroqLLMClient",
    "RouterLLMClient",
    "get_default_llm_client",
]
