"""LLM interface + Google Gemini implementation.

The system is backend-pluggable through the `LLMClient` Protocol: primitives
never call the Gemini SDK directly — they go through a client's `chat()`,
which lets tests inject a stub with deterministic responses.

One provider is supported in production: Google Gemini via the `google-genai`
SDK. Per-component model routing is handled by `RouterLLMClient`, which maps
each agent's `kind` to a model id from `archforge.config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import GEMINI_API_KEY_ENV, load_llm_routes


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


# ─── Gemini implementation ───────────────────────────────────────────────────


def _is_gemma(model: str) -> bool:
    """Gemma models are served on the Gemini API but with unreliable
    `system_instruction` support — callers must prepend the system prompt."""
    return model.startswith("gemma")


class GeminiLLMClient:
    """Real Google Gemini client. Requires `GEMINI_API_KEY` in the environment."""

    def __init__(self, model: str = "gemma-4-31b-it", api_key: str | None = None) -> None:
        # Lazily import so the package is importable without google-genai
        # installed (e.g. static analysis). A missing key/SDK surfaces only
        # when a real call is attempted.
        from google import genai
        from google.genai import types

        self.model = model  # overridden per-call by the router
        self._types = types
        self._client = genai.Client(
            api_key=api_key or os.environ.get(GEMINI_API_KEY_ENV)
        )

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        model = kwargs.pop("model", None) or self.model
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 1024)

        # Gemma on the Gemini API doesn't reliably honor a separate system
        # instruction, so fold it into the user contents for those models.
        if _is_gemma(model):
            contents = f"{system}\n\n{user}" if system else user
            config = self._types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        else:
            contents = user
            config = self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        response = self._client.models.generate_content(
            model=model, contents=contents, config=config
        )

        try:
            text = response.text or ""
        except Exception:
            # Blocked / empty responses surface as ValueError on `.text`.
            text = ""

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )


# ─── Per-component router ───────────────────────────────────────────────────


class RouterLLMClient:
    """Dispatch each `chat()` to a specific Gemini model based on `kind`.

    `kind` is set by primitives (their own name) and by the judge ("judge").
    The route table maps component → model id (see `config.load_llm_routes`).
    A single underlying Gemini client is reused across all kinds; only the
    per-call `model` differs.
    """

    def __init__(self, routes: dict[str, str], gemini_client: LLMClient) -> None:
        self._routes = routes
        self._gemini = gemini_client

    @property
    def model(self) -> str:
        return "router"

    def chat(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        kind = kwargs.pop("kind", "default")
        model_id = self._routes.get(kind, self._routes["default"])
        return self._gemini.chat(system, user, model=model_id, **kwargs)


# ─── Factory ────────────────────────────────────────────────────────────────


def get_default_llm_client() -> LLMClient:
    """Return a router wired to the per-component Gemini models.

    Raises RuntimeError if `GEMINI_API_KEY` is not set — there is no silent
    fallback. Set the key (e.g. via a `.env` loaded at startup) before calling.
    """
    if not os.environ.get(GEMINI_API_KEY_ENV):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment (or a "
            ".env file loaded at startup) before running ArchForge."
        )
    routes = load_llm_routes()
    return RouterLLMClient(routes=routes, gemini_client=GeminiLLMClient())


__all__ = [
    "LLMResult",
    "LLMClient",
    "GeminiLLMClient",
    "RouterLLMClient",
    "get_default_llm_client",
]
