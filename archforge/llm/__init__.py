"""Provider-agnostic LLM abstraction (spec §3 Judge/Architect, Phase 4).

Both thinking components — the Judge (Phase 5) and the Architect (Phase 6) — talk
to a model only through the `LLMClient` protocol. This is the **single seam**
where "real vs fake" lives: tests plug in `ScriptedLLM` (deterministic, free,
crashable); Phase 11 plugs in `Anthropic`/`OpenAI` clients. Neither the Judge nor
the Architect changes when the provider changes.
"""

from __future__ import annotations

from archforge.llm.base import (
    Completion,
    LLMClient,
    LLMError,
    Message,
    Role,
    Usage,
)
from archforge.llm.scripted import ScriptedLLM

__all__ = ["Completion", "LLMClient", "LLMError", "Message", "Role", "Usage", "ScriptedLLM"]
