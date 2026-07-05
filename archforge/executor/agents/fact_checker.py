"""fact_checker primitive – verify claims against available evidence."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json


SYSTEM_PROMPT = """\
You are the **fact_checker** primitive in a multi-agent pipeline.

Your job is to evaluate whether each claim is supported by the evidence
provided, and to flag unsupported or weakly-supported statements. You do
not summarise, classify, or generate new content — you only verify.

Inputs to this primitive commonly include:
- "claims": list of claims to verify (typically strings).
- "evidence": a corpus or list of supporting documents / summaries.

Return JSON with:
- "verdicts": list of {"claim": <str>, "verdict": "supported" | "weakly_supported" |
   "unsupported" | "irrelevant", "confidence": <float 0-1>, "reason": <short str>}.
  One entry per claim, in the same order as input.
- "unverified_claims": extracted list of unsupported + weakly_supported claims
  so downstream agents can react. Empty list if everything checks out.
- "evidence_gaps": list of brief descriptions of types of evidence that would
  have helped. Empty list if evidence was sufficient.

Verdicts must be conservative: when in doubt, mark "weakly_supported" rather
than "supported". Never invent evidence that wasn't provided.
"""


class FactCheckerAgent:
    name = "fact_checker"
    role = "validate"

    def __init__(self) -> None:
        self.primitive = Primitive(
            name="fact_checker",
            level=0,
            role="validate",
            system_prompt=SYSTEM_PROMPT,
            input_schema={"type": "object", "required": ["claims", "evidence"]},
            output_schema={"type": "object", "required": ["verdicts"]},
        )

    def run(self, input: dict[str, Any], llm: LLMClient) -> AgentResult:
        # The fact_checker accepts claims + evidence from upstream.
        # Phase 1 callers usually pass {"claims":[...], "evidence": "..."}.
        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {
                "claims": input.get("claims", []),
                "evidence": input.get("evidence", input.get("text", "")),
                "context": input.get("context", {}),
            },
        )


__all__ = ["FactCheckerAgent"]
