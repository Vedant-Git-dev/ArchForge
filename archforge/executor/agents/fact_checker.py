"""fact_checker primitive – verify claims against available evidence."""

from __future__ import annotations

from typing import Any

from ...core.primitive import Primitive
from ...logging import get_logger
from ..llm import LLMClient
from .base import AgentResult, BaseAgent, call_llm_json

log = get_logger("agent.fact_checker")


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
        # Claims to verify. A well-formed upstream (an explicit `claims`
        # source, or the summarizer) hands us concrete claims; otherwise we
        # derive them from the summarizer's `key_points`, or from its
        # `summary` string as a single claim. Evidence is the source text
        # the claims were drawn from — the engine carries the original input
        # as `input`, with upstream `text` as a fallback. When the pipeline
        # topology supplies neither, we run on empty and the warnings below
        # make that loss visible instead of silent.
        claims = input.get("claims")
        if not claims:
            claims = input.get("key_points") or (
                [input["summary"]] if input.get("summary") else []
            )
        evidence = input.get("evidence")
        if not evidence:
            evidence = input.get("input") or input.get("text") or ""

        if not claims:
            log.warning(
                "fact_checker: no claims to verify (input_keys=%s); expected "
                "claims/key_points/summary from upstream",
                list(input.keys()),
            )
        if not evidence:
            log.warning(
                "fact_checker: no evidence source (input_keys=%s); expected "
                "evidence/input/text to carry the source text",
                list(input.keys()),
            )

        return call_llm_json(
            llm,
            SYSTEM_PROMPT,
            {
                "claims": claims,
                "evidence": evidence,
                "context": input.get("context", {}),
            },
            kind=self.name,
        )


__all__ = ["FactCheckerAgent"]
