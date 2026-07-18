"""fact_checker primitive — verify-and-gate verified facts into the writer.

Role: ``validate``. Position: before the writer (the ``no_validator`` seed's
``before_generate`` slot). The validator's verdicts must flow INTO the
producer of the final output.

This primitive is the dataflow-disciplined successor to the old
verdicts-only fact_checker. The prior version was a *field-dropper*: it
expected pre-made ``claims`` and ``evidence`` and emitted only
``{verdicts, unverified_claims, evidence_gaps}``, dropping the source — so a
``reader → fact_checker → writer`` chain severed the document and the writer
refused (the run-2/3 regression). Two fixes made here:

  1. **Input-flexible.** It reads the source from ``text`` / ``source_text``
     (never preferring ``input`` — the engine threads the raw ``--input``
     *path string* as ``input`` on every node, so preferring it fed the
     model ``"./sample2.txt"`` instead of the document). When no claims are
     supplied upstream, it asks the LLM to **extract** salient claims from
     the source first, so it works bare-after-reader, not just after a
     summarizer.
  2. **Forwards the source as provenance.** It emits ``verified_facts`` —
     the supported/weakly-supported claims the writer composes from (the
     "only verified facts" gate; unsupported/irrelevant claims are dropped
     here) — AND carries ``source_text`` through unchanged as *provenance*
     so downstream (writer, critic) never lose the document. The
     ``source_text`` is injected **deterministically** (post-LLM), not by
     asking the model to echo the doc back (which would be token-wasteful
     and lossy).

A combined extract-and-verify happens in ONE LLM call. The honest
limitation, kept visible: when ``source_text`` IS the evidence a claim was
extracted from, verification is partly self-referential — it catches
*mis-extraction* ("I read 'causes' but the doc says 'correlates'"), not
*fabrication* (there are no fabricated claims; they came from the doc).
Catching fabrication-in-the-*answer* is the ``critic``'s job, post-writer.
"""

from __future__ import annotations

from typing import Any, Mapping

from ...core.primitive import Primitive
from ...logging import get_logger
from .base import AgentCallable, AgentResult, call_llm_json

log = get_logger("agent.fact_checker")

FACT_CHECKER_PROMPT = """\
You are the **fact_checker** primitive in a multi-agent pipeline — a verify-\
and-gate: you verify claims against the source and pass ONLY the verified \
facts downstream, dropping unsupported ones.

You receive a JSON payload with:
- "source_text": the authoritative source document (your only evidence).
- "task": the task the pipeline is serving (extract only claims relevant to \
  it when extracting).
- "claims": a list of claims to verify. MAY BE EMPTY.
- "extract": true when `claims` is empty — you must FIRST pull salient \
  factual claims (relevant to `task`) OUT of `source_text`, then verify each \
  against `source_text`. When `extract` is false, just verify the supplied \
  `claims`.
- "context": optional metadata.

Verify conservatively against `source_text` ONLY:
- "supported": the claim is directly stated/entailed by the source.
- "weakly_supported": the source hints at it but is ambiguous or partial.
- "unsupported": the source does not support it.
- "irrelevant": not a factual claim relevant to the task.

Return JSON with:
- "verified_facts": list of {"fact": <str>, "verdict": "supported" |
   "weakly_supported", "confidence": <0-1>} for the SUPPORTED and
   WEAKLY_SUPPORTED claims only. Drop unsupported/irrelevant from this list —
   they do not pass the gate. When extracting, the `fact` is your re-statement
   of the extracted claim; when verifying supplied claims, echo the claim.
- "verdicts": every claim's verdict (one entry per claim: {"claim", "verdict",
   "confidence", "reason"}) — full audit incl. unsupported ones.
- "unverified_claims": the unsupported + weakly_supported claims (so downstream
   can react). Empty if everything is supported.
- "evidence_gaps": brief descriptions of evidence that would have helped.
   Empty if sufficient.

Do NOT echo `source_text` back — it is forwarded by the runtime, not by you.
Do not summarise, classify, or generate new content beyond the fields above.
Verdicts must be conservative: when in doubt, "weakly_supported" not "supported".
"""

FACT_CHECKER_SPEC = Primitive(
    name="fact_checker",
    level=0,
    role="validate",
    kind="llm",
    system_prompt=FACT_CHECKER_PROMPT,
    input_schema={"type": "object", "required": []},
    output_schema={"type": "object", "required": ["verified_facts"]},
    params={},
)


def _fact_checker_shape(input: Mapping[str, Any]) -> tuple[dict[str, Any], str, bool]:
    """Return (llm_payload, source_text, had_source) for the call.

    Split out so the callable can re-use `source_text` to forward as
    provenance without re-parsing. Source is read from `text`/`source_text` —
    deliberately NOT preferring `input` (the engine's raw `--input` path
    string; preferring it is the run-2/3 bug).
    """
    source_text = input.get("text") or input.get("source_text") or ""
    if not isinstance(source_text, str):
        source_text = str(source_text) if source_text is not None else ""

    claims = input.get("claims") or input.get("key_points") or None
    if not claims and input.get("summary"):
        # A summarizer's single summary string as the lone claim to vet.
        claims = [input["summary"]]
    if claims is not None and not isinstance(claims, list):
        claims = [claims]
    # extract=True when no claims arrived: ask the LLM to pull claims from
    # the source itself, so this works bare-after-reader.
    extract = not bool(claims)

    payload = {
        "task": input.get("task", ""),
        "source_text": source_text,
        "claims": claims or [],
        "extract": extract,
        "context": input.get("context", {}),
    }
    return payload, source_text, bool(source_text)


def fact_checker(input: Mapping[str, Any], ctx=None) -> AgentResult:
    """Verify-and-gate. Forwards `source_text` as provenance; emits
    `verified_facts` for the supported/weakly claims only. Never empties the
    pipeline (warns + forwards the source even if there's nothing to verify).
    """
    if ctx is None or getattr(ctx, "llm", None) is None:
        raise RuntimeError(
            "llm-kind agent 'fact_checker' needs ctx.llm; "
            f"got ctx={ctx!r}"
        )
    payload, source_text, had_source = _fact_checker_shape(input)
    if not had_source:
        log.warning(
            "fact_checker: no source text (input_keys=%s); expected "
            "text/source_text upstream — forwarding empty source",
            list(input.keys()),
        )
    if payload["extract"]:
        log.debug(
            "fact_checker: no claims upstream (input_keys=%s); extracting "
            "claims from source_text (len=%d)",
            list(input.keys()), len(source_text),
        )

    res = call_llm_json(
        ctx.llm, FACT_CHECKER_PROMPT, payload,
        kind="fact_checker", max_tokens=1024, temperature=0.2,
    )
    out = res.output
    # Deterministic provenance forward (NOT via the LLM — see module docstring).
    # `set` overrides any model echo with the authoritative source bytes.
    out["source_text"] = source_text
    out.setdefault("verified_facts", [])
    if not isinstance(out.get("verified_facts"), list):
        out["verified_facts"] = []
    out.setdefault("unverified_claims", [])
    out.setdefault("evidence_gaps", [])
    return res


FACT_CHECKER_AGENT: AgentCallable = fact_checker


__all__ = ["FACT_CHECKER_SPEC", "FACT_CHECKER_AGENT", "fact_checker"]
