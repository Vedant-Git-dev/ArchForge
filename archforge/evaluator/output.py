"""Output-quality evaluator (Phase 1: simple, no diagnosis).

Three surfaces (per plan.md):
- accuracy     → LLM judge: does the output satisfy the task? (0-1)
- speed_normalized → wall-clock time vs a soft SLA (0-1)
- cost_normalized  → token usage vs a soft budget (0-1, lower=better)

Phase 1 deliberately does NOT emit diagnoses or structural scores —
those are Phase 2+ work. Composite score is a fixed linear blend
(50% accuracy, 25% speed, 25% cost).
"""

from __future__ import annotations

import json
from typing import Any

from ..config import (
    COST_BUDGET_TOKENS,
    COST_PENALTY_FLOOR,
    DIAGNOSIS_ACCURACY_LOW,
    DIAGNOSIS_COST_LOW,
    DIAGNOSIS_SPEED_LOW,
    SPEED_SLA_SECONDS,
    SPEED_PENALTY_FLOOR,
    STRUCTURAL_ROOTS,
)
from ..core.experience import OutputScores, StructuralScores
from ..core.pipeline import PipelineDAG
from ..core.task import Task
from ..executor.engine import PipelineResult
from ..executor.llm import LLMClient
from ..logging import get_logger

log = get_logger("evaluator.output")


JUDGE_PROMPT = """\
You are evaluating the output of a multi-agent pipeline that was asked
to satisfy a particular task. Be precise and conservative.

Return JSON with these fields:
- "accuracy": float in [0, 1] — does the output actually satisfy the task?
  1.0 = fully satisfied, 0.0 = completely missed, intermediate = partial.
- "completeness": float in [0, 1] — are there obvious gaps in the output?
- "rationale": short string explaining the score in 1-3 sentences.
  Reference specific evidence from the output when relevant.

You are NOT evaluating the pipeline's structure (efficiency, parallelism,
redundancy) — only the OUTPUT the user would see.
"""


COMBINED_JUDGE_PROMPT = """\
You are the judge AND diagnostician of one multi-agent pipeline run.

Step 1 — score the OUTPUT: return "accuracy" and "completeness" floats in \
[0,1], and a short "rationale".

Step 2 — diagnose. The data PRE-MARKS which metrics are POOR (below their \
floor) under "poor_signals". You MUST emit a diagnosis for every pre-marked \
poor metric, AND for any other poor metric or structural defect you detect. \
Reason ONLY from the supplied data (task, input_word_count, final output, \
the normalized metrics with their floors, the pipeline topology with each \
node's agent_type + role and per-node traces). Do not invent metrics, nodes, \
or verdicts.

A metric is poor when its normalized value is below its floor. A structural \
defect is poor when the topology has unused output nodes, redundant agents, \
a too-long chain, or agents that don't contribute to THIS task.

Emit MULTIPLE diagnoses when more than one thing is poor — do not collapse \
them into one. You may emit several diagnoses on the same or different axes.

Return JSON with:
- "accuracy": float in [0,1] — does the output satisfy the task?
- "completeness": float in [0,1] — are there obvious gaps?
- "rationale": 1-3 sentences on the score.
- "diagnoses": array, one entry per poor cause. [] only if genuinely nothing
  is poor. Each entry:
  - "axis": one of "accuracy", "speed", "cost", "structure", "all"
    ("all" is for root causes that span multiple axes)
  - "severity": float in [0,1] (1.0 = severe)
  - "structural_root": EXACTLY one of {roots}, or "unknown:<short>" if none fit
    (novel roots are kept for future learning, not matched to a fix today)
  - "reason": 1-2 sentences naming the concrete cause from the data
  - "target_nodes": list of the pipeline node_ids this diagnosis is about,
    copied from the supplied topology. REQUIRED and non-empty for roots that
    target specific agents (unnecessary_agents, redundant_agents, \
unused_outputs); empty list for pipeline-wide roots
    (serial_bottleneck, deep_chain, no_critique_loop, no_validator).

Allowed roots and what they mean:
- "no_validator": no validate-role node is present (accuracy at risk)
- "serial_bottleneck": long serial critical path + low parallelism → slow
- "redundant_agents": two+ agents duplicate each other's work
- "unused_outputs": a node whose output no downstream node reads
- "no_critique_loop": a generate step with no critique→revision cycle
- "unnecessary_agents": agent nodes that do not earn their place for THIS
  task — either they don't contribute to the final output, or the step is
  overkill given the task and the input size. 
  Decide necessity per agent from the task, input_word_count, and
  each node's trace — do NOT catenate module names. Give the offending
  node_ids in target_nodes.
- "deep_chain": an unusually long, fragile dependency chain

If cost is poor because the pipeline did too much work for a small input,
that is "unnecessary_agents" (the steps that were overkill), NOT a separate
root. Diagnose only metrics that are ACTUALLY poor — never invent problems.
Return only the JSON object.
"""


# Soft SLA / budget values are centralized in `archforge.config` and
# re-imported above — see SPEED_SLA_SECONDS, SPEED_PENALTY_FLOOR,
# COST_BUDGET_TOKENS, COST_PENALTY_FLOOR.


def _parse_json_dict(text: str) -> dict[str, Any] | None:
    """Tolerant JSON-object parse shared by the score judge and the combined
    judge. Returns the dict on success, None on any parse failure.
    """
    s = text.strip()
    for candidate in (s, s.strip().strip("```").strip()):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _build_judge_payload(
    task: Task,
    result: PipelineResult,
    structural: StructuralScores | None,
    roles: dict[str, str],
    *,
    speed: float | None = None,
    cost: float | None = None,
) -> dict[str, Any]:
    """Grounding for the combined judge — every fact the LLM should reason over.

    Pre-computes the speed/cost normalizations (deterministic, no LLM) and
    hands the LLM the *normalized* values plus their diagnosis floors and a
    pre-marked ``poor_signals`` map. The cost=0.0 / tokens=14287 case stays
    silent only if the model can't see the floor — so we show it the floor.
    Accuracy is determined by the model's own Step 1, so we pass its floor
    only and let it self-check.
    """
    payload: dict[str, Any] = {
        "task_description": task.description,
        "task_type": task.type,
        "output": result.final_output,
    }
    # structural surface + per-node traces always travel together
    topology = [
        {
            "node_id": t.node_id,
            "agent_type": t.agent_type,
            "role": roles.get(t.agent_type, "unknown"),
            "duration_seconds": round(t.duration_seconds, 3),
            "tokens": t.total_tokens,
        }
        for t in result.traces
    ]
    payload["topology"] = topology
    payload["raw_signals"] = {
        "wall_time_seconds": round(result.wall_time_seconds, 3),
        "total_tokens": result.total_tokens,
    }
    if structural is not None:
        # Pre-mark which metrics are poor so the model cannot stay silent on
        # a tripped floor (the silent-on-cost=0 bug). Accuracy is judged by
        # the model in Step 1; we expose its floor for self-check.
        poor = {
            "speed": bool(speed is not None and speed < DIAGNOSIS_SPEED_LOW),
            "cost": bool(cost is not None and cost < DIAGNOSIS_COST_LOW),
            "accuracy_self_check": "diagnose if the accuracy you return in Step 1 is below "
            f"{DIAGNOSIS_ACCURACY_LOW}",
        }
        if structural.unused_outputs:
            poor["structural_unused_outputs"] = True
        if structural.redundant_agents:
            poor["structural_redundant_agents"] = True
        payload["metrics"] = {
            "speed_normalized": None if speed is None else round(speed, 3),
            "speed_floor": DIAGNOSIS_SPEED_LOW,
            "cost_normalized": None if cost is None else round(cost, 3),
            "cost_floor": DIAGNOSIS_COST_LOW,
            "accuracy_floor": DIAGNOSIS_ACCURACY_LOW,
            "pipeline_length": structural.pipeline_length,
            "critical_path_length": structural.critical_path_length,
            "parallelism_ratio": structural.parallelism_ratio,
            "dependency_depth": structural.dependency_depth,
            "structural_score": structural.score,
            "unused_output_nodes": structural.unused_outputs,
            "redundant_agent_nodes": structural.redundant_agents,
        }
        payload["poor_signals"] = poor
    return payload


def _normalize(value: float, low: float, high: float, *, lower_is_better: bool) -> float:
    """Map `value` to 0-1, with linear decay between low and high.

    If `lower_is_better`, the value is treated as a penalty:
    - below `low` → 1.0
    - above `high` → 0.0
    - in between → linearly scaled

    If `lower_is_better=False`, the inverse applies (e.g. for accuracy).
    """
    if lower_is_better:
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        # Linear: value=low → 1, value=high → 0
        return 1.0 - (value - low) / (high - low)
    else:
        # For "more is better" (accuracy etc.), clamp to [0,1].
        return max(0.0, min(1.0, value))


class OutputEvaluator:
    """Compute OutputScores for a finished pipeline run.

    Stateless caller — instantiate once per session.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def evaluate(
        self,
        task: Task,
        result: PipelineResult,
        *,
        user_rating: float | None = None,
    ) -> OutputScores:
        accuracy, completeness, _rationale = self._judge(task, result)

        speed = _normalize(
            result.wall_time_seconds, SPEED_SLA_SECONDS, SPEED_PENALTY_FLOOR, lower_is_better=True
        )
        cost = _normalize(
            result.total_tokens, COST_BUDGET_TOKENS, COST_PENALTY_FLOOR, lower_is_better=True
        )
        log.info(
            "evaluate: judge accuracy=%.3f completeness=%.3f | speed=%.3f (wall=%.3fs) cost=%.3f (tok=%d)",
            accuracy, completeness, speed, result.wall_time_seconds, cost, result.total_tokens,
        )

        return OutputScores(
            accuracy=accuracy,
            completeness=completeness,
            speed_normalized=speed,
            cost_normalized=cost,
            user_rating=user_rating,
        )

    # ----- LLM judge -----

    def _judge(self, task: Task, result: PipelineResult) -> tuple[float, float, str]:
        """Run the LLM judge on the final output. Returns (accuracy, completeness, rationale)."""
        payload = {
            "task_description": task.description,
            "task_type": task.type,
            "output": result.final_output,
            "context": {
                "wall_time_seconds": round(result.wall_time_seconds, 3),
                "tokens_used": result.total_tokens,
            },
        }
        user_msg = json.dumps(payload, ensure_ascii=False)
        log.debug("evaluate: invoking judge (output_len=%d)", len(result.final_output))
        response = self.llm.chat(
            system=JUDGE_PROMPT,
            user=user_msg,
            kind="judge",
            temperature=0.0,
            max_tokens=512,
        )
        data = self._parse_judge(response.text)
        if data.get("rationale") == "judge parse failed":
            log.warning("evaluate: judge response failed to parse; using midpoint fallback")
        log.debug("evaluate: judge returned %d chars", len(response.text))
        return (
            float(data.get("accuracy", 0.0)),
            float(data.get("completeness", 0.0)),
            str(data.get("rationale", "")),
        )

    @staticmethod
    def _parse_judge(text: str) -> dict[str, Any]:
        obj = _parse_json_dict(text)
        if obj is None:
            # Failure mode: midpoint score, the experience is still useful.
            return {"accuracy": 0.5, "completeness": 0.5, "rationale": "judge parse failed"}
        return obj

    # ----- combined score + diagnosis judge (Phase 2) -----
    #
    # ONE judge call that returns scores AND raw diagnoses. Folds the second
    # LLM call the diagnostician would otherwise have made into the existing
    # scoring call → diagnosis adds ZERO LLM calls vs the Phase 1 run path.
    # The structural metrics must be computed by the caller before this call
    # (they're pure topology, no LLM) and passed in to ground the diagnoses.

    def evaluate_with_diagnosis(
        self,
        task: Task,
        result: PipelineResult,
        *,
        structural: StructuralScores,
        roles: dict[str, str],
        user_rating: float | None = None,
    ) -> tuple[OutputScores, list[dict[str, Any]] | None]:
        """Score + raw diagnoses in ONE judge call.

        Returns (scores, raw_diagnoses). `raw_diagnoses` is the LLM's parsed
        `diagnoses` list (each entry a dict with axis/severity/structural_root/
        reason — NOT yet clamped to the vocabulary), or None if the judge gave
        no usable diagnoses list or failed to parse at all. Sanitizing that
        list (clamp roots, drop malformed, fall back to the deterministic floor,
        augment deterministic structural facts) is the Diagnostician's job.
        """
        accuracy, completeness, _rationale, raw, speed, cost = self._judge_with_diagnosis(
            task, result, structural, roles
        )
        log.info(
            "evaluate_with_diagnosis: judge accuracy=%.3f completeness=%.3f | "
            "speed=%.3f cost=%.3f | raw_diagnoses=%s",
            accuracy, completeness, speed, cost,
            "n/a" if raw is None else f"{len(raw)} entries",
        )
        return (
            OutputScores(
                accuracy=accuracy,
                completeness=completeness,
                speed_normalized=speed,
                cost_normalized=cost,
                user_rating=user_rating,
            ),
            raw,
        )

    def _judge_with_diagnosis(
        self,
        task: Task,
        result: PipelineResult,
        structural: StructuralScores,
        roles: dict[str, str],
    ) -> tuple[float, float, str, list[dict[str, Any]] | None, float, float]:
        # Speed/cost are deterministic normalizations — compute them BEFORE
        # the judge call so we can ground the diagnosis with the normalized
        # values + their floors (otherwise the model sees only raw token
        # counts and can't tell cost=0.0 is poor, which is the silence bug).
        speed = _normalize(
            result.wall_time_seconds, SPEED_SLA_SECONDS, SPEED_PENALTY_FLOOR, lower_is_better=True
        )
        cost = _normalize(
            result.total_tokens, COST_BUDGET_TOKENS, COST_PENALTY_FLOOR, lower_is_better=True
        )
        payload = _build_judge_payload(
            task, result, structural, roles, speed=speed, cost=cost
        )
        response = self.llm.chat(
            system=COMBINED_JUDGE_PROMPT.format(roots=", ".join(STRUCTURAL_ROOTS)),
            user=json.dumps(payload, ensure_ascii=False),
            kind="judge",
            temperature=0.0,
            max_tokens=1024,
        )
        data = _parse_json_dict(response.text)
        if data is None:
            log.warning("_judge_with_diagnosis: parse failure; returning neutral scores + no raw diagnoses")
            return 0.5, 0.5, "judge parse failed", None, speed, cost
        acc = float(data.get("accuracy", 0.0))
        comp = float(data.get("completeness", 0.0))
        rationale = str(data.get("rationale", ""))
        raw = data.get("diagnoses")
        if not isinstance(raw, list):
            raw = None
        return acc, comp, rationale, raw, speed, cost


__all__ = ["OutputEvaluator", "SPEED_SLA_SECONDS", "COST_BUDGET_TOKENS"]
