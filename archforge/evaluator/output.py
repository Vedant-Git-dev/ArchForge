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
    SPEED_SLA_SECONDS,
    SPEED_PENALTY_FLOOR,
)
from ..core.experience import OutputScores
from ..core.task import Task
from ..executor.engine import PipelineResult
from ..executor.llm import LLMClient


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


# Soft SLA / budget values are centralized in `archforge.config` and
# re-imported above — see SPEED_SLA_SECONDS, SPEED_PENALTY_FLOOR,
# COST_BUDGET_TOKENS, COST_PENALTY_FLOOR.


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
        response = self.llm.chat(
            system=JUDGE_PROMPT,
            user=user_msg,
            kind="judge",
            temperature=0.0,
            max_tokens=512,
        )
        data = self._parse_judge(response.text)
        return (
            float(data.get("accuracy", 0.0)),
            float(data.get("completeness", 0.0)),
            str(data.get("rationale", "")),
        )

    @staticmethod
    def _parse_judge(text: str) -> dict[str, Any]:
        # Tolerant parser: try direct JSON, then a fenced block, then fallback.
        for candidate in (text, text.strip().strip("```").strip()):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
        # Last-ditch: search for the JSON-like substring
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                pass
        # Failure mode: midpoint score, the experience is still useful.
        return {"accuracy": 0.5, "completeness": 0.5, "rationale": "judge parse failed"}


__all__ = ["OutputEvaluator", "SPEED_SLA_SECONDS", "COST_BUDGET_TOKENS"]
