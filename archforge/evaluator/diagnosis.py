"""Diagnostician — Phase 2 Surface 3 (why each POOR metric is poor).

The LLM call that produces diagnoses is FOLDED INTO the score judge —
see OutputEvaluator.evaluate_with_diagnosis — so diagnosis adds ZERO LLM
calls vs the Phase 1 run path. This module owns everything that turns the
judge's *raw* diagnoses into the clean list stored on an experience:

  - clamp each `structural_root` to the controlled vocabulary
    (`STRUCTURAL_ROOTS`); anything novel is kept as an `unknown:<tag>` for
    future learning but matches no seeded intervention today
  - validate axis, clamp severity to [0,1], default an empty reason
  - fall back to the deterministic rule floor when the judge gives no usable
    raw list (offline runs with no LLM, or a judge parse failure)
  - augment: the structural-fact diagnoses (unused_outputs / redundant_agents)
    are deterministic regardless of what the LLM said, so they are appended
    whenever the LLM omitted them — they can never be "talked away".

The deterministic floor is also the standalone path when no LLM is wired in
(`evaluate(raw_diagnoses=None)`), which is what the offline test path uses.
For the live path, main.py passes the raw diagnoses the evaluator returned.
"""

from __future__ import annotations

from typing import Any

from ..config import (
    DIAGNOSIS_ACCURACY_LOW,
    DIAGNOSIS_BOTTLENECK_MIN_PATH,
    DIAGNOSIS_COST_LOW,
    DIAGNOSIS_DEEP_CHAIN_MIN,
    DIAGNOSIS_PARALLELISM_LOW,
    DIAGNOSIS_SPEED_LOW,
    STRUCTURAL_ROOTS,
)
from ..core.experience import Diagnosis, OutputScores, StructuralScores
from ..core.pipeline import PipelineDAG

# Axes the diagnostician is allowed to speak to.
_AXES = ("accuracy", "speed", "cost", "structure")

# Structural facts that are true regardless of the LLM's reading, and so are
# ALWAYS merged into the final list when present (never omitted, never spoken
# away by an LLM that says "nothing structural is wrong").
_DETERMINISTIC_ROOTS = {"unused_outputs", "redundant_agents"}


def _clamp_severity(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


def _sanitize_root(root: Any) -> str:
    """Constrain structural_root to the controlled vocabulary.

    Anything outside {STRUCTURAL_ROOTS} and not an "unknown:..." escape is
    mapped to a bare "unknown" so the intervention matcher can never pick up
    a hallucinated key. Novel-but-plausible roots keep a short tag for future
    discovery. The matching detail (which key an intervention library looks
    up) is a Phase 2 follow-on deliverable.
    """
    if not isinstance(root, str):
        return "unknown"
    r = root.strip().lower()
    if r in STRUCTURAL_ROOTS:
        return r
    if r.startswith("unknown") and len(r) <= 40:
        return r
    return "unknown"


def _valid_axis(a: Any) -> str | None:
    if isinstance(a, str) and a.strip().lower() in _AXES:
        return a.strip().lower()
    return None


class Diagnostician:
    """Turn raw judge diagnoses into the final, sanitized diagnosis list.

    ``evaluate(raw_diagnoses=None)`` → the deterministic rule floor (offline).
    ``evaluate(raw_diagnoses=[...])``  → finalize the LLM list (live path).
    Either branch can also be used directly; main.py calls evaluate() with
    the raw list the score judge returned.
    """

    def evaluate(
        self,
        output: OutputScores,
        structural: StructuralScores,
        pipeline: PipelineDAG,
        *,
        roles: dict[str, str] | None = None,
        raw_diagnoses: list[dict[str, Any]] | None = None,
    ) -> list[Diagnosis]:
        roles = roles or {}
        if raw_diagnoses is None:
            # No usable LLM diagnosis data → deterministic floor.
            return self._rule_based(output, structural, pipeline, roles)
        return self._finalize(raw_diagnoses, output, structural, pipeline, roles)

    # ----- finalize the LLM-authored list -----

    def _finalize(
        self,
        raw: list[dict[str, Any]],
        output: OutputScores,
        structural: StructuralScores,
        pipeline: PipelineDAG,
        roles: dict[str, str],
    ) -> list[Diagnosis]:
        floor = self._rule_based(output, structural, pipeline, roles)
        parsed: list[Diagnosis] = []
        for d in raw:
            if not isinstance(d, dict):
                continue
            axis = _valid_axis(d.get("axis"))
            if axis is None:
                continue
            reason = str(d.get("reason", "")).strip() or "(no reason given)"
            parsed.append(
                Diagnosis(
                    axis=axis,
                    severity=_clamp_severity(d.get("severity")),
                    reason=reason,
                    structural_root=_sanitize_root(d.get("structural_root")),
                )
            )

        if not parsed:
            # The LLM considered and found nothing diagnosable — but the
            # deterministic structural facts stand regardless; keep them.
            return [d for d in floor if d.structural_root in _DETERMINISTIC_ROOTS]

        # Augment: append any deterministic structural-fact diagnoses the LLM
        # omitted. These are topology facts, so they override omission.
        have = {d.structural_root for d in parsed}
        out = list(parsed)
        for d in floor:
            if d.structural_root in _DETERMINISTIC_ROOTS and d.structural_root not in have:
                out.append(d)
        return out

    # ----- deterministic fallback floor -----

    def _rule_based(
        self,
        output: OutputScores,
        structural: StructuralScores,
        pipeline: PipelineDAG,
        roles: dict[str, str],
    ) -> list[Diagnosis]:
        """Approximate diagnosis from metrics + topology. No LLM.

        The floor: emitted only when no LLM raw list is available or the
        judge failed to parse. Roots and severity here are approximate — the
        clamped LLM list overrides with content-grounded cause whenever one is
        returned. Still used (a) offline with no LLM, and (b) to source the
        deterministic structural-fact diagnoses merged into any LLM list.
        """
        diag: list[Diagnosis] = []
        has_validator = any(roles.get(n.agent_type) == "validate" for n in pipeline.nodes)
        has_chunker = any(n.agent_type == "chunker" for n in pipeline.nodes)

        if output.accuracy < DIAGNOSIS_ACCURACY_LOW and not has_validator:
            sev = (DIAGNOSIS_ACCURACY_LOW - output.accuracy) / DIAGNOSIS_ACCURACY_LOW
            diag.append(
                Diagnosis(
                    axis="accuracy",
                    severity=_clamp_severity(sev),
                    reason=f"accuracy={output.accuracy:.2f} below {DIAGNOSIS_ACCURACY_LOW:.1f} "
                    "and no validate-role node is present in the pipeline",
                    structural_root="no_validator",
                )
            )

        if (
            output.speed_normalized < DIAGNOSIS_SPEED_LOW
            and structural.parallelism_ratio < DIAGNOSIS_PARALLELISM_LOW
            and structural.critical_path_length >= DIAGNOSIS_BOTTLENECK_MIN_PATH
        ):
            sev = (DIAGNOSIS_SPEED_LOW - output.speed_normalized) / DIAGNOSIS_SPEED_LOW
            diag.append(
                Diagnosis(
                    axis="speed",
                    severity=_clamp_severity(sev),
                    reason=f"critical_path={structural.critical_path_length} edges with "
                    f"parallelism_ratio={structural.parallelism_ratio:.2f} — essentially serial",
                    structural_root="serial_bottleneck",
                )
            )

        if output.cost_normalized < DIAGNOSIS_COST_LOW and has_chunker:
            sev = (DIAGNOSIS_COST_LOW - output.cost_normalized) / DIAGNOSIS_COST_LOW
            diag.append(
                Diagnosis(
                    axis="cost",
                    severity=_clamp_severity(sev),
                    reason="cost is high and a chunker is present — over-segmentation is plausible",
                    structural_root="over_chunking",
                )
            )

        if structural.unused_outputs:
            diag.append(
                Diagnosis(
                    axis="structure",
                    severity=_clamp_severity(0.3 * len(structural.unused_outputs)),
                    reason=f"{len(structural.unused_outputs)} leaf node(s) produce output nobody reads: "
                    f"{structural.unused_outputs}",
                    structural_root="unused_outputs",
                )
            )

        if structural.redundant_agents:
            diag.append(
                Diagnosis(
                    axis="structure",
                    severity=_clamp_severity(0.4 * len(structural.redundant_agents)),
                    reason=f"{len(structural.redundant_agents)} structurally-duplicate agent node(s): "
                    f"{structural.redundant_agents}",
                    structural_root="redundant_agents",
                )
            )

        if structural.dependency_depth >= DIAGNOSIS_DEEP_CHAIN_MIN:
            sev = (structural.dependency_depth - DIAGNOSIS_DEEP_CHAIN_MIN + 1) / structural.dependency_depth
            diag.append(
                Diagnosis(
                    axis="structure",
                    severity=_clamp_severity(sev),
                    reason=f"dependency_depth={structural.dependency_depth} forms a long fragile chain",
                    structural_root="deep_chain",
                )
            )

        return diag


__all__ = ["Diagnostician"]
