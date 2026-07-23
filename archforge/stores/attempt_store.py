"""AttemptStore — append-only Attempts (spec §3), grouped by parent spec_id.

Layout: `attempts/<parent_spec_id>.jsonl`, one Attempt per line. Grouping by
parent makes the Architect's dedup query (E7) a cheap linear scan of one file:
"has this (parent, kind, target) already been tried and rejected/rolled-back?"

The store assigns `attempt_id` (its own content hash, stable + collision-free)
on first append so Attempts are individually addressable like Specs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from archforge.models import Attempt, Verdict
from archforge.stores._jsonl import append_jsonl, read_jsonl

# Verdicts that should block re-proposing the same (parent, kind, target) —
# the Architect consults `match` to skip known dead ends (spec E7).
_BLOCKING_VERDICTS: frozenset[Verdict] = frozenset({Verdict.REJECTED, Verdict.ROLLED_BACK})


class UnknownAttemptError(KeyError):
    """A requested attempt_id is not in the store."""


class AttemptStore:
    """Append-only Attempt memory, grouped per parent spec_id."""

    ROOT_SUBDIR = "attempts"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.attempts_dir = self.root / self.ROOT_SUBDIR

    # ----------------------------------------------------------------- append
    def append(self, attempt: Attempt) -> str:
        """Persist `attempt`, assign its `attempt_id` if unset, return the id."""

        if attempt.attempt_id is None:
            attempt.attempt_id = self._compute_id(attempt)
        existing = self.get(attempt.attempt_id)
        if existing is not None:
            return attempt.attempt_id  # idempotent: same Attempt already persisted
        append_jsonl(
            self._path_for_parent(attempt.parent_spec_id),
            attempt.model_dump(mode="json"),
        )
        return attempt.attempt_id

    # ----------------------------------------------------------------- read
    def get(self, attempt_id: str) -> Attempt | None:
        for att in self.all():
            if att.attempt_id == attempt_id:
                return att
        return None

    def require(self, attempt_id: str) -> Attempt:
        att = self.get(attempt_id)
        if att is None:
            raise UnknownAttemptError(attempt_id)
        return att

    def for_parent(self, parent_spec_id: str) -> list[Attempt]:
        return [
            Attempt.model_validate(rec)
            for rec in read_jsonl(self._path_for_parent(parent_spec_id))
        ]

    def all(self) -> list[Attempt]:
        records: list[Attempt] = []
        if not self.attempts_dir.exists():
            return records
        for path in sorted(self.attempts_dir.glob("*.jsonl")):
            for rec in read_jsonl(path):
                records.append(Attempt.model_validate(rec))
        return records

    # ----------------------------------------------------------------- dedup
    def match(self, parent_spec_id: str, kind: str, target: str) -> list[Attempt]:
        """Prior attempts on the same (parent, change.kind, change.target)."""

        return [
            att
            for att in self.for_parent(parent_spec_id)
            if att.change.kind.value == kind and att.change.target == target
        ]

    def blocking(
        self, parent_spec_id: str, kind: str, target: str
    ) -> list[Attempt]:
        """Prior attempts that should stop the Architect re-proposing this change (E7)."""

        return [
            att for att in self.match(parent_spec_id, kind, target)
            if att.verdict in _BLOCKING_VERDICTS
        ]

    # ----------------------------------------------------------------- paths
    def _path_for_parent(self, parent_spec_id: str) -> Path:
        return self.attempts_dir / f"{parent_spec_id}.jsonl"

    @staticmethod
    def _compute_id(attempt: Attempt) -> str:
        payload = json.dumps(attempt.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]
