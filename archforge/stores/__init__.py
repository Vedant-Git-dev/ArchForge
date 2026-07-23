"""Immutable, append-only filesystem stores for ArchForge.

SpecStore   — content-addressed Pipeline Specs (one file each) + active pointer.
TraceStore  — append-only Traces (JSONL, one file per spec_id).
AttemptStore— append-only Attempts (JSONL, one file per parent_spec_id).

Design notes (spec §3, §4):
  * No store *mutates* a committed Spec object; promotion and rollback only move
    the `active` pointer and the `archived` set. This realizes immutability (I2)
    and makes rollback a pointer swap (E6/I3).
  * `SpecStore.set_active` is the ONLY mutator of the incumbent pointer (I1 —
    single source of truth).
  * All JSONL is one record per line (newline-delimited JSON) for simple append.
"""

from __future__ import annotations

from archforge.stores.attempt_store import AttemptStore
from archforge.stores.spec_store import SpecStore
from archforge.stores.trace_store import TraceStore

__all__ = ["SpecStore", "TraceStore", "AttemptStore"]
