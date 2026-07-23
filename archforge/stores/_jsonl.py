"""Tiny shared filesystem helpers for the stores.

Kept deliberately small and dependency-free. Atomic writes (write to a temp file
then `os.replace`) avoid a half-written file if the process dies mid-write — so
a crash leaves either the old record or the new one, never a truncated one. This
underpins the "fail closed, incumbent untouched" property (spec §6 E10 recovery).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record as newline-delimited JSON, creating parent dirs."""

    ensure_dir(path.parent)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    # O_APPEND writes are atomic for small lines (< PIPE_BUF) on POSIX.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")
        fh.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all newline-delimited JSON records (empty file -> [])."""

    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_text_atomic(path: Path, text: str) -> None:
    """Write text atomically — a crash leaves the prior file or the new one."""

    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_text(path: Path) -> str | None:
    """None if the file does not exist; its contents otherwise."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return fh.read()
