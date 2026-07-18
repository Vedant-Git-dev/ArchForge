"""reader primitive — deterministic file/text ingestion.

The reader was the first LLM primitive (its old job: "take raw input and
produce a structured, normalised representation"), but cleaning/normalising
input never needed a model — and it could never read a *file*, only the
engine's ``outer_input`` string. It is now ``kind="deterministic"``: a pure
Python transform that

  - reads any provided file via a graceful decoder registry (text, JSON,
    CSV/TSV, plus import-guarded pandas for xlsx/parquet). No pdf/docx libs
    ship, so those degrade to a "not text-extractable" note rather than
    crashing — the never-raise contract. "Any file type": unknown
    extensions fall through to text decode, so ``.md``/``.py``/``.log``/
    ``.yaml``/code just work.
  - or ingests inline text from ``input["input"]``.
  - never raises: a missing file, a decode failure, a binary blob all land
    in ``notes`` with ``ok`` staying ``True`` (soft-fail channel, spec §6),
    mirroring ``regex_extractor``.

The output is ``{"text", "source", "notes"}``. The old ``tokens_estimate``
field is **gone**: nothing in ``archforge`` read it (the cost evaluator keys
on ``AgentResult.total_tokens``, the engine's bookkeeping — not this output
field), so it was vestigial LLM-era baggage. ``output_schema`` only ever
required ``text``, so dropping the field is purely subtractive.

Deterministic → zero-token ``AgentResult`` (``total_tokens=0``, ``model="",
``cost=0.0``), same as every deterministic agent.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from ...core.primitive import Primitive
from .base import AgentCallable, AgentResult

# Optional, import-guarded binary/structured decoders. ArchForge ships no PDF
# or spreadsheet libs by default; when one happens to be installed we use it,
# otherwise the file degrades to a "not text-extractable" note — never a raise.
try:  # pandas: xlsx / parquet → CSV text
    import pandas as _pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # noqa: BLE001 — optional dep
    _pd = None
    _HAS_PANDAS = False

try:  # pypdf: pdf → text (not a declared dep; used opportunistically)
    from pypdf import PdfReader as _PdfReader  # type: ignore
    _HAS_PYPDF = True
except Exception:  # noqa: BLE001 — optional dep
    _PdfReader = None
    _HAS_PYPDF = False

# Extensions we won't even try to text-decode — short-circuit to a binary note
# so a 50MB image doesn't get latin-1'd into a garbage string downstream.
_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif",
    ".pdf",  # pdf has its own decoder path below (when pypdf is present)
    ".zip", ".gz", ".tar", ".bz2", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".ogg",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".sqlite", ".db",
}

READER_SPEC = Primitive(
    name="reader",
    level=0,
    role="ingest",
    kind="deterministic",
    system_prompt="",  # inert for the deterministic kind
    input_schema={
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "path": {"type": "string"},
            "context": {"type": "object"},
        },
    },
    output_schema={"type": "object", "required": ["text"]},
    params={},
)


# ─── decoder registry ────────────────────────────────────────────────────────


def _decode_bytes(path: Path) -> tuple[str, list[str]]:
    """Return (text, notes) for a file's raw bytes. Never raises — on any
    structural failure the caller falls back to raw text decode.
    """
    notes: list[str] = []
    suffix = path.suffix.lower()

    # Structured text decoders — produce nicer downstream output than raw text.
    try:
        if suffix == ".json":
            data = json.loads(path.read_bytes().decode("utf-8"))
            text = json.dumps(data, indent=2, ensure_ascii=False)
            notes.append("decoded as JSON (re-indented)")
            return text, notes
        if suffix in (".csv", ".tsv"):
            delim = "\t" if suffix == ".tsv" else ","
            raw = path.read_bytes().decode("utf-8", errors="replace")
            rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
            text = "\n".join(" | ".join(r) for r in rows)
            notes.append(f"decoded as {suffix[1:].upper()} ({len(rows)} rows)")
            return text, notes
    except Exception as exc:  # noqa: BLE001 — fall through to text decode
        notes.append(f"{suffix} decode failed ({type(exc).__name__}); falling back to text")

    # Optional binary/structured decoders — only run when the lib is present.
    if suffix in (".xlsx", ".xls", ".xlsm", ".parquet") and _HAS_PANDAS:
        try:
            if suffix == ".parquet":
                df = _pd.read_parquet(path)
            else:
                df = _pd.read_excel(path)
            text = df.to_csv(index=False)
            notes.append(f"decoded via pandas as CSV ({len(df)} rows)")
            return text, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"pandas decode failed ({type(exc).__name__}); trying text")

    if suffix == ".pdf" and _HAS_PYPDF:
        try:
            reader = _PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            notes.append("decoded via pypdf")
            return text, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"pypdf decode failed ({type(exc).__name__}); not text-extractable")
            return "", notes

    if suffix == ".pdf" and not _HAS_PYPDF:
        notes.append("pdf: pypdf not installed, not text-extractable")
        return "", notes

    # Known-binary extensions short-circuit before a garbage latin-1 decode.
    if suffix in _BINARY_SUFFIXES:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        notes.append(f"binary file ({suffix}), not text-extractable")
        if size >= 0:
            notes[-1] += f" ({size} bytes)"
        return "", notes

    # Default: text decode. utf-8 first; latin-1 never fails (maps every byte)
    # so the fallback only fires for genuinely non-utf-8 encodings.
    raw = path.read_bytes()
    # Null-byte sniff: even an unknown extension with NULs is binary.
    if b"\x00" in raw[:8192]:
        notes.append(f"binary file (null bytes detected, {len(raw)} bytes), not text-extractable")
        return "", notes
    try:
        text = raw.decode("utf-8")
        notes.append("encoding: utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        notes.append("encoding: latin-1 (utf-8 failed)")
    return text, notes


def _read_file(path: Path) -> tuple[str, list[str]]:
    """Read a file path → (text, notes). Never raises."""
    notes: list[str] = []
    try:
        size = path.stat().st_size
    except OSError as exc:
        notes.append(f"file not readable: {exc.__class__.__name__}: {path}")
        return "", notes
    notes.append(f"file: {path.name} ({size} bytes)")
    return _decode_bytes(path)


# ─── the agent ───────────────────────────────────────────────────────────────


def reader(input: Mapping[str, Any], ctx=None) -> AgentResult:
    """Deterministic file/text reader.

    ``ctx`` is the ``AgentContext``; this agent ignores ``ctx.llm`` (the
    zero-token contract). Never raises — every failure (missing file, binary
    blob, decode error) is captured in ``output["notes"]`` with ``ok=True``.

    Input contract (backward-compatible with the engine's root payload
    ``{"input": outer_input, ...}``):
      - ``input["path"]`` → explicit file path (takes precedence).
      - else ``input["input"]`` → if it points at an existing file, read it;
        otherwise treat it as inline text (empty → ``source="user"``).
    """
    t0 = time.perf_counter()
    notes: list[str] = []
    text = ""
    source = "user"

    raw_path = input.get("path")
    raw_input = input.get("input")

    # Explicit path wins. A missing file is a soft-fail, never an exception.
    if raw_path is not None and str(raw_path).strip():
        path = Path(os.path.expanduser(str(raw_path)))
        source = f"file:{path.name}"
        if path.is_file():
            text, file_notes = _read_file(path)
            notes.extend(file_notes)
        else:
            notes.append(f"file not found: {path}")

    elif raw_input is not None and str(raw_input).strip():
        candidate = str(raw_input).strip()
        # Auto-detect a file path embedded in the engine's outer_input — this
        # is "reader can read any file provided": passing a path as the run
        # input reads it, passing prose ingests the prose.
        maybe_path = Path(os.path.expanduser(candidate))
        if maybe_path.is_file():
            source = f"file:{maybe_path.name}"
            text, file_notes = _read_file(maybe_path)
            notes.extend(file_notes)
        else:
            source = "inline"
            text = str(raw_input)
            notes.append(f"inline text ({len(text)} chars)")
    else:
        notes.append("empty input")

    output: dict[str, Any] = {"text": text, "source": source, "notes": notes}

    return AgentResult(
        output=output,
        text="",  # deterministic: no raw LLM text
        prompt_tokens=0,
        completion_tokens=0,
        model="",
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        cost=0.0,
        ok=True,
        error=None,
    )


READER_AGENT: AgentCallable = reader


__all__ = ["READER_SPEC", "READER_AGENT", "reader"]
