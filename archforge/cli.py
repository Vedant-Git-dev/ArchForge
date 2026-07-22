"""ArchForge command-line interface.

Phase 0 skeleton: subcommands are wired so `--help` is meaningful; each
subcommand is a stub that is filled in over later phases (notably Phase 9).
"""

from __future__ import annotations

import argparse
import sys

PROG = "archforge"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="ArchForge — a self-improving meta-layer over multi-agent systems.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("evolve", help="run one Propose-Evaluate-Commit cycle (Phase 9)")
    sub.add_parser("evolve-loop", help="repeat evolve until budget cap or plateau (Phase 9)")
    sub.add_parser("approve", help="drain the structural-change approval queue (Phase 9)")
    sub.add_parser("status", help="print the incumbent Spec and lineage (Phase 9)")
    sub.add_parser("report", help="print aggregate deltas across attempts (Phase 9)")

    lint_p = sub.add_parser("lint", help="run the Spec Linter on a JSON Spec file")
    lint_p.add_argument("path", help="path to a Spec JSON file")
    return parser


def _cmd_lint(path: str) -> int:
    # Minimal, dependency-light implementation; the full CLI surface lands in Phase 9.
    import json
    from pathlib import Path

    from archforge.lint import lint
    from archforge.models import Spec

    spec = Spec.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    errors = lint(spec)
    if not errors:
        print("OK: spec is structurally valid.")
        return 0
    for e in errors:
        loc = f" [{e.location}]" if e.location else ""
        print(f"{e.code}{loc}: {e.message}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "lint":
        return _cmd_lint(args.path)
    # Remaining subcommands land in later phases.
    print(f"[stub] '{args.command}' is not implemented yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
