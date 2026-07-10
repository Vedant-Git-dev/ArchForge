"""ArchForge CLI entry point.

`archforge run "describe the task"` — runs the end-to-end loop.

The CLI composes the same objects the library exposes
(`Architect`, `Engine`, `OutputEvaluator`, `ExperienceStore`). It is a
thin shell: all real behaviour lives in the modules it imports. Tests
should target those, not this file.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone

import click
from dotenv import load_dotenv

from .architect.designer import Architect
from .config import data_dir as _config_data_dir
from .core.experience import Experience
from .core.task import Task
from .evaluator.output import OutputEvaluator
from .evaluator.structural import StructuralEvaluator
from .executor.embeddings import get_default_embedding_client
from .executor.engine import Engine
from .executor.llm import get_default_llm_client
from .store.experience_store import ExperienceStore


# ─── helpers ────────────────────────────────────────────────────────────────


def _data_dir() -> str:
    return _config_data_dir()


def _experiences_dir() -> str:
    return os.path.join(_data_dir(), "experiences")


def _new_exp_id() -> str:
    return f"exp-{uuid.uuid4().hex[:12]}"


# ─── commands ───────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="archforge")
def cli() -> None:
    """ArchForge — self-learning multi-agent pipeline builder."""
    # Load a .env from the project root / cwd if present, so GEMINI_API_KEY and
    # the ARCHFORGE_* knobs work without manually exporting them. Existing
    # real env vars always win (load_dotenv does not override by default).
    load_dotenv()


@cli.command("run")
@click.argument("description", nargs=-1, required=True)
@click.option("--type", "task_type", default="general", help="Task type label.")
@click.option("--input", "input_text", default="", help="Optional initial input for the pipeline.")
@click.option("--data-dir", default=None, help="Override data directory.")
@click.option("--verbose", "-v", is_flag=True, help="Print per-node trace.")
@click.option(
    "--no-store",
    is_flag=True,
    help="Skip persisting the experience (useful for dry runs).",
)
def run_cmd(
    description: tuple[str, ...],
    task_type: str,
    input_text: str,
    data_dir: str | None,
    verbose: bool,
    no_store: bool,
) -> None:
    """Run a task end-to-end and print the pipeline output + score."""
    desc = " ".join(description).strip()
    if not desc:
        click.echo("Task description cannot be empty.", err=True)
        sys.exit(2)

    data_dir = data_dir or _data_dir()
    exp_dir = os.path.join(data_dir, "experiences")

    # 1. Build embedding + llm clients.
    embedder = get_default_embedding_client()
    llm = get_default_llm_client()

    # 2. Load experience store. Need the embedding dim first.
    dim = getattr(embedder, "dim", 384)
    store = ExperienceStore(dirpath=exp_dir, dim=dim)

    # 3. Build the Architect.
    architect = Architect(store=store, embedder=embedder)

    # 4. Compose a pipeline for the task.
    task = Task.new(desc, type=task_type)
    if input_text:
        task.metadata["initial_input"] = input_text

    click.echo(
        f"→ Architect: composing pipeline for {task.type!r} task...",
        err=True,
    )
    decision = architect.compose(task)
    if decision.triggered_from == "retrieval":
        click.echo(
            f"  Replaying pipeline from experience {decision.matched_experience_id} "
            f"(prior score {decision.matched_pipeline_score:.2f})",
            err=True,
        )
    else:
        click.echo("  No similar past run — building default pipeline.", err=True)

    # 5. Execute.
    click.echo("→ Executing pipeline...", err=True)
    engine = Engine(llm=llm)
    pipeline_input = input_text or task.description
    result = engine.run(decision.pipeline, task, outer_input=pipeline_input)

    if verbose:
        for trace in result.traces:
            click.echo(
                f"    [{trace.agent_type}] {trace.duration_seconds*1000:.0f} ms "
                f"({trace.total_tokens} tok)",
                err=True,
            )

    # 6. Evaluate.
    click.echo("→ Evaluating...", err=True)
    evaluator = OutputEvaluator(llm=llm)
    output = evaluator.evaluate(task, result)

    # 7. Compute composite + persist.
    exp = Experience(
        id=_new_exp_id(),
        task=task,
        pipeline=decision.pipeline,
    )
    exp.output = output
    # Structural metrics are a pure-topology calculation (no execution
    # data, no LLM). Phase 2 fills the field Phase 1 left zeroed; the
    # composite formula is unchanged (weights stay fixed until Phase 6).
    exp.structural = StructuralEvaluator().evaluate(decision.pipeline)
    exp.wall_time_seconds = result.wall_time_seconds
    exp.token_estimate = result.total_tokens
    exp.final_output = result.final_output
    exp.composite_score = exp.compute_composite()
    exp.timestamp = datetime.now(timezone.utc)

    if not no_store:
        store.append(exp)
        # Re-save index after append so the next invocation has it.
        try:
            store.save_index()
        except Exception as e:  # noqa: BLE001 — persistence failure shouldn't crash the run
            click.echo(f"  (warning: failed to save index: {e})", err=True)

    # 8. Print to user.
    click.echo("")
    click.echo("=== Output ===")
    click.echo(result.final_output or "(empty)")
    click.echo("")
    click.echo("=== Scores ===")
    summary = {
        "accuracy": round(output.accuracy, 3),
        "completeness": round(output.completeness, 3),
        "speed": round(output.speed_normalized, 3),
        "cost": round(output.cost_normalized, 3),
        "composite": round(exp.composite_score, 3),
        "structural": round(exp.structural.score, 3),
        "critical_path": exp.structural.critical_path_length,
        "parallelism": round(exp.structural.parallelism_ratio, 3),
        "wall_time_seconds": round(result.wall_time_seconds, 3),
        "tokens": result.total_tokens,
        "trigger": decision.triggered_from,
    }
    click.echo(json.dumps(summary, indent=2))


@cli.command("inspect")
@click.option("--data-dir", default=None, help="Override data directory.")
@click.option("--last", "last_n", default=5, help="Show the most recent N experiences.")
def inspect_cmd(data_dir: str | None, last_n: int) -> None:
    """List recent experiences (for debugging / manual inspection)."""
    data_dir = data_dir or _data_dir()
    exp_dir = os.path.join(data_dir, "experiences")
    embedder = get_default_embedding_client()
    store = ExperienceStore(dirpath=exp_dir, dim=getattr(embedder, "dim", 384))
    store.recompute_embeddings(embedder)
    all_exps = store.all()
    if not all_exps:
        click.echo("No experiences yet.")
        return
    recent = sorted(all_exps, key=lambda e: e.timestamp, reverse=True)[:last_n]
    click.echo(f"Total experiences: {len(all_exps)}")
    for e in recent:
        click.echo(
            f"  - {e.id} [{e.task.type}] composite={e.composite_score:.2f} "
            f"task={e.task.description[:80]!r}"
        )


if __name__ == "__main__":
    cli()
