# Phase 1 — Skeleton (End-to-End Working)

> Task in → pipeline built → pipeline run → scored → stored → retrieved next time.

Phase 1 is the **retrieve-and-replay** skeleton. No diagnosis, no
intervention, no parallelism, no structural scoring. Just a closed loop
that proves the architecture compiles end-to-end and that **retrieval
learning works** — a repeated task reuses the pipeline that solved it
before.

This document is the authoritative record of what shipped: the scope, the
data model, the call graph, the file list, the tests, and the verify
checklist trace. If you are picking up Phase 2, read this first.

---

## 1. Scope (what was built, what was deliberately cut)

### Built
| Layer | What | Status |
|---|---|---|
| Core data model | `Task`, `PipelineDAG`, `AgentNode`, `Edge`, `Experience`, `Primitive`, `Diagnosis`, `OutputScores`, `StructuralScores` | ✅ shipped (Diagnosis/StructuralScores are schema-only placeholders for Phase 2) |
| Backends | `GroqLLMClient` (OpenAI-compat), `FakeLLMClient` (deterministic stubs), `EmbeddingClient` (MiniLM-L6-v2, 384d), `HashingEmbeddingClient` (zero-dep fallback) | ✅ shipped |
| Primitives | 6 base agents: `reader`, `chunker`, `classifier`, `summarizer`, `fact_checker`, `writer` | ✅ shipped with YAML metadata + in-code impl |
| Executor | Linear topological-sort DAG engine; refuses cycles; merges predecessor outputs | ✅ shipped |
| Evaluator | Output quality only: `accuracy` (LLM judge) + `speed_normalized` + `cost_normalized` → composite | ✅ shipped |
| Store | JSONL file + numpy kNN task index; persists, reloads, dedup by id | ✅ shipped |
| Architect | Embed task → kNN task index → replay best match above threshold OR build default linear pipeline | ✅ shipped |
| CLI | `archforge run`, `archforge inspect` | ✅ shipped |
| Tests | 35 tests across 7 files | ✅ all passing in ~0.1 s |

### Deliberately cut (slots reserved in schema for later phases)
- **Structural scoring** — `StructuralScores` exists with zero defaults; Phase 2 fills it.
- **Diagnosis generation** — `Diagnosis` dataclass exists and `Experience.diagnoses` is an empty list; Phase 2 populates via the evaluator.
- **Intervention library** — `Experience.interventions_applied` / `interventions_helped` fields exist; Phase 2 adds the library + matching.
- **Parallel execution** — engine walks topo order sequentially; the `Engine.run` signature is already shaped for fan-out/fan-in (Phase 2).
- **Pipeline embedding + dual retrieval** — `PipelineDAG.embedding` field reserved; Phase 3.
- **Templates, primitive discovery, weight learning, budgeting, API** — Phases 4 / 5 / 6.

The schema was designed to be **forward-compatible**: later phases populate
fields that already exist, they don't migrate the format.

---

## 2. End-to-end data flow

```
CLI: archforge run "describe the task"
  │
  └─► main.py:run_cmd
        │
        ├─ embedder = get_default_embedding_client()     # MiniLM or Hashing fallback
        ├─ llm      = get_default_llm_client()            # Groq if GROQ_API_KEY else Fake
        ├─ store    = ExperienceStore(dirpath=…, dim)
        ├─ task     = Task.new(description, type)
        │
        └─► Architect.compose(task)
              │
              ├─ embedding   = embedder.embed_one(text)        # text = f"{type}: {description}"
              ├─ store.recompute_embeddings(embedder)          # fill missing task embeddings on reload
              ├─ hits         = store.search_by_task(emb, k=5) # brute-force cosine kNN
              ├─ replayable   = [h for h in hits if h.score >= 0.5]
              │
              ├─ if replayable:  pick max composite_score → PipelineDAG.from_dict(exp.pipeline)
              └─ else:           PipelineDAG.linear(DEFAULT_PIPELINE_AGENTS)
              │
              └─ ArchitectureDecision(pipeline, triggered_from=[retrieval|default], …)
        │
        └─► Engine.run(pipeline, task, outer_input)
              │
              ├─ topo_order()  # raises on cycle
              ├─ for node in order:
              │     payload = _build_node_input(node, predecessors' outputs, task, outer_input)
              │     agent   = pool.get(node.agent_type)
              │     result  = agent.run(payload, llm)        # → JSON via call_llm_json
              │     record NodeTrace
              ├─ final_output = writer node's `output` field (or leaf fallback)
              └─ PipelineResult(final_output, traces, tokens, wall_time)
        │
        └─► OutputEvaluator.evaluate(task, result)
              │
              ├─ accuracy, completeness = LLM judge task vs output (temperature=0)
              ├─ speed_normalized       = linear decay 5 s → 60 s, lower is better
              ├─ cost_normalized        = linear decay 500 → 8000 tokens, lower is better
              └─ OutputScores(…, user_rating=None)
        │
        └─► Experience(...)
              ├─ compute_composite()  # 0.5·acc + 0.25·speed + 0.25·cost
              ├─ store.append(exp)    # writes one JSONL line
              └─ store.save_index()   # embeddings.npy + index_ids.json
        │
        └─► stdout: final output + scores JSON
```

The single most important loop invariant: **the 6th run of a similar task
must replay, not rebuild.** Everything else (scoring, persistence) is in
service of that.

---

## 3. Data model

All dataclasses live in `archforge/core/`. Every one serialises via
`to_dict`/`from_dict` (needed because the experience store round-trips
through JSON). Embeddings are deliberately **not** serialised into the
JSONL — they live in the numpy sidecar — so the JSONL stays human-readable.

### `Task` (`core/task.py`)
| field | type | notes |
|---|---|---|
| id | `str` | `task-<12hex>` |
| description | `str` | the task text |
| type | `str` | free-form label: `"summary"`, `"analysis"`, …; never an enum |
| metadata | `dict` | arbitrary (domain, input_size, …) |
| embedding | `list[float]` | 384d, filled by EmbeddingClient, blank in JSONL (sidecar stores it) |

`embedding` is not persisted to JSONL; `to_dict` records only `has_embedding`. The store recomputes from `f"{type}: {description}"` lazily.

### `PipelineDAG` (`core/pipeline.py`)
| field | purpose |
|---|---|
| `nodes: list[AgentNode]` | the agents |
| `edges: list[Edge]` | data flow (directed) |
| `fingerprint: dict` | structural descriptor (Phase 1: minimal) |
| `embedding: list[float]` | reserved for Phase 3 pipeline index |

Helpers: `linear(agent_types)` (constructor), `topo_order()` (Kahn's algorithm, raises on cycle), `has_cycle()`, `roots()/leaves()/successors()/predecessors()`, `compute_fingerprint()`, `content_hash()` (stable topology hash for dedup/subgraph mining), `to_dict()/from_dict()`.

`content_hash` is **id-blind**: two pipelines with the same shape hash identically, even though `PipelineDAG.id` is random per construction. This is what makes "have I seen this topology before" queries cheap later.

### `AgentNode` / `Edge`
- `AgentNode`: `{id, agent_type, level (0=base), config}` — `agent_type` is the primitive name looked up at execution.
- `Edge`: `{source, target, data_type}` — `data_type` is a free-form label (`"any"` by default).

### `Experience` (`core/experience.py`)
One row per pipeline run. Stored as one JSONL line.

| block | fields | Phase 1 status |
|---|---|---|
| task link | `task: Task` | populated |
| pipeline | `pipeline: PipelineDAG`, `pipeline_hash: str` | populated (hash auto-filled in `__post_init__`) |
| output | `output: OutputScores` (`accuracy`, `completeness`, `speed_normalized`, `cost_normalized`, `user_rating`) | populated |
| structural | `structural: StructuralScores` (pipeline_length, critical_path, parallelism_ratio, redundant_agents, unused_outputs, dependency_depth, score) | **zero defaults — Phase 2** |
| composite | `composite_score: float` | populated |
| diagnosis | `diagnoses: list[Diagnosis]` | **empty list — Phase 2** |
| interventions | `interventions_applied: list[str]`, `interventions_helped: dict[str,bool]` | **empty — Phase 2** |
| meta | `wall_time_seconds`, `token_estimate`, `final_output`, `timestamp`, `generation` | populated |

Composite formula (Phase 1 — fixed weights, Phase 6 makes these per-task-type learnable):

```python
composite = 0.50 * accuracy + 0.25 * speed_normalized + 0.25 * cost_normalized
```

Implementation note: `default_weights()` is a `@staticmethod`, **not** a class-level dict field. Python's `@dataclass` rejects mutable dict defaults ("mutable default `<class 'dict'>` is not allowed"). This was a real bug hit during scaffold — recorded so Phase 2 doesn't repeat it.

### `Primitive` (`core/primitive.py`)
The unit of composition. `name`, `level` (0=base, 1+=evolved), `role` (`ingest`|`transform`|`analyze`|`validate`|`generate`|`compose`), `system_prompt`, `input_schema`, `output_schema`, plus evolved-primitive provenance (`source_subgraph`, `fusing_prompt`, `validation_score`, `created_from_n_experiences`, `created_at`, `can_unwrap`) — all `None`/empty for Phase 1 base primitives.

### `Diagnosis` / `OutputScores` / `StructuralScores`
- `Diagnosis`: `{axis, severity, reason, structural_root}` — schema-ready, **empty list in every Phase 1 experience**.
- `OutputScores`: 5 output surfaces ({accuracy, completeness, speed_normalized, cost_normalized, user_rating}); Phase 1 fills 3.
- `StructuralScores`: 6 structural metrics + `score`; Phase 1 leaves all zero.

---

## 4. Backends

### 4.1 LLM — `executor/llm.py`

A single `LLMClient` protocol with two implementations:
- `GroqLLMClient` — uses `groq` SDK; model defaults to `llama-3.1-8b-instant` (env: `ARCHFORGE_GROQ_MODEL`). Returns `LLMResult{text, prompt_tokens, completion_tokens, model}`. `groq_cost_usd()` gives USD from a small pricing table (used by Phase 1's cost normalisation only indirectly — cost is token-count based today).
- `FakeLLMClient` — deterministic; cycles through a script of responses; records a `call_log` for tests. Falls back to generic JSON when a script slot is `None`.

`get_default_llm_client()` picks Groq if `GROQ_API_KEY` is set, else Fake. **Primitives never import Groq directly** — they call `llm.chat(system, user)`. That's the seam Phase 2+ tests rely on.

### 4.2 Embeddings — `executor/embeddings.py`

- `EmbeddingClient` — wraps `sentence-transformers/all-MiniLM-L6-v2` (384d), lazy-loaded, L2-normalised so cosine == dot product. Env override `ARCHFORGE_EMBEDDING_MODEL`.
- `HashingEmbeddingClient` — zero-dependency 256d fallback via token hashing. **Not semantically meaningful** — only used for tests / offline runs that don't want the ~80 MB model download.

`get_default_embedding_client(force_name=…)` returns MiniLM by default, hashing if `force_name="hashing"` or if MiniLM fails to load. Tests pass `force_name="hashing"` explicitly via the `hashing_embedder` fixture so the suite runs in 0.1 s with no network.

---

## 5. Primitives — `executor/agents/`

### 5.1 Contract — `base.py`

All primitives share one shape:
```
run(input: dict, llm: LLMClient) -> AgentResult{output, text, prompt_tokens, completion_tokens, model}
```
The shared heart is `call_llm_json(llm, system_prompt, payload, max_tokens, temperature)`:
1. `json.dumps(payload)` → user message
2. `llm.chat(system=…, user=…, **opts)` → `LLMResult`
3. `_extract_json(text)` → dict (tolerates markdown fences, bare prose, partial JSON; on failure returns `{"raw_text": text, "text": text}`).
4. Wrap as `AgentResult`.

`_extract_json` is **deliberately forgiving** — a malformed LLM response must not crash the whole pipeline. Test `test_primitive_handles_malformed_json_gracefully` codifies this.

### 5.2 The 6 base primitives

| primitive | role | i/o contract | what it does (domain-agnostic) |
|---|---|---|---|
| `reader` | ingest | in `{input}` → out `{text, tokens_estimate, source, notes}` | normalise raw input into structured substrate |
| `chunker` | transform | in `{text}` → out `{chunks[], strategy, warnings}` | segment input; strategy adapts to shape (prose / code / table / mixed / single) |
| `classifier` | analyze | in `{items or chunks or [text]}` → out `{categories[], summary, ambiguous_items}` | dynamic taxonomy (never hardcoded) |
| `summarizer` | analyze | in `{input}` → out `{summary, style, key_points?, length_chars}` | abstractive / structured / hierarchical, picked from input shape |
| `fact_checker` | validate | in `{claims, evidence}` → out `{verdicts[], unverified_claims, evidence_gaps}` | conservative verdicts; never invents evidence |
| `writer` | generate | in `{task, evidence}` → out `{output, format, satisfies_task, open_questions}` | terminal; the `output` field is what the pipeline returns to the user |

Each has a YAML descriptor in `data/primitives/*.yaml` (name, level, role, schemas, description). The YAML currently describes the same primitive the in-code class implements; Phase 4 template work will make YAML-driven agents first-class.

### 5.3 Registry — `executor/agents/registry.py`

- `PrimitivePool` late-binds built-ins so YAML overrides win.
- `BUILTIN_CONSTRUCTORS` maps the 6 base names → constructor lambdas.
- `get(name)` loads YAMLs once (lazy), then instantiates the in-code class.
- `primitives()` materialises every built-in for a complete catalogue.
- `default_data_dir()` resolves `<project_root>/data` unless `ARCHFORGE_DATA_DIR` is set (used by tests + CLI).

---

## 6. Execution engine — `executor/engine.py`

`Engine(llm, pool=None).run(pipeline, task, outer_input="") → PipelineResult`

1. **Cycle guard** — `pipeline.has_cycle()` → raise immediately.
2. **Topo order** — Kahn's; preserves original node order among ready nodes.
3. **Sequential dispatch** — one node at a time (Phase 2 adds parallel branches).
4. **Input assembly** — `_build_node_input()`:
   - Single predecessor → predecessor's output fields are merged **at top level** so e.g. `chunker` reads `.text` directly.
   - Multiple predecessors → wrapped under `"predecessors": {id: output}`.
   - Always adds `task`, `task_type`, `input` (outer), `context` (task metadata).
5. **Trace** — each node records `NodeTrace{agent_type, duration_seconds, prompt_tokens, completion_tokens, output, text}`.
6. **Final output** — `_extract_final_output()`: writer's `output` field; falls back to leaf's `text`/`output`; empty string if nothing.

`PipelineResult` aggregates traces, totals, wall time. **Engine is pipeline-agnostic** — it doesn't know what agents do, only how to route data along edges (per plan §Execution Engine).

---

## 7. Evaluator — `evaluator/output.py`

`OutputEvaluator(llm).evaluate(task, result, user_rating=None) → OutputScores`

| surface | how computed | range |
|---|---|---|
| accuracy | LLM judge (system prompt in module); asks "does output satisfy the task?" | [0, 1] |
| completeness | same judge call | [0, 1] |
| speed_normalized | `_normalize(wall_time, low=5 s, high=60 s, lower_is_better=True)` | [0, 1] (1 = under 5 s, 0 = over 60 s) |
| cost_normalized | `_normalize(total_tokens, low=500, high=8000, lower_is_better=True)` | [0, 1] |
| user_rating | passed through if given, else None | [0, 1] or None |

`_normalize` is a linear decay between `low` (full credit) and `high` (no credit) — monotonic, clamped. Used for both speed and cost. **Phase 1 does not weight by USD** — cost is token-count based; `groq_cost_usd` exists in `llm.py` for future use.

Judge temperature is 0 for determinism; on parse failure, the evaluator returns a neutral `{accuracy: 0.5, completeness: 0.5}` rather than crashing (so one bad judge response doesn't poison the experience store).

---

## 8. Experience store — `store/`

Two files under `<data_dir>/experiences/`:

| file | holds |
|---|---|
| `experiences.jsonl` | one `Experience.to_dict()` per line, append-only |
| `embeddings.npy` | `(n × 384)` float32 matrix, row i ↔ experience i |
| `index_ids.json` | list of experience ids in matrix row order |

### `TaskIndex` (`store/task_index.py`)
- **Brute-force cosine kNN** over an in-memory numpy matrix. Embeddings are pre-normalised so cosine == dot product.
- `add(id, vec)`, `search(query, k)` (uses `argpartition` for O(n) top-k then sorts the slice), `save(dir)`, `TaskIndex.load(dir, dim)`.

This is intentionally simple — Phase 3 may swap for HNSW; the `TaskIndex` API is the seam.

### `ExperienceStore` (`store/experience_store.py`)
- Constructed with `(dirpath, dim)`. Lazy-loads JSONL + index on init.
- `append(exp)` — dedups by id (in-place replace if seen); writes one line.
- `recompute_embeddings(embedder)` — fills any experience whose `task.embedding` is empty (i.e. loaded from JSONL on a fresh process). **This is the trick that keeps embeddings out of the JSONL while still being reconstructable.**
- `search_by_task(query, k, min_score)` → `list[ScoredHit(experience, score, rank)]`.

`ScoredHit.score` is raw cosine similarity; the Architect applies its own threshold (Phase 1: 0.5).

**Why JSON + numpy, not SQLite (decided 2026-07-03):**
- Experience count stays <50 K in Phase 1–3; file load + brute-force kNN is ≤50 ms at that scale.
- JSONL is inspectable (`cat`) and diffable — SQLite needs the CLI + a schema to read.
- The seam (`TaskIndex` + `ExperienceStore` interfaces) is shaped so a future `SqliteStore` can drop in without touching callers.
- Revisit when count >50 K **or** Phase 4 templates need joins across experiences + interventions (see `phase1` memory: `phase1-learnings-2026-07-03`).

---

## 9. Architect — `architect/designer.py`

`Architect(store, embedder, replay_similarity_threshold=0.5, top_k=5).compose(task) → ArchitectureDecision`

```
1. embedding = embedder.embed_one(f"{task.type}: {task.description}")
2. store.recompute_embeddings(embedder)        # cheap no-op if all populated
3. hits = store.search_by_task(embedding, k=5)
4. replayable = [h for h in hits if h.score >= 0.5 and h.experience.pipeline.nodes]
5. if replayable:
       best = max(replayable, key=composite_score)
       return replay(PipelineDAG.from_dict(best.pipeline), "retrieval", best.id, best.score)
   else:
       return default(PipelineDAG.linear(DEFAULT_PIPELINE_AGENTS), "default")
```

`DEFAULT_PIPELINE_AGENTS = [reader, chunker, classifier, summarizer, fact_checker, writer]`.

`ArchitectureDecision` carries **provenance**: `triggered_from`, `matched_experience_id`, `matched_pipeline_score`, the full `candidates` list, and `task_embedding_dim`. The CLI prints `triggered_from` and the integration test asserts it flips from `"default"` → `"retrieval"` on the 6th run.

Phase 1 does **no mutation**: it either replays verbatim or builds the default. Phase 2's diagnosed interventions layer on top of this same compose() flow.

---

## 10. CLI — `main.py`

`archforge run "..." [--type general] [--input …] [--data-dir …] [-v] [--no-store]`
- Composes `Architect → Engine → OutputEvaluator`, then `store.append` + `save_index`.
- `--verbose` prints one line per node with ms + token count.
- `--no-store` runs without persisting (dry runs).
- Output to stdout: the writer's `output`, then a `=== Scores ===` JSON block (accuracy, completeness, speed, cost, composite, wall_time, tokens, trigger).

`archforge inspect [--data-dir …] [--last 5]`
- Loads the store, recomputes embeddings, prints total count + the most recent N experiences with composite score + truncated task text.

Environment knobs:
- `GROQ_API_KEY` — enables real Groq calls; absence → `FakeLLMClient` (generic JSON stubs).
- `ARCHFORGE_EMBEDDING_MODEL` — defaults to `all-MiniLM-L6-v2`; set to `hashing` for the zero-dep fallback.
- `ARCHFORGE_DATA_DIR` — override the data root (default `<project_root>/data`).
- `ARCHFORGE_GROQ_MODEL` — override the Groq model id.

---

## 11. File inventory

```
pyproject.toml                # entrypoint 'archforge', deps groq/sentence-transformers/numpy/click/pyyaml
.gitignore
README.md
plan.md

archforge/
  __init__.py
  main.py                     # click CLI: run, inspect
  core/
    task.py                   # Task
    pipeline.py               # PipelineDAG, AgentNode, Edge, topo_order, fingerprint, content_hash
    experience.py             # Experience, OutputScores, StructuralScores, Diagnosis, compute_composite
    primitive.py              # Primitive
  architect/
    designer.py               # Architect, ArchitectureDecision, DEFAULT_PIPELINE_AGENTS
  executor/
    llm.py                    # LLMClient protocol, GroqLLMClient, FakeLLMClient, groq_cost_usd
    embeddings.py             # EmbeddingClient (MiniLM), HashingEmbeddingClient, factory
    engine.py                 # Engine, PipelineResult, NodeTrace, _build_node_input, _extract_final_output
    agents/
      base.py                 # BaseAgent protocol, AgentResult, call_llm_json, _extract_json
      reader.py, chunker.py, classifier.py, summarizer.py, fact_checker.py, writer.py
      registry.py             # PrimitivePool, BUILTIN_CONSTRUCTORS, default_pool, default_data_dir
  evaluator/
    output.py                 # OutputEvaluator, JUDGE_PROMPT, _normalize, SLA/budget constants
  store/
    task_index.py             # TaskIndex (brute-force cosine kNN, numpy persisted)
    experience_store.py       # ExperienceStore, ScoredHit

data/primitives/*.yaml        # 6 base primitive descriptors
data/experiences/             # runtime artifacts (gitignored): experiences.jsonl, embeddings.npy, index_ids.json

tests/
  conftest.py                 # hashing_embedder fixture, ScriptedLLM helper, tmp_data_dir
  test_pipeline.py            # DAG construction, topo sort, cycle detection, fingerprint/hash, serialization
  test_executor.py            # linear run, cycle refusal, predecessor input flow, writer extraction
  test_primitives.py          # each primitive's i/o contract + malformed-JSON tolerance
  test_evaluator.py           # judge accuracy parse, speed/cost normalisation, composite weights
  test_architect.py           # empty store default, replay on match, pick-highest-composite
  test_store.py               # append/iterate, persist+reload, dedup, kNN, min_score
  test_integration.py         # 5-task run + 6th retrieval hit (the Phase 1 contract)
```

---

## 12. Tests (35, all passing in ~0.1 s)

The suite uses `HashingEmbeddingClient` (256d, deterministic, zero-dep) and `FakeLLMClient` (scripted responses) so no network and no model download is required.

| file | count | covers |
|---|---|---|
| `test_pipeline.py` | 7 | DAG helpers, topo sort (linear + diamond), cycle detection, fingerprint hash stability/difference, serialization round-trip |
| `test_executor.py` | 4 | linear pipeline runs all nodes, cycle refusal, predecessor output flows into child's input, writer `output` field extraction |
| `test_primitives.py` | 8 | each of 6 primitives returns its contract field; malformed JSON tolerated; JSON-in-markdown-fence parsed |
| `test_evaluator.py` | 6 | LLM judge accuracy parse, speed full/zero credit, cost full/low credit, composite collapses at acc=0 |
| `test_architect.py` | 4 | empty store → default; similar task → retrieval; dissimilar task → falls through; highest composite picked among ties |
| `test_store.py` | 5 | append+iterate, persist+reload, id dedup, kNN returns nearest, min_score filter |
| `test_integration.py` | 1 | **the Phase 1 verification contract** (see §13) |

Key fixtures (`conftest.py`):
- `hashing_embedder` — `HashingEmbeddingClient` instance.
- `tmp_data_dir(tmp_path, monkeypatch)` — sets `ARCHFORGE_DATA_DIR` to a tmp dir so tests never touch repo `data/`.
- `ScriptedLLM` — a `FakeLLMClient` variant whose responses are keyed by primitive name (robust to pipeline-shape changes); used by the integration test.

---

## 13. Verification against plan.md Phase 1 checklist

The plan's "Verify" section lists three checks. All three are exercised by `test_integration.py::test_end_to_end_5_tasks_then_retrieve` and confirmed by a manual CLI run.

### Checklist trace

| plan.md verify | how satisfied | evidence |
|---|---|---|
| **"Run 5 tasks, confirm all store experiences"** | Integration test runs 5 similar `"summarize the foo …"` tasks through the full `Architect → Engine → Evaluator → Store` loop with a scripted LLM; asserts `len(store) == 5` and every experience has `composite_score > 0`. | `test_integration.py` lines: loop over 5 descriptions → `assert len(store) == 5` + `assert exp.composite_score > 0.0` |
| **"Run a similar task, confirm it retrieves the prior pipeline"** | 6th task `"summarize the foo writeup"` runs through `Architect.compose`; asserts `decision.triggered_from == "retrieval"`, the replayed pipeline equals `DEFAULT_PIPELINE_AGENTS`, and `matched_experience_id ∈ {exp-0…exp-4}`. | `test_integration.py`: `assert decision.triggered_from == "retrieval"` … |
| **"Score is calculated and stored correctly"** | `test_evaluator.py` verifies the composite formula (`acc=0` collapses to ≤0.5); `test_store.py::test_persist_and_reload` verifies the round-trip; the integration test verifies the stored `composite_score` is positive and the JSONL reloads. | evaluator + store tests + integration |

### Manual CLI confirmation (real end-to-end, hashing embeddings for speed)

```
$ rm -rf /tmp/_sl_validate
$ ARCHFORGE_EMBEDDING_MODEL=hashing \
  ARCHFORGE_DATA_DIR=/tmp/_sl_validate \
  archforge run "summarize this short note about a cat"
→ Architect: composing pipeline for 'general' task...
  No similar past run — building default pipeline.     ← trigger: default
=== Output ===
(empty)
=== Scores ===
{ … "composite": 0.462, "trigger": "default" }

$ (same command, a second time)
→ Architect: composing pipeline for 'general' task...
  Replaying pipeline from experience exp-c54b38cd8c21 (prior score 0.46)   ← trigger: retrieval
{ … "composite": 0.462, "trigger": "retrieval" }

$ archforge inspect
Total experiences: 2
  - exp-6500d8716826 [general] composite=0.46 task='summarize this short note about a cat'
  - exp-c54b38cd8c21 [general] composite=0.46 task='summarize this short note about a cat'
```

The first run **builds** (no prior experience); the second run of the same task **replays** the first's pipeline. **Retrieval learning is demonstrated.** The `(empty)` output is expected with the `FakeLLMClient`/hashing fallback path (no `GROQ_API_KEY` → generic JSON stubs) — the point of the manual run is to prove the *loop wiring*, which is exactly what the scripted integration test covers precisely.

---

## 14. Known limitations & forward notes (for Phase 2)

1. **Output is empty on the no-API-key path.** Without `GROQ_API_KEY`, `GroqLLMClient` isn't constructed and `FakeLLMClient` returns generic `{"status": "ok"}` JSON — the writer's `output` field is then empty. Real tasks need `GROQ_API_KEY` set. This is a *demo path* limitation, not a code bug: the integration test uses a per-agent scripted LLM that returns proper writer output.
2. **Cost normalisation is token-count based**, not USD. `groq_cost_usd` exists but is unused in Phase 1. Phase 6 budget enforcement will wire it in.
3. **`WEIGHTS` is a `@staticmethod default_weights()`**, not a class-level dict. Don't try to "simplify" it back to a plain dict — `@dataclass` rejects mutable dict defaults. (Recorded to spare Phase 2 the same bug.)
4. **Engine predecessor-merge convention**: single predecessor → fields at top level; multiple → under `predecessors`. Primitives that want flat fields from a fan-in (e.g. `vote_aggregator` in Phase 2) will need to read from `predecessors`.
5. **Embeddings are recomputed, not stored in JSONL.** `to_dict()` records `has_embedding` only. On reload, `recompute_embeddings` fills the numpy index. If you change the embedding model, old vectors become invalid — delete `data/experiences/embeddings.npy` + `index_ids.json` and let `recompute` rebuild.
6. **`HashingEmbeddingClient` is NOT semantic.** It's a deterministic shape-only fallback for tests. Production retrieval quality depends on MiniLM (default).

### Phase 2 entry points (already reserved in the schema)
- `Experience.diagnoses` — the evaluator's diagnosis generator fills this.
- `Experience.interventions_applied` / `interventions_helped` — the intervention library + matching fills this.
- `Experience.structural` (`StructuralScores`) — the structural-metrics calculator fills this.
- `Engine.run` is already shaped for fan-out/fan-in (topo order + predecessor-merge) — Phase 2 adds a parallel scheduler over the same `PipelineDAG`.
- `Architect.compose` returns `ArchitectureDecision.candidates` — Phase 2's intervention step consumes those to rank mutations.

---

## 15. Quick command reference

```bash
# install (editable)
pip install -e . --break-system-packages

# run all tests (no network, no model download — ~0.1 s)
pytest -q

# real run with Groq
export GROQ_API_KEY=…
archforge run "summarize this file" --verbose

# offline / fast / no-API-key demo (uses hashing embeddings)
ARCHFORGE_EMBEDDING_MODEL=hashing \
  ARCHFORGE_DATA_DIR=/tmp/sl \
  archforge run "summarize this short note about a cat"

# inspect stored experiences
archforge inspect --last 10
```

---

## 16. Reproduction

```bash
cd "/media/vedant/Storage/My Projects/Agentic AI/ArchForge"
pytest -v                       # 35 passed
ARCHFORGE_EMBEDDING_MODEL=hashing \
  ARCHFORGE_DATA_DIR=/tmp/_sl_validate \
  archforge run "summarize this short note about a cat"   # trigger: default
# (repeat the same command)                                  # trigger: retrieval
archforge inspect                                           # 2 experiences
```
