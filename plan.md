# SelfLearner — Adaptive Pipeline Builder

A self-learning multi-agent system that assembles, evaluates, diagnoses, and evolves task pipelines. The system learns which agent compositions work for which tasks, invents new primitive agents by compressing frequent subgraphs, and improves its own architecture over time.

**Domain-agnostic** — nothing in this system is specific to code, documents, finance, or any other domain. The primitive agents, the evaluator axes, the pipeline topology, and the learning mechanisms all operate on abstract task→pipeline→score tuples.

---

## Core Loop

```
Task → Classify → Retrieve → Architect designs Pipeline → Execute → Evaluate (output + structural + diagnosis)
  → Store Experience → Feed back into Retrieval / Intervention Library / Primitive Discoverer
```

Every run produces an experience tuple. Experiences feed three learning subsystems. The system gets better at **what to compose** (retrieval), **how to fix** (diagnosis→intervention), and **what to invent** (primitive emergence).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                              USER                                 │
│                   "Do this task"                                  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                        TASK ENTRY                                 │
│  • Accepts task description + metadata                            │
│  • Produces task embedding (vector)                              │
│  • Produces task type label (classification)                     │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     DUAL RETRIEVER                                 │
│                                                                   │
│  Index 1: Task similarity                                         │
│    - Embeds task description + type + metadata                    │
│    - Finds "what similar tasks have been solved before"          │
│                                                                   │
│  Index 2: Pipeline structural similarity                          │
│    - Embeds pipeline topology fingerprint                        │
│    - Finds "which past pipelines share shape with this need"      │
│                                                                   │
│  Combines both indices with learned weights                       │
│  Returns: ranked list of (task, pipeline, score) tuples          │
│                                                                   │
│  Key insight: different tasks (summarize doc vs generate API      │
│  docs) can share near-identical pipeline structure.              │
│  Task embedding alone misses this. Dual retrieval catches it.    │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                       ARCHITECT                                   │
│                                                                   │
│  1. Receives retrieved candidate pipelines                        │
│  2. Reads diagnoses from past low-scoring runs                   │
│  3. Matches diagnoses to learned interventions                   │
│  4. Applies targeted mutations (NOT random)                      │
│  5. Composes using both base and evolved primitives              │
│  6. Emits a Pipeline DAG for execution                           │
│                                                                   │
│  Decision inputs:                                                 │
│    - Retrieved pipeline (starting point)                          │
│    - Diagnoses from past similar runs (what failed)               │
│    - Intervention library (learned fixes for failure modes)       │
│    - Available primitives (base + evolved)                        │
│    - Task metadata (input size, complexity estimate)              │
│                                                                   │
│  Fallback: if no retrieval hit, assemble from scratch            │
│  using task-type default templates                                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    EXECUTION ENGINE                               │
│                                                                   │
│  • Takes a Pipeline DAG                                          │
│  • Resolves dependencies (topological sort)                       │
│  • Executes parallel branches concurrently                        │
│  • Streams data between agents (output→input edges)              │
│  • Handles failures (retry, skip, or abort per policy)           │
│  • Reports per-agent timing and token usage                      │
│                                                                   │
│  The engine is pipeline-agnostic. It doesn't know what agents    │
│  do. It only knows how to route data along edges.                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                       EVALUATOR                                   │
│                                                                   │
│  Three evaluation surfaces:                                       │
│                                                                   │
│  SURFACE 1: Output Quality                                        │
│    accuracy   — does the output satisfy the task? (0-1)          │
│    completeness — are there gaps? (0-1)                            │
│    speed      — wall-clock time vs SLA (0-1 normalized)          │
│    cost       — tokens + API calls in USD (0-1 normalized)       │
│    user_rating — explicit feedback when available (0-1)          │
│                                                                   │
│  SURFACE 2: Structural Quality                                    │
│    pipeline_length     — number of agent nodes                   │
│    critical_path       — longest serial path through DAG         │
│    parallelism_ratio   — parallel nodes / total nodes            │
│    redundant_agents    — agents with >80% output overlap         │
│    unused_outputs      — agents whose output no one reads       │
│    dependency_depth    — max depth in DAG                        │
│                                                                   │
│  SURFACE 3: Diagnosis                                             │
│    For each low metric, the evaluator explains WHY it is low.    │
│    "accuracy=0.3 because: 2 of 3 claims are unverified, no       │
│     validation step present in pipeline"                         │
│    "speed=0.2 because: 4 sequential analyzers create a           │
│     bottleneck, could be parallelized"                            │
│    "cost=0.4 because: chunker produces 200 chunks for a          │
│     500-word input, over-chunking"                               │
│                                                                   │
│  Composite score = weighted sum of output + structural metrics   │
│  Weights are NOT hardcoded — they are learned per task-type.      │
│                                                                   │
│  Diagnoses are structured, not free text. They map to            │
│  intervention patterns in the Architect's library.               │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     EXPERIENCE STORE                               │
│                                                                   │
│  Stores every run as an experience tuple:                        │
│                                                                   │
│  {                                                                │
│    task_embedding: [...],         // vector                      │
│    task_type: str,                // "analysis", "generation"... │
│    task_metadata: {...},          // size, domain, complexity    │
│    pipeline: PipelineDAG,         // the actual DAG run          │
│    pipeline_fingerprint: {...},  // topological fingerprint     │
│    pipeline_embedding: [...],     // structural vector           │
│    output_scores: {...},          // accuracy, speed, cost...    │
│    structural_scores: {...},      // length, depth, redundancy..│
│    diagnosis: [...],              // structured failure reasons   │
│    composite_score: float,        // the single number           │
│    interventions_applied: [...],  // what the architect changed  │
│    interventions_helped: bool,    // did the fix work?           │
│    primitives_used: [...],        // base + evolved agents used  │
│    timestamp: datetime,                                           │
│    generation: int,               // how many mutations deep     │
│  }                                                                │
│                                                                   │
│  Indexed by:                                                      │
│    - task embedding (for kNN task retrieval)                      │
│    - pipeline embedding (for kNN structural retrieval)            │
│    - diagnosis patterns (for intervention lookup)                 │
│    - score (for ranking + pruning)                                │
│    - pipeline_hash (for dedup and subgraph mining)                │
└──────────────────────────────────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
     ┌──────────────┐ ┌────────┐ ┌──────────────┐
     │  INTERVENTION │ │TEMPLATE │ │  PRIMITIVE    │
     │  LIBRARY      │ │DISTILLER│ │  DISCOVERER   │
     └──────────────┘ └────────┘ └──────────────┘

```

---

## The Three Learning Subsystems

### 1. Intervention Library (Reasoned Mutations)

**Problem with random mutations**: You mutate 20 things, 19 are useless, you keep the 1 that worked. Sample-inefficient. You burn runs to discover what one diagnosis could tell you.

**Solution**: The evaluator produces a diagnosis. The Architect reads the diagnosis and applies a *targeted* intervention.

#### Data Structures

```python
@dataclass
class Diagnosis:
    axis: str              # "accuracy" | "speed" | "cost" | "structure"
    severity: float        # 0-1, how bad
    reason: str            # natural language explanation
    structural_root: str   # categorical: "no_validator", "serial_bottleneck",
                           # "over_chunking", "redundant_agents", etc.

@dataclass
class Intervention:
    id: str
    diagnosis_pattern: str      # what structural_root to match
    mutation_type: str          # "insert" | "delete" | "parallelize" | "swap" | "merge"
    target_slot: str            # where in pipeline: "after_generate", "replace_X", etc.
    agent_to_insert: str | None # for insert/swap
    success_rate: float         # learned: (times_helped) / (times_tried)
    times_tried: int
    times_helped: int
    last_updated: datetime
```

#### Intervention Logic

```
When the Architect receives diagnosed failures:

1. Match each diagnosis to candidate interventions
   (by structural_root → diagnosis_pattern)

2. Rank candidates by success_rate

3. If best candidate success_rate > 0.5: apply it (exploitation)
   If best candidate success_rate < 0.5: apply it but also try
     one random mutation (exploration)
   If no candidate matches: create a new intervention entry
     with a random mutation, track it going forward

4. After execution, check: did the intervention help?
   (comparing to pre-intervention baseline)

5. Update success_rate of the intervention
```

#### Seeded Interventions (Start With These)

The system boots with a small set of known-good interventions. It discovers the rest.

| Diagnosis Root | Intervention |
|---|---|
| `no_validator` | Insert a validator agent after each generate step |
| `serial_bottleneck` | Parallelize the serial agents, add vote/merge aggregator |
| `over_chunking` | Swap chunker for a larger-chunk variant or remove chunker |
| `redundant_agents` | Delete one of the redundant pair, keep the higher-scoring one |
| `unused_outputs` | Delete the agent producing unused output |
| `no_critique_loop` | Insert critique→revision cycle after generate step |
| `deep_chain` | Flatten by merging consecutive compatible agents |

The system is NOT limited to these. As it runs, it discovers new diagnosis→intervention pairs. The seed just gives it a faster start.

---

### 2. Template Distiller (Pattern Compression)

After many tasks, the Architect doesn't need to search individual experiences. It can work from **compressed templates** — abstract pipeline recipes that represent "what works for task-shape X."

#### How Templates Form

```
1. Cluster task+pipeline pairs by structural fingerprint
2. Within each cluster, find the sub-DAG that is common
   to the top-scoring pipelines (a "frequent subgraph refinement")
3. Ask a judge: "Does this sub-DAG represent a coherent strategy?"
4. If yes → extract it as a template
5. Templates get used BEFORE retrieval for new tasks
```

#### Template Format

```yaml
name: "parallel-analyze-vote"
version: 3
triggers:
  task_types: ["analysis", "audit", "review"]
  input_characteristics:
    size: large           # "small" | "medium" | "large"
    complexity: high      # "low" | "medium" | "high"
  required_outputs:
    - high_accuracy       # triggers the vote step

topology:
  ingest: [reader, chunker]
  parallel_branches: N    # fan out to N analyzers (N learned)
  merge: vote_aggregator
  post_merge: [validator, writer]

learned_params:
  optimal_N: 3            # learned: 3 branches is the sweet spot
  preferred_analyzers: [classifier, stat_analyzer, keyword_extractor]

stats:
  derived_from: 47 experiences
  avg_score: 0.87
  last_used: 2026-06-28
```

#### Template vs Retrieval Priority

```
For a new task:
  1. Check templates → is there a template whose triggers match?
  2. If yes: instantiate the template, skip retrieval
  3. If no: fall back to dual retrieval
  4. Either way: apply diagnosed interventions
  5. Either way: evaluate and store as experience
```

Templates are **faster** (no kNN search) and **more reliable** (distilled from many runs). But they can go stale. If a template's recent avg_score drops below its historical avg, the system falls back to retrieval for that task type until the template is refreshed.

---

### 3. Primitive Discoverer (Emergent Agents)

The system starts with a set of base primitives. Over time, it **invents new primitives** by compressing frequent subgraphs into single agents.

This is the deepest learning mechanism. It changes the vocabulary the Architect can compose with.

#### Discovery Algorithm

```
PERIODICALLY (every N tasks or on schedule):

1. MINE frequent subgraphs
   - Scan all pipelines in Experience Store
   - Extract connected subgraphs of size 2-4
   - Count frequency and avg_score for each subgraph pattern

2. FILTER candidates
   A subgraph is a candidate for becoming a primitive IF:
     - frequency > THRESHOLD (appears in >X% of high-scoring pipelines)
     - avg_score of pipelines containing it > SCORE_THRESHOLD
     - the subgraph appears TOGETHER more often than its components
       appear SEPARATELY (tests that it's a real pattern, not coincidence)

3. EVALUATE coherence
   - Present the subgraph to a judge LLM:
     "Here are N agents that always appear together in this order.
      Do they form a single coherent capability? If so, what would
      you call it and what should its system prompt be?"
   - If judge says no → discard (it's coincidence, not compression)
   - If judge says yes → proceed to invention

4. INVENT the primitive
   - Create new agent definition:
     name: from judge suggestion
     prompt: from judge suggestion, refined by testing
     inputs: the first agent in the subgraph's input schema
     outputs: the last agent in the subgraph's output schema
   - Register in Agent Pool at the next level

5. VALIDATE
   - Run K tasks using the new primitive (replacing the subgraph)
   - Compare scores to the same tasks using the subgraph
   - If new primitive scores >= subgraph: KEEP
   - If new primitive scores < subgraph: DISCARD, log why

6. RETROFIT
   - Replace the subgraph with the new primitive in existing
     Experience Store entries (for cleaner retrieval)
   - The original subgraph is archived but not deleted
     (in case the primitive needs to be "unwrapped" later)
```

#### Primitive Hierarchy

```
Level 0: Base primitives (human-defined)
  ingest:    reader, fetcher, db_query
  transform: chunker, filter, normalizer, deduplicator
  analyze:   classifier, summarizer, extractor, stat_analyzer
  validate:  fact_checker, consistency_linter, format_validator, hallucination_detector
  generate:  writer, code_generator, translator, reformatter
  compose:   fan_out, fan_in_merge, vote_aggregator, critic

Level 1: First-generation evolved primitives
  e.g. "verified_fetcher"    = fetcher + fact_checker
  e.g. "structured_summary" = chunker + summarizer + format_validator
  e.g. "classified_extract"  = classifier + extractor

Level 2: Second-generation (composed from Level 1)
  e.g. "research_brief"  = verified_fetcher + structured_summary
  e.g. "tagged_analysis" = classified_extract + stat_analyzer

Level N: Further compression...
```

Each level deepens the vocabulary. The Architect can compose at ANY level — mixing Level 0 and Level 2 primitives in the same pipeline.

#### Properties of Evolved Primitives

| Property | Why it matters |
|---|---|
| **Faster** | 1 LLM call instead of N (the fused prompt handles all steps) |
| **Cheaper** | Fewer tokens consumed (no intermediate serialization) |
| **More reliable** | The fused prompt is battle-tested over many runs |
| **Less flexible** | Can't customize internal steps independently |
| **Discoverable** | Other agents can use it without knowing what's inside |
| **Reversible** | Can be "unwrapped" back into its component subgraph if needed |

---

## Dual Retrieval (Detail)

### Why Two Indices

| Retrieval by | Catches | Misses |
|---|---|---|
| Task similarity only | "I've done a similar task before" | Structural analogies across different task types |
| Pipeline similarity only | "This shape worked before regardless of task" | Task-specific nuances that change what agents should do |
| Both (dual) | Both signals, weighted per task-type | Nothing (strictly dominates single-index) |

### Pipeline Fingerprint (Structural Embedding)

```python
def pipeline_fingerprint(pipeline: DAG) -> dict:
    """Deterministic, interpretable structural descriptor."""
    return {
        "agent_type_set": sorted(set(n.agent_type for n in pipeline.nodes)),
        "has_validation": any(n.role == "validate" for n in pipeline.nodes),
        "has_critique_loop": has_cycle_like_pattern(pipeline),
        "parallel_branch_count": count_parallel_branches(pipeline),
        "depth": max_depth(pipeline),
        "width": max_width(pipeline),
        "fan_out_count": count_fan_outs(pipeline),
        "fan_in_count": count_fan_ins(pipeline),
        "agent_categories": count_by_category(pipeline),
    }
```

This fingerprint is hashed into a vector for kNN search. It can also be compared directly (structural edit distance) for fine ranking.

### Retrieval Logic

```python
def retrieve(task_embedding, expected_pipeline_shape, top_k=5):
    # Index 1: task similarity
    task_hits = task_index.search(task_embedding, k=top_k * 2)

    # Index 2: structural similarity
    pipeline_hits = pipeline_index.search(embed_pipeline(expected_pipeline_shape), k=top_k * 2)

    # Combine with learned weights
    # Some task types are better served by task matching
    # Others by structural matching
    w_task = learned_task_weight(task_type)
    w_pipeline = 1.0 - w_task

    candidates = merge_and_rank(
        task_hits, weight=w_task,
        pipeline_hits, weight=w_pipeline,
        take=top_k
    )

    return candidates
```

The weights (`w_task`, `w_pipeline`) are learned by tracking which index produced the candidate that the Architect actually used and that scored highest. Over time, the system learns "for `analysis` tasks, trust pipeline similarity more; for `generation` tasks, trust task similarity more."

---

## Structural Evaluation (Detail)

### Metrics

| Metric | Formula | What it catches |
|---|---|---|
| Pipeline length | `len(nodes)` | Over-composition, unnecessary steps |
| Critical path | `longest_serial_path(DAG)` | Serial bottlenecks |
| Parallelism ratio | `parallel_nodes / total_nodes` | Under-utilized concurrency |
| Redundancy | `count(pairs with >80% output Jaccard)` | Duplicate work |
| Unused outputs | `count(nodes with zero out-degree and not terminal)` | Dead agents |
| Dependency depth | `max_depth(DAG)` | Fragile deep chains |
| Fan-out ratio | `fan_out_nodes / total_nodes` | Over/under parallelization |

### Structural Score

```python
def structural_score(metrics: dict, task_type: str) -> float:
    """
    Not a fixed formula. Learns ideal structural profiles
    per task type by observing which structural patterns
    correlate with high composite scores.
    """
    # Baseline: penalize deviation from learned ideal
    ideal = learned_ideal_profile(task_type)
    deviation = weighted_distance(metrics, ideal)

    # Hard constraints: always penalize these
    penalty = 0.0
    if metrics["unused_outputs"] > 0:
        penalty += 0.1 * metrics["unused_outputs"]
    if metrics["redundancy"] > 0:
        penalty += 0.15 * metrics["redundancy"]

    return max(0, 1.0 - deviation - penalty)
```

The "learned ideal profile" is updated periodically by regressing structural metrics against composite scores across all experiences for a given task type.

---

## Composite Score

```python
def composite_score(
    output: OutputScores,
    structural: StructuralScores,
    task_type: str
) -> float:
    w = learned_weights(task_type)
    # defaults: accuracy=0.30, speed=0.15, cost=0.15,
    #           user_rating=0.10, structural=0.30
    return (
        w.accuracy   * output.accuracy +
        w.speed      * output.speed_normalized +
        w.cost       * output.cost_normalized +
        w.user_rating * (output.user_rating or 0.5) +
        w.structural * structural.score
    )
```

All weights are per-task-type, learned from observed correlations between metric improvements and outcome improvements.

---

## Data Model — Full Schema

```python
# ─── Task ───

@dataclass
class Task:
    id: str
    description: str
    type: str                    # "analysis", "generation", "extraction", ...
    metadata: dict               # arbitrary: domain, input_size, etc.
    embedding: list[float]       # vector from task description + metadata


# ─── Pipeline ───

@dataclass
class AgentNode:
    id: str
    agent_type: str              # refers to primitive name
    level: int                   # 0=base, 1+=evolved
    config: dict                 # agent-specific parameters

@dataclass
class Edge:
    source: str                  # agent node id
    target: str                  # agent node id
    data_type: str               # what kind of data flows on this edge

@dataclass
class PipelineDAG:
    id: str
    nodes: list[AgentNode]
    edges: list[Edge]
    fingerprint: dict            # structural fingerprint
    embedding: list[float]       # structural vector


# ─── Experience ───

@dataclass
class Diagnosis:
    axis: str
    severity: float
    reason: str
    structural_root: str

@dataclass
class Experience:
    id: str
    task_id: str
    task_embedding: list[float]
    task_type: str
    task_metadata: dict

    pipeline: PipelineDAG
    pipeline_hash: str

    # Scores
    output_accuracy: float
    output_completeness: float
    output_speed_normalized: float
    output_cost_normalized: float
    user_rating: float | None
    structural_score: float
    composite_score: float

    # Structural metrics
    pipeline_length: int
    critical_path_length: int
    parallelism_ratio: float
    redundant_agents: list[str]
    unused_outputs: list[str]
    dependency_depth: int

    # Diagnosis
    diagnoses: list[Diagnosis]

    # Intervention tracking
    interventions_applied: list[str]    # intervention IDs
    interventions_helped: dict[str, bool]  # id → helped?

    # Meta
    primitives_used: list[str]
    timestamp: datetime
    generation: int                     # mutation depth from original


# ─── Intervention ───

@dataclass
class Intervention:
    id: str
    diagnosis_pattern: str
    mutation_type: str
    target_slot: str
    agent_to_insert: str | None
    success_rate: float
    times_tried: int
    times_helped: int


# ─── Primitive ───

@dataclass
class Primitive:
    name: str
    level: int                     # 0=base, 1+=evolved
    role: str                      # "ingest", "transform", "analyze", etc.
    system_prompt: str
    input_schema: dict
    output_schema: dict

    # Only for evolved primitives:
    source_subgraph: list[str] | None   # component agent names
    fusing_prompt: str | None            # the prompt that replaces the subgraph
    validation_score: float | None       # score from validation run
    created_from_n_experiences: int | None
    created_at: datetime | None
    can_unwrap: bool = True


# ─── Template ───

@dataclass
class Template:
    name: str
    version: int
    triggers: dict                # task_types, input_characteristics, required_outputs
    topology: dict                # abstract DAG shape
    learned_params: dict          # optimal_N, preferred_agents, etc.
    stats: dict                   # derived_from_count, avg_score, last_used
    created_at: datetime
    last_validated: datetime
```

---

## Build Plan

### Phase 1: Skeleton — End-to-End Working

**Goal**: Task in → pipeline built → pipeline run → scored → stored → retrieved next time

**Scope**:
- 6 base primitives: `reader`, `chunker`, `classifier`, `summarizer`, `fact_checker`, `writer`
- Linear pipeline executor (no parallelism yet)
- kNN retrieval by task embedding only (single index)
- Simple evaluator: output accuracy + speed + cost (no structural, no diagnosis)
- Experience Store: SQLite + in-memory vector index (ChromaDB or simple numpy)
- Architect: retrieves best past pipeline, applies it as-is (no mutation)

**Deliverables**:
- `selflearner/` package with working end-to-end
- CLI: `selflearner run "summarize this file"` → output + score
- 5 test tasks that demonstrate retrieval learning (same task type, second run uses prior experience)

**Verify**:
- [ ] Run 5 tasks, confirm all store experiences
- [ ] Run a similar task, confirm it retrieves the prior pipeline
- [ ] Score is calculated and stored correctly

---

### Phase 2: Reasoned Mutations

**Goal**: Architect diagnoses failures and applies targeted interventions, not random mutations

**Scope**:
- Evaluator produces structured diagnoses (not just scores)
- Intervention Library with seeded interventions
- Architect matches diagnoses → interventions → applies mutations
- Intervention success tracking and success_rate updates
- Structured evaluation: add structural metrics
- Pipeline DAG executor supports parallelism (fan-out / fan-in)

**Deliverables**:
- Diagnosis output format and evaluator prompt that produces it
- Intervention schema + seed data
- Architect mutation logic (match diagnosis → pick intervention → apply)
- Structural metrics calculator
- Parallel execution in DAG engine

**Verify**:
- [ ] Run a task that fails accuracy → next similar task inserts a validator
- [ ] Run a task that is slow → next similar task parallelizes the bottleneck
- [ ] Intervention success_rate updates correctly
- [ ] Structural metrics are calculated and stored
- [ ] Parallel branches execute concurrently

---

### Phase 3: Dual Retrieval

**Goal**: Two retrieval indices (task + pipeline structure) that combine with learned weights

**Scope**:
- Pipeline fingerprint function
- Pipeline embedding index (second vector store)
- Dual retrieval with weight learning
- Cross-referencing task↔pipeline in Experience Store

**Deliverables**:
- `pipeline_fingerprint()` function
- Second vector index populated from pipeline fingerprints
- Retrieval logic that queries both indices and merges results
- Weight learning: track which index the winning candidate came from

**Verify**:
- [ ] Two different task types that share a pipeline structure are connected via structural retrieval
- [ ] Retrieval weights shift over time based on which index produces better candidates
- [ ] Dual retrieval outperforms single-index retrieval on a test set

---

### Phase 4: Template Distillation

**Goal**: Compress frequent pipeline patterns into reusable templates

**Scope**:
- Pipeline clustering by structural fingerprint
- Template extraction (common sub-DAG from high-scoring cluster)
- Template coherence judge (LLM-based)
- Template storage and lookup
- Template priority over retrieval
- Template staleness detection and refresh

**Deliverables**:
- Template miner (periodic job)
- Template storage format (see schema)
- Architect template injection (check templates before retrieval)
- Staleness tracker (fall back to retrieval if template degrades)

**Verify**:
- [ ] After 50+ tasks, system has extracted at least 2 templates
- [ ] New tasks matching template triggers use template directly
- [ ] Stale templates are detected and the system falls back to retrieval
- [ ] Template-based runs are faster than retrieval-based runs (skip kNN search)

---

### Phase 5: Primitive Discovery

**Goal**: System invents new primitives by compressing frequent subgraphs into single agents

**Scope**:
- Frequent subgraph mining over Experience Store
- Candidate filtering (frequency + coherence + compression benefit)
- LLM judge for coherence evaluation
- Primitive invention (name + prompt generation)
- Primitive validation (run K tasks, compare to subgraph)
- Primitive registration in Agent Pool
- Experience Store retrofit (replace subgraphs with new primitives)
- Primitive unwrapping (revert to subgraph if primitive degrades)

**Deliverables**:
- Subgraph miner (periodic job)
- Coherence judge prompt and integration
- Primitive creation pipeline (name + prompt + registration)
- Validation test harness
- Agent Pool with level tracking
- Retrofit logic for Experience Store
- Unwrap mechanism

**Verify**:
- [ ] After 100+ tasks, system discovers at least 1 new primitive
- [ ] New primitive scores >= its source subgraph on validation tasks
- [ ] Architect uses the new primitive in subsequent pipelines
- [ ] Primitive can be unwrapped and the subgraph restored
- [ ] Level 2 primitive can be created from two Level 1 primitives

---

### Phase 6: Hardening and Scale

**Goal**: Production-quality, observable, configurable

**Scope**:
- Learnable score weights (per task type)
- Cost tracking and budget enforcement
- Pipeline visualization (render DAG as SVG/mermaid)
- Dashboard: score trends, primitive evolution, template usage
- Exploration decay (less random mutation as confidence grows)
- Experience pruning (archive low-score entries, keep high-signal)
- Multi-user task isolation
- API layer (REST or gRPC)

**Deliverables**:
- Weight learner
- Budget tracker + enforcement
- Visualization module
- Dashboard web UI (or CLI dashboard)
- API server
- Documentation

**Verify**:
- [ ] Score weights have shifted from defaults based on observed data
- [ ] Budget limits are enforced (system refuses tasks that would exceed budget)
- [ ] Dashboard shows pipeline DAGs, score trends, primitive tree
- [ ] API accepts tasks and returns results
- [ ] Exploration rate decays as experience accumulates

---

## Folder Structure

```
selflearner/
├── README.md
├── plan.md                    # this file
├── pyproject.toml
├── selflearner/
│   ├── __init__.py
│   ├── main.py                # CLI entry point
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── task.py            # Task dataclass + embedding
│   │   ├── pipeline.py        # PipelineDAG, AgentNode, Edge
│   │   ├── experience.py      # Experience dataclass + store
│   │   └── primitive.py       # Primitive dataclass + pool
│   │
│   ├── architect/
│   │   ├── __init__.py
│   │   ├── retriever.py       # Dual retrieval (task + pipeline indices)
│   │   ├── designer.py        # Architect: retrieve → diagnose → compose pipeline
│   │   ├── interventions.py   # Intervention library + matching
│   │   └── templates.py       # Template distiller + template store
│   │
│   ├── executor/
│   │   ├── __init__.py
│   │   ├── engine.py          # DAG execution engine (topo sort + parallel)
│   │   └── agents/
│   │       ├── __init__.py
│   │       ├── base.py        # BaseAgent protocol/interface
│   │       ├── reader.py
│   │       ├── chunker.py
│   │       ├── classifier.py
│   │       ├── summarizer.py
│   │       ├── fact_checker.py
│   │       ├── writer.py
│   │       ├── fan_out.py
│   │       ├── fan_in.py
│   │       ├── vote.py
│   │       └── critic.py
│   │
│   ├── evaluator/
│   │   ├── __init__.py
│   │   ├── output.py          # Output quality scoring
│   │   ├── structural.py      # Structural metrics + scoring
│   │   └── diagnosis.py       # Diagnosis generation (why each metric is low)
│   │
│   ├── learner/
│   │   ├── __init__.py
│   │   ├── primitive_discoverer.py  # Frequent subgraph mining + primitive invention
│   │   ├── weight_learner.py        # Learn score weights per task type
│   │   └── weight_tracker.py        # Track which index/candidate won
│   │
│   ├── store/
│   │   ├── __init__.py
│   │   ├── experience_store.py # SQLite + vector index for experiences
│   │   ├── task_index.py      # Task embedding vector index
│   │   └── pipeline_index.py  # Pipeline embedding vector index
│   │
│   └── api/
│       ├── __init__.py
│       └── server.py          # REST/gRPC API (Phase 6)
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_executor.py
│   ├── test_evaluator.py
│   ├── test_architect.py
│   ├── test_retriever.py
│   ├── test_interventions.py
│   ├── test_primitives.py
│   └── test_integration.py
│
└── data/
    ├── primitives/            # base primitive definitions (YAML)
    ├── interventions/         # seeded interventions (YAML)
    └── experiences/           # SQLite DB + vector indices
```

---

## Design Principles

1. **Domain-agnostic**: No agent, metric, or data structure assumes a specific domain. "Reader" reads generic input. "Classifier" classifies generic data. Task types are labels, not enums.

2. **Diagnosis before intervention**: Every mutation is reasoned. The evaluator explains *why* something failed. The Architect matches that reason to a learned fix. Random mutation is the fallback, not the default.

3. **Two evaluation surfaces**: Output quality answers "did the task get done well?" Structural quality answers "is the pipeline itself well-designed?" Both feed learning.

4. **Primitives evolve**: The agent pool is not fixed. Frequent co-occurrence patterns get compressed into new primitives. The composition vocabulary grows richer over time.

5. **Dual retrieval**: Task similarity and pipeline structural similarity are orthogonal signals. Combining both catches structural analogies that task embedding alone would miss.

6. **Compression is learning**: Templates compress many experiences into one recipe. Primitives compress many steps into one agent. Each level of compression makes the next level possible (same as language evolution).

7. **Reversible**: Every evolved primitive can be unwrapped back into its source subgraph. Every template can be bypassed (fall back to retrieval). Nothing is locked in.

8. **Progressive complexity**: The system boots simple (Phase 1: retrieve-and-replay) and adds learning mechanisms incrementally. Each phase is independently useful. You can ship after Phase 2 and have something valuable.
