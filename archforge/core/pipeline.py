"""Pipeline DAG: AgentNode, Edge, PipelineDAG + fingerprint primitives."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field


def new_pipeline_id() -> str:
    return f"pipe-{uuid.uuid4().hex[:12]}"


def new_node_id() -> str:
    """A unique pipeline-local node id.

    `PipelineDAG.linear` uses deterministic ``n0..n{n-1}`` ids so its edges
    can reference nodes before they exist; mutations build on an arbitrary
    existing DAG and append nodes, so they generate a fresh uuid-derived id
    that never collides with the linear ids or with any prior mutation's ids.
    Tests look nodes up by agent_type / topology, never by exact generated id.
    """
    return f"n{uuid.uuid4().hex[:8]}"


@dataclass
class AgentNode:
    """One agent in a pipeline. References a Primitive by `agent_type`."""

    id: str
    agent_type: str  # the primitive name to look up at execution time
    level: int = 0  # 0 = base primitive, 1+ = evolved
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "level": self.level,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentNode":
        return cls(
            id=data["id"],
            agent_type=data["agent_type"],
            level=data.get("level", 0),
            config=data.get("config", {}),
        )


@dataclass
class Edge:
    """Data flow between two agents. Direction matters."""

    source: str  # source node id
    target: str  # target node id
    data_type: str = "any"  # semantic label of what flows

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "data_type": self.data_type}

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        return cls(
            source=data["source"],
            target=data["target"],
            data_type=data.get("data_type", "any"),
        )


@dataclass
class PipelineDAG:
    """Directed acyclic graph of agents."""

    id: str
    nodes: list[AgentNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    fingerprint: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    # ----- construction helpers -----

    @classmethod
    def linear(cls, agent_types: list[str]) -> "PipelineDAG":
        """Convenience: build a linear pipeline with one node per agent_type.

        Node ids are auto-generated as "n0", "n1", ... so edges can be referenced
        deterministically.
        """
        nodes = [
            AgentNode(id=f"n{i}", agent_type=agent_types[i], level=0)
            for i in range(len(agent_types))
        ]
        edges: list[Edge] = []
        if len(nodes) >= 2:
            edges = [
                Edge(source=f"n{i}", target=f"n{i + 1}", data_type="any")
                for i in range(len(nodes) - 1)
            ]
        dag = cls(id=new_pipeline_id(), nodes=nodes, edges=edges)
        dag.fingerprint = dag.compute_fingerprint()
        return dag

    # ----- structural introspection -----

    def node_by_id(self, node_id: str) -> AgentNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def successors(self, node_id: str) -> list[AgentNode]:
        out: list[AgentNode] = []
        for e in self.edges:
            if e.source == node_id:
                n = self.node_by_id(e.target)
                if n is not None:
                    out.append(n)
        return out

    def predecessors(self, node_id: str) -> list[AgentNode]:
        out: list[AgentNode] = []
        for e in self.edges:
            if e.target == node_id:
                n = self.node_by_id(e.source)
                if n is not None:
                    out.append(n)
        return out

    def roots(self) -> list[AgentNode]:
        targets = {e.target for e in self.edges}
        return [n for n in self.nodes if n.id not in targets]

    def leaves(self) -> list[AgentNode]:
        sources = {e.source for e in self.edges}
        return [n for n in self.nodes if n.id not in sources]

    # ----- topological sort (Kahn's algorithm) -----

    def topo_order(self) -> list[AgentNode]:
        """Return nodes in a valid execution order. Raises ValueError on cycle."""
        if not self.nodes:
            return []

        in_degree: dict[str, int] = {n.id: 0 for n in self.nodes}
        for e in self.edges:
            in_degree[e.target] = in_degree.get(e.target, 0) + 1

        # sources: nodes with no incoming edges, in original order
        queue = [n for n in self.nodes if in_degree[n.id] == 0]
        order: list[AgentNode] = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for e in self.edges:
                if e.source == node.id:
                    in_degree[e.target] -= 1
                    if in_degree[e.target] == 0:
                        succ = self.node_by_id(e.target)
                        if succ is not None:
                            queue.append(succ)

        if len(order) != len(self.nodes):
            raise ValueError(f"Cycle detected in pipeline {self.id}")
        return order

    def has_cycle(self) -> bool:
        try:
            self.topo_order()
            return False
        except ValueError:
            return True

    # ----- fingerprint (Phase 1: simple; Phase 3 extends) -----

    def compute_fingerprint(self) -> dict:
        """Deterministic structural descriptor.

        Phase 1: enough for the experience store to dedup and the architect
        to score simple structural heuristics. Phase 3 replaces this with a
        hashed embedding for the pipeline index.
        """
        agent_types = [n.agent_type for n in self.nodes]
        return {
            "agent_type_set": sorted(set(agent_types)),
            "agent_sequence": agent_types,
            "length": len(self.nodes),
            "depth": self._depth(),
        }

    def _depth(self) -> int:
        """Longest path length (in nodes) from any root to any leaf."""
        if not self.nodes:
            return 0
        memo: dict[str, int] = {}

        def depth(nid: str) -> int:
            if nid in memo:
                return memo[nid]
            preds = self.predecessors(nid)
            if not preds:
                memo[nid] = 1
                return 1
            d = 1 + max(depth(p.id) for p in preds)
            memo[nid] = d
            return d

        return max(depth(n.id) for n in self.nodes)

    def critical_path(self) -> list[AgentNode]:
        """Longest root→leaf path (by node count). Returns the node list.

        Phase 2 structural evaluation uses this to define the "critical
        path" — the serial chain that bounds the pipeline's wall-clock
        time regardless of how wide the rest of the DAG fans out.

        Raises ValueError if the DAG has a cycle (via topo_order); callers
        that may receive unvalidated pipelines should check has_cycle first.
        """
        if not self.nodes:
            return []
        order = self.topo_order()
        best: dict[str, int] = {}
        parent: dict[str, str | None] = {n.id: None for n in self.nodes}
        for node in order:
            preds = self.predecessors(node.id)
            if not preds:
                best[node.id] = 1
            else:
                p = max(preds, key=lambda q: best[q.id])
                best[node.id] = best[p.id] + 1
                parent[node.id] = p.id
        leaves = self.leaves()
        pool = leaves if leaves else self.nodes
        end = max(pool, key=lambda n: best[n.id])
        path: list[AgentNode] = []
        cur: str | None = end.id
        while cur is not None:
            n = self.node_by_id(cur)
            if n is not None:
                path.append(n)
            cur = parent[cur]
        path.reverse()
        return path

    def depth(self) -> int:
        """Node count on the longest root→leaf path (0 if empty).

        Public alias of the recurrence used by `_depth`, kept for callers
        that want the count without reconstructing the whole path.
        """
        return self._depth()

    # ----- structural mutation (Phase 2) --------------------------------------
    #
    # The fixed vocabulary of plan.md's intervened mutations — insert, delete,
    # parallelize, swap (replace), merge — as pure functions on the DAG. Each
    # returns a NEW PipelineDAG (the originals are never mutated), recomputes
    # the fingerprint, and uses fresh unique ids so the result is cycle-free
    # and addressable. These are the operations the Intervention Library
    # (Phase 2.2) names declaratively and the Architect (Phase 2.3) dispatches;
    # keeping them here — DAG-level and agent-agnostic — is what lets an
    # intervention be a *description of an edit* rather than imperative code.
    #
    # Scope note: only the structural edit ships here. The *semantic* side of
    # two seeds is deliberately coarse and lands later: replace_node swaps an
    # agent_type but not a "larger-chunk chunker variant" (no separate variant
    # exists in the registry yet) — the variant swap is Phase 6 config work;
    # merge_chain collapses a chain to the first node's agent_type, not to a
    # fused-prompt primitive (that is Phase 5's Primitive Discoverer).

    def _clone_with(self, nodes: list[AgentNode], edges: list[Edge]) -> "PipelineDAG":
        dag = PipelineDAG(id=new_pipeline_id(), nodes=nodes, edges=edges)
        dag.fingerprint = dag.compute_fingerprint()
        return dag

    def insert_after(self, node_id: str, agent_type: str, *, level: int = 0) -> "PipelineDAG":
        """Splice a new node between `node_id` and ALL of its successors.

        X→S becomes X→M→S for every successor S of X. If X was a leaf (no
        successors) M becomes the new leaf. This is the primitive behind the
        ``no_validator`` / ``no_critique_loop`` seeds and the bottom half of
        the serial-bottleneck parallelize (an aggregator would be inserted
        after the fan-out).
        """
        if self.node_by_id(node_id) is None:
            raise KeyError(f"insert_after: unknown node id {node_id!r}")
        new = new_node_id()
        succs = self.successors(node_id)
        succ_ids = {s.id for s in succs}
        # Copy node objects (from_dict) so the caller's pipeline is untouched —
        # every mutation is functional: it returns a new DAG.
        nodes = [AgentNode.from_dict(n.to_dict()) for n in self.nodes]
        nodes.append(AgentNode(id=new, agent_type=agent_type, level=level))
        edges: list[Edge] = [
            Edge.from_dict(e.to_dict())
            for e in self.edges
            if not (e.source == node_id and e.target in succ_ids)
        ]
        edges.append(Edge(source=node_id, target=new, data_type="any"))
        for s in succs:
            edges.append(Edge(source=new, target=s.id, data_type="any"))
        return self._clone_with(nodes, edges)

    def insert_before(self, node_id: str, agent_type: str, *, level: int = 0) -> "PipelineDAG":
        """Splice a new node between ALL predecessors of `node_id` and itself.

        P→X becomes P→M→X for every predecessor P of X. If X was a root (no
        predecessors) M becomes the new root. Dual of `insert_after`; lets an
        intervention phrase e.g. "validate before generate" symmetrically.
        """
        if self.node_by_id(node_id) is None:
            raise KeyError(f"insert_before: unknown node id {node_id!r}")
        new = new_node_id()
        preds = self.predecessors(node_id)
        pred_ids = {p.id for p in preds}
        nodes = [AgentNode.from_dict(n.to_dict()) for n in self.nodes]
        nodes.append(AgentNode(id=new, agent_type=agent_type, level=level))
        edges: list[Edge] = [
            Edge.from_dict(e.to_dict())
            for e in self.edges
            if not (e.target == node_id and e.source in pred_ids)
        ]
        edges.append(Edge(source=new, target=node_id, data_type="any"))
        for p in preds:
            edges.append(Edge(source=p.id, target=new, data_type="any"))
        return self._clone_with(nodes, edges)

    def delete_node(self, node_id: str) -> "PipelineDAG":
        """Remove a node and bypass it: each predecessor is connected to each
        successor (the all-to-all bypass — the DAG-correct generalization of
        "cut out the middle of a chain" when a node has >1 pred or >1 succ).

        Deleting a leaf drops its incoming edges (succs empty → no bypass).
        Deleting a root drops its outgoing edges (preds empty → no bypass).
        Self-loop bypasses (pred==succ) are skipped; duplicate edges are
        deduped. Backs the ``redundant_agents`` / ``unused_outputs`` /
        ``unnecessary_agents`` seeds.
        """
        if self.node_by_id(node_id) is None:
            raise KeyError(f"delete_node: unknown node id {node_id!r}")
        pred_ids = [p.id for p in self.predecessors(node_id)]
        succ_ids = [s.id for s in self.successors(node_id)]
        nodes = [AgentNode.from_dict(n.to_dict()) for n in self.nodes if n.id != node_id]
        edges: list[Edge] = [
            Edge.from_dict(e.to_dict())
            for e in self.edges
            if e.source != node_id and e.target != node_id
        ]
        existing = {(e.source, e.target) for e in edges}
        for p in pred_ids:
            for s in succ_ids:
                if p != s and (p, s) not in existing:
                    existing.add((p, s))
                    edges.append(Edge(source=p, target=s, data_type="any"))
        return self._clone_with(nodes, edges)

    def replace_node(self, node_id: str, agent_type: str, *, level: int | None = None) -> "PipelineDAG":
        """Swap a node's agent_type in place, preserving its id and all edges.

        The structural swap primitive — backs any "swap agent X for a variant"
        intervention. A same-type replace is a structural no-op (and
        content_hash is identical, correctly). Distinct agent variants keyed
        by config, not agent_type, land with Phase 6.
        """
        if self.node_by_id(node_id) is None:
            raise KeyError(f"replace_node: unknown node id {node_id!r}")
        nodes: list[AgentNode] = []
        for n in self.nodes:
            if n.id == node_id:
                nodes.append(AgentNode(
                    id=n.id, agent_type=agent_type,
                    level=n.level if level is None else level, config=n.config,
                ))
            else:
                nodes.append(AgentNode.from_dict(n.to_dict()))
        edges = [Edge.from_dict(e.to_dict()) for e in self.edges]
        return self._clone_with(nodes, edges)

    def parallelize(self, node_id: str, agent_type: str, *, n: int = 1, level: int = 0) -> "PipelineDAG":
        """Fan out `n` sibling nodes of type `agent_type` alongside `node_id`.

        Each new sibling inherits node_id's predecessor set and successor
        set, so the siblings execute concurrently and the existing successor
        acts as the implicit fan-in (the engine already merges multi-predecessor
        inputs under a ``predecessors`` key). The dedicated vote/merge
        aggregator primitive the plan names is a Phase-2.5 base-primitive
        addition; this primitive supplies the structural fan-out it needs.
        Requires node_id to have a successor to merge into — parallelize a
        leaf and you get extra dangling leaves by construction.
        """
        if self.node_by_id(node_id) is None:
            raise KeyError(f"parallelize: unknown node id {node_id!r}")
        if n < 1:
            raise ValueError("parallelize: n must be >= 1")
        pred_ids = [p.id for p in self.predecessors(node_id)]
        succ_ids = [s.id for s in self.successors(node_id)]
        siblings = [AgentNode(id=new_node_id(), agent_type=agent_type, level=level)
                    for _ in range(n)]
        nodes = [AgentNode.from_dict(nd.to_dict()) for nd in self.nodes] + siblings
        edges = [Edge.from_dict(e.to_dict()) for e in self.edges]
        for sib in siblings:
            for p in pred_ids:
                edges.append(Edge(source=p, target=sib.id, data_type="any"))
            for s in succ_ids:
                edges.append(Edge(source=sib.id, target=s, data_type="any"))
        return self._clone_with(nodes, edges)

    def merge_chain(self, node_ids: list[str], *, merged_agent_type: str | None = None,
                    level: int = 0) -> "PipelineDAG":
        """Collapse a consecutive chain of nodes into a single node.

        Requires `node_ids` to be a *simple chain*: each consecutive pair has
        a forward edge, no non-consecutive edges cross the set, and only the
        first node has external predecessors and only the last has external
        successors. The merged node takes the first node's agent_type unless
        `merged_agent_type` is given, and inherits the chain's external
        edges. Backs the ``deep_chain`` seed ("flatten by merging consecutive
        compatible agents"); the *fused-prompt* merge is Phase 5.
        """
        if not node_ids:
            raise ValueError("merge_chain: node_ids must be non-empty")
        for nid in node_ids:
            if self.node_by_id(nid) is None:
                raise KeyError(f"merge_chain: unknown node id {nid!r}")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("merge_chain: node_ids must be unique")

        sset = set(node_ids)
        consecutive = {(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)}
        # Validate: a clean simple chain.
        for e in self.edges:
            if e.source in sset and e.target in sset:
                if (e.source, e.target) not in consecutive:
                    raise ValueError(
                        f"merge_chain: edge {e.source}→{e.target} crosses/loops the chain; "
                        "node_ids must be a simple consecutive chain")
            if e.source in sset and e.target not in sset and e.source != node_ids[-1]:
                raise ValueError(
                    f"merge_chain: node {e.source} is not the chain end but has an external successor")
            if e.target in sset and e.source not in sset and e.target != node_ids[0]:
                raise ValueError(
                    f"merge_chain: node {e.target} is not the chain start but has an external predecessor")
        for i in range(len(node_ids) - 1):
            if not any(e.source == node_ids[i] and e.target == node_ids[i + 1] for e in self.edges):
                raise ValueError(
                    f"merge_chain: no edge {node_ids[i]}→{node_ids[i + 1]}; not a consecutive chain")

        first, last = node_ids[0], node_ids[-1]
        ext_preds = [p.id for p in self.predecessors(first)]
        ext_succs = [s.id for s in self.successors(last)]
        merged_type = merged_agent_type if merged_agent_type is not None \
            else self.node_by_id(first).agent_type
        merged = AgentNode(id=new_node_id(), agent_type=merged_type, level=level)
        nodes = [AgentNode.from_dict(nd.to_dict()) for nd in self.nodes if nd.id not in sset]
        nodes.append(merged)
        edges = [Edge.from_dict(e.to_dict()) for e in self.edges
                 if e.source not in sset and e.target not in sset]
        for p in ext_preds:
            edges.append(Edge(source=p, target=merged.id, data_type="any"))
        for s in ext_succs:
            edges.append(Edge(source=merged.id, target=s, data_type="any"))
        return self._clone_with(nodes, edges)

    def content_hash(self) -> str:
        """Stable hash of the canonical topology. Used for dedup.

        Ignores the pipeline id (which is random per construction) so two
        pipelines with the same shape hash the same way.
        """
        canon = {
            "nodes": sorted([n.to_dict() for n in self.nodes], key=lambda d: d["id"]),
            "edges": sorted([e.to_dict() for e in self.edges], key=lambda d: (d["source"], d["target"])),
        }
        raw = json.dumps(canon, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ----- serialization -----

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "fingerprint": self.fingerprint,
            "has_embedding": bool(self.embedding),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineDAG":
        dag = cls(
            id=data["id"],
            nodes=[AgentNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[Edge.from_dict(e) for e in data.get("edges", [])],
            fingerprint=data.get("fingerprint", {}),
            embedding=[],
        )
        return dag
