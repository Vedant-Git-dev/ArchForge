"""Pipeline DAG: AgentNode, Edge, PipelineDAG + fingerprint primitives."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field


def new_pipeline_id() -> str:
    return f"pipe-{uuid.uuid4().hex[:12]}"


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
