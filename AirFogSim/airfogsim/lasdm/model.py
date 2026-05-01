from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class GraphStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class SFCFailureReason(str, Enum):
    NO_CANDIDATE = "no_candidate"
    TASK_FAILED = "task_failed"
    DEADLINE_MISSED = "deadline_missed"
    RELIABILITY_VIOLATION = "reliability_violation"
    ACCURACY_VIOLATION = "accuracy_violation"
    ENERGY_VIOLATION = "energy_violation"
    INVALID_GRAPH = "invalid_graph"


@dataclass(frozen=True)
class LASDMQoS:
    deadline_s: float
    priority: float = 1.0
    reliability_min: float = 0.0
    accuracy_min: float = 0.0
    max_energy_j: Optional[float] = None
    continuity_min: float = 0.0

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LASDMQoS":
        return cls(
            deadline_s=float(raw["deadline_s"]),
            priority=float(raw.get("priority", 1.0)),
            reliability_min=float(raw.get("reliability_min", 0.0)),
            accuracy_min=float(raw.get("accuracy_min", 0.0)),
            max_energy_j=None if raw.get("max_energy_j") is None else float(raw["max_energy_j"]),
            continuity_min=float(raw.get("continuity_min", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deadline_s": self.deadline_s,
            "priority": self.priority,
            "reliability_min": self.reliability_min,
            "accuracy_min": self.accuracy_min,
            "max_energy_j": self.max_energy_j,
            "continuity_min": self.continuity_min,
        }


@dataclass(frozen=True)
class LASDMSFCNode:
    node_id: str
    service_type: str
    required_capabilities: Tuple[str, ...]
    input_semantic: str = "any"
    output_semantic: str = "any"
    cpu_mb: float = 0.0
    memory_mb: float = 0.0
    storage_mb: float = 0.0
    optional: bool = False
    qos_override: Optional[LASDMQoS] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LASDMSFCNode":
        return cls(
            node_id=str(raw["node_id"]),
            service_type=str(raw.get("service_type", raw["node_id"])),
            required_capabilities=tuple(raw.get("required_capabilities", [])),
            input_semantic=str(raw.get("input_semantic", "any")),
            output_semantic=str(raw.get("output_semantic", "any")),
            cpu_mb=float(raw.get("cpu_mb", raw.get("cpu", 0.0))),
            memory_mb=float(raw.get("memory_mb", 0.0)),
            storage_mb=float(raw.get("storage_mb", 0.0)),
            optional=bool(raw.get("optional", False)),
            qos_override=LASDMQoS.from_dict(raw["qos_override"]) if raw.get("qos_override") else None,
            metadata=dict(raw.get("metadata", {})),
        )

    def effective_qos(self, graph_qos: LASDMQoS) -> LASDMQoS:
        return self.qos_override or graph_qos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "service_type": self.service_type,
            "required_capabilities": list(self.required_capabilities),
            "input_semantic": self.input_semantic,
            "output_semantic": self.output_semantic,
            "cpu_mb": self.cpu_mb,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "optional": self.optional,
            "qos_override": self.qos_override.to_dict() if self.qos_override else None,
            "metadata": dict(self.metadata),
        }


@dataclass
class LASDMServiceChain:
    sfc_id: str
    source_node_id: str
    sink_node_id: str
    payload_semantic: str
    payload_mb: float
    qos: LASDMQoS
    nodes: Dict[str, LASDMSFCNode]
    edges: List[Tuple[str, str]]
    context: Dict[str, Any] = field(default_factory=dict)
    status: GraphStatus = GraphStatus.PENDING
    submit_time: Optional[float] = None
    finish_time: Optional[float] = None
    failure_reason: Optional[SFCFailureReason] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LASDMServiceChain":
        nodes = {node.node_id: node for node in (LASDMSFCNode.from_dict(item) for item in raw.get("nodes", []))}
        edges = []
        for edge in raw.get("edges", []):
            if isinstance(edge, dict):
                edges.append((str(edge["from"]), str(edge["to"])))
            else:
                edges.append((str(edge[0]), str(edge[1])))
        chain = cls(
            sfc_id=str(raw["sfc_id"]),
            source_node_id=str(raw["source_node_id"]),
            sink_node_id=str(raw.get("sink_node_id", raw["source_node_id"])),
            payload_semantic=str(raw.get("payload_semantic", "any")),
            payload_mb=float(raw.get("payload_mb", 0.0)),
            qos=LASDMQoS.from_dict(raw.get("qos", {})),
            nodes=nodes,
            edges=edges,
            context=dict(raw.get("context", {})),
        )
        chain.validate()
        return chain

    @property
    def required_capabilities(self) -> List[str]:
        caps: List[str] = []
        for node in self.nodes.values():
            caps.extend(node.required_capabilities)
        return _ordered_unique(caps)

    def predecessors(self, node_id: str) -> List[str]:
        return [src for src, dst in self.edges if dst == node_id]

    def successors(self, node_id: str) -> List[str]:
        return [dst for src, dst in self.edges if src == node_id]

    def source_service_nodes(self) -> List[str]:
        return [node_id for node_id in self.nodes if not self.predecessors(node_id)]

    def terminal_service_nodes(self) -> List[str]:
        return [node_id for node_id in self.nodes if not self.successors(node_id)]

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError(f"SFC {self.sfc_id} must contain at least one service node")
        unknown = sorted({item for edge in self.edges for item in edge if item not in self.nodes})
        if unknown:
            raise ValueError(f"SFC {self.sfc_id} references unknown nodes: {', '.join(unknown)}")
        self.topological_order()
        if self.qos.deadline_s <= 0:
            raise ValueError(f"SFC {self.sfc_id} deadline_s must be positive")
        if self.payload_mb < 0:
            raise ValueError(f"SFC {self.sfc_id} payload_mb cannot be negative")

    def topological_order(self) -> List[str]:
        indegree = {node_id: 0 for node_id in self.nodes}
        outgoing: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for src, dst in self.edges:
            indegree[dst] += 1
            outgoing[src].append(dst)

        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        order: List[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for dst in outgoing[node_id]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        if len(order) != len(self.nodes):
            raise ValueError(f"SFC {self.sfc_id} contains a cycle")
        return order

    def mark_running(self, current_time: float) -> None:
        self.status = GraphStatus.RUNNING
        self.submit_time = current_time

    def mark_succeeded(self, current_time: float) -> None:
        self.status = GraphStatus.SUCCEEDED
        self.finish_time = current_time

    def mark_failed(self, reason: SFCFailureReason, current_time: float) -> None:
        self.status = GraphStatus.FAILED
        self.failure_reason = reason
        self.finish_time = current_time

    def mark_timed_out(self, current_time: float) -> None:
        self.status = GraphStatus.TIMED_OUT
        self.failure_reason = SFCFailureReason.DEADLINE_MISSED
        self.finish_time = current_time

    def is_terminal(self) -> bool:
        return self.status in {GraphStatus.SUCCEEDED, GraphStatus.FAILED, GraphStatus.TIMED_OUT}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sfc_id": self.sfc_id,
            "source_node_id": self.source_node_id,
            "sink_node_id": self.sink_node_id,
            "payload_semantic": self.payload_semantic,
            "payload_mb": self.payload_mb,
            "qos": self.qos.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [{"from": src, "to": dst} for src, dst in self.edges],
            "context": dict(self.context),
            "status": self.status.value,
            "submit_time": self.submit_time,
            "finish_time": self.finish_time,
            "failure_reason": self.failure_reason.value if self.failure_reason else None,
        }


def _ordered_unique(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
