from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .env_adapter import LASDMEnvAdapter
from .instance_directory import ServiceInstanceDirectory


@dataclass(frozen=True)
class TopologyNode:
    node_id: str
    node_type: str
    region_id: str
    position: Tuple[float, float, float]
    cpu: float = 0.0
    memory: float = 0.0
    storage: float = 0.0
    load_ratio: float = 0.0
    trust_score: float = 1.0
    energy_consumption: float = 0.0
    service_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopologyEdge:
    src: str
    dst: str
    link_type: str
    distance_m: float
    rate_mbps: float
    latency_s: float
    reliability: float
    is_wireless: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str, str]:
        return (self.src, self.dst, self.link_type)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicTopology:
    time_s: float
    nodes: Dict[str, TopologyNode]
    edges: List[TopologyEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def edge_keys(self) -> set[Tuple[str, str, str]]:
        return {edge.key() for edge in self.edges}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }


class TopologyBuilder:
    """Extracts a dynamic aerial-ground topology from AirFogSim snapshots."""

    def __init__(
        self,
        env_adapter: Optional[LASDMEnvAdapter] = None,
        directory: Optional[ServiceInstanceDirectory] = None,
    ):
        self.env_adapter = env_adapter or LASDMEnvAdapter(directory=directory)
        self.directory = directory

    def build(self, env: Any, current_time: Optional[float] = None) -> DynamicTopology:
        now = _resolve_time(env, current_time)
        snapshot = self.env_adapter.sync_from_env(env, current_time=now) if env is not None else {}
        return self.from_snapshot(snapshot, current_time=now)

    def from_snapshot(self, snapshot: Mapping[str, Any], current_time: float = 0.0) -> DynamicTopology:
        raw_nodes = dict(snapshot.get("nodes", {}) or {})
        raw_tasks = dict(snapshot.get("tasks", {}) or {})
        raw_links = dict(snapshot.get("links", {}) or {})
        computing_by_node = dict(raw_tasks.get("computing_by_node", {}) or {})
        services_by_node = self._services_by_node()

        nodes: Dict[str, TopologyNode] = {}
        for node_id, raw in raw_nodes.items():
            profile = dict(raw.get("fog_profile", {}) or {})
            service_count = len(services_by_node.get(str(node_id), []))
            max_concurrency = max(1, service_count if service_count else int(profile.get("max_concurrency", 1) or 1))
            load_ratio = min(1.0, float(computing_by_node.get(node_id, 0)) / float(max_concurrency))
            nodes[str(node_id)] = TopologyNode(
                node_id=str(node_id),
                node_type=str(raw.get("node_type") or "unknown"),
                region_id=str(raw.get("region_id") or _default_region_id(str(node_id), raw)),
                position=_position_tuple(raw.get("position")),
                cpu=float(profile.get("cpu", 0.0) or 0.0),
                memory=float(profile.get("memory", 0.0) or 0.0),
                storage=float(profile.get("storage", 0.0) or 0.0),
                load_ratio=load_ratio,
                trust_score=float(raw.get("trust_score", 1.0) or 1.0),
                energy_consumption=float(raw.get("energy_consumption", 0.0) or 0.0),
                service_count=service_count,
                metadata={"authenticated": raw.get("authenticated")},
            )

        edges = self._infer_edges(nodes, raw_links)
        return DynamicTopology(
            time_s=float(current_time),
            nodes=nodes,
            edges=edges,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "raw_link_keys": sorted(str(key) for key in raw_links.keys()),
            },
        )

    def neighbor_map_by_region(self, topology: DynamicTopology) -> Dict[str, List[str]]:
        regions = sorted({node.region_id for node in topology.nodes.values() if node.region_id})
        result = {region: set() for region in regions}
        node_region = {node_id: node.region_id for node_id, node in topology.nodes.items()}
        for edge in topology.edges:
            if edge.rate_mbps <= 0.0 or edge.reliability <= 0.0:
                continue
            a = node_region.get(edge.src)
            b = node_region.get(edge.dst)
            if not a or not b or a == b:
                continue
            result.setdefault(a, set()).add(b)
            result.setdefault(b, set()).add(a)
        return {region: sorted(values) for region, values in result.items()}

    def _infer_edges(self, nodes: Mapping[str, TopologyNode], raw_links: Mapping[str, Any]) -> List[TopologyEdge]:
        edges: List[TopologyEdge] = []
        measured_links = raw_links.get("measured_links", [])
        if measured_links is None:
            measured_links = []
        if not isinstance(measured_links, Sequence) or isinstance(measured_links, (str, bytes)):
            raise ValueError("links.measured_links must be a sequence of measured link records")
        for raw in measured_links:
            if not isinstance(raw, Mapping):
                raise ValueError("measured link records must be mappings")
            src = str(raw.get("src", "") or "")
            dst = str(raw.get("dst", "") or "")
            if src not in nodes or dst not in nodes:
                continue
            src_node = nodes[src]
            dst_node = nodes[dst]
            link_type = str(raw.get("link_type") or _link_type(src_node.node_type, dst_node.node_type)).lower()
            if link_type == "none":
                continue
            distance = _distance(src_node.position, dst_node.position)
            is_wireless = bool(raw.get("is_wireless", _is_wireless_link_type(link_type)))
            if is_wireless:
                rate = _float(raw.get("rate_mbps_sum", raw.get("rate_mbps")), 0.0)
                transmitted_mbit = _transmitted_mbit(raw)
                active_count = int(_float(raw.get("task_count"), 0.0))
                if active_count <= 0 or rate <= 0.0 or transmitted_mbit <= 0.0:
                    continue
                latency = transmitted_mbit / rate
                reliability = 1.0
                metadata = {
                    "measured": True,
                    "metric_source": "airfogsim_last_tick_wireless",
                    "transmitted_mbit": transmitted_mbit,
                    "allocated_rb_count": int(_float(raw.get("allocated_rb_count"), 0.0)),
                    "task_count": active_count,
                    "success_count": int(_float(raw.get("success_count"), 0.0)),
                    "failure_count": int(_float(raw.get("failure_count"), 0.0)),
                    "simulation_interval_s": _float(raw.get("simulation_interval_s"), 0.0),
                }
            else:
                rate = _float(raw.get("capacity_mbps", raw.get("rate_mbps")), 0.0)
                queue_before = _float(raw.get("queue_bytes_before"), 0.0)
                transmitted_bytes = _float(raw.get("transmitted_bytes"), 0.0)
                active_count = int(_float(raw.get("active_flow_count"), 0.0))
                if active_count <= 0:
                    continue
                capacity_bytes_per_s = rate * 1e6 / 8.0
                queue_delay_s = queue_before / capacity_bytes_per_s if capacity_bytes_per_s > 0.0 else 0.0
                latency = _float(raw.get("prop_ms"), 0.0) / 1000.0 + queue_delay_s
                reliability = 1.0 if rate > 0.0 and transmitted_bytes > 0.0 else 0.0
                link_type = _link_type(src_node.node_type, dst_node.node_type)
                if link_type == "none":
                    link_type = str(raw.get("link_type") or "wired").lower()
                metadata = {
                    "measured": True,
                    "metric_source": "airfogsim_last_tick_wired",
                    "queue_bytes_before": queue_before,
                    "queue_bytes_after": _float(raw.get("queue_bytes_after"), 0.0),
                    "queue_delay_s": queue_delay_s,
                    "transmitted_bytes": transmitted_bytes,
                    "active_flow_count": active_count,
                    "simulation_interval_s": _float(raw.get("simulation_interval_s"), 0.0),
                }
            edges.append(
                TopologyEdge(
                    src=src,
                    dst=dst,
                    link_type=link_type,
                    distance_m=distance,
                    rate_mbps=rate,
                    latency_s=latency,
                    reliability=reliability,
                    is_wireless=is_wireless,
                    metadata=metadata,
                )
            )
        return edges

    def _services_by_node(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}
        if self.directory is None:
            return result
        for instance in self.directory.all():
            result.setdefault(instance.node_id, []).append(instance.instance_id)
        return result


def _resolve_time(env: Any = None, current_time: Optional[float] = None) -> float:
    if current_time is not None:
        return float(current_time)
    return float(getattr(env, "simulation_time", 0.0) or 0.0)


def _position_tuple(position: Any) -> Tuple[float, float, float]:
    if isinstance(position, Mapping):
        return (float(position.get("x", 0.0) or 0.0), float(position.get("y", 0.0) or 0.0), float(position.get("z", 0.0) or 0.0))
    if isinstance(position, Sequence) and not isinstance(position, (str, bytes)):
        values = [float(item or 0.0) for item in list(position)[:3]]
        while len(values) < 3:
            values.append(0.0)
        return tuple(values)  # type: ignore[return-value]
    return (0.0, 0.0, 0.0)


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _transmitted_mbit(raw: Mapping[str, Any]) -> float:
    if raw.get("transmitted_mbit") is not None:
        return _float(raw.get("transmitted_mbit"), 0.0)
    return _float(raw.get("transmitted_bytes"), 0.0) * 8e-6


def _is_wireless_link_type(link_type: str) -> bool:
    value = str(link_type).lower()
    return not value.endswith("2c") and not value.startswith("c2") and value != "wired"


def _link_type(src_type: str, dst_type: str) -> str:
    code = {
        "vehicle": "v",
        "uav": "u",
        "rsu": "i",
        "cloud_server": "c",
        "cloud": "c",
    }
    a = code.get(str(src_type), "")
    b = code.get(str(dst_type), "")
    if not a or not b:
        return "none"
    if a == "c" and b == "c":
        return "none"
    return f"{a}2{b}"


def _default_region_id(node_id: str, raw: Mapping[str, Any]) -> str:
    node_type = str(raw.get("node_type") or "")
    if node_type == "rsu":
        return node_id
    return str(raw.get("nearest_rsu") or raw.get("region") or "global")
