from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

    DEFAULT_RANGES_M = {
        "v2v": 300.0,
        "v2u": 650.0,
        "u2v": 650.0,
        "v2i": 500.0,
        "i2v": 500.0,
        "u2i": 1000.0,
        "i2u": 1000.0,
        "u2u": 900.0,
        "i2i": 2500.0,
        "i2c": float("inf"),
        "c2i": float("inf"),
    }

    def __init__(
        self,
        env_adapter: Optional[LASDMEnvAdapter] = None,
        directory: Optional[ServiceInstanceDirectory] = None,
        max_link_distance_m: Optional[float] = None,
        link_ranges_m: Optional[Mapping[str, float]] = None,
    ):
        self.env_adapter = env_adapter or LASDMEnvAdapter(directory=directory)
        self.directory = directory
        self.max_link_distance_m = max_link_distance_m
        self.link_ranges_m = dict(self.DEFAULT_RANGES_M)
        if link_ranges_m:
            self.link_ranges_m.update({str(key).lower(): float(value) for key, value in link_ranges_m.items()})

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
            a = node_region.get(edge.src)
            b = node_region.get(edge.dst)
            if not a or not b or a == b:
                continue
            result.setdefault(a, set()).add(b)
            result.setdefault(b, set()).add(a)
        return {region: sorted(values) for region, values in result.items()}

    def _infer_edges(self, nodes: Mapping[str, TopologyNode], raw_links: Mapping[str, Any]) -> List[TopologyEdge]:
        edges: List[TopologyEdge] = []
        for src, dst in itertools.permutations(sorted(nodes), 2):
            src_node = nodes[src]
            dst_node = nodes[dst]
            link_type = _link_type(src_node.node_type, dst_node.node_type)
            if link_type == "none":
                continue
            distance = _distance(src_node.position, dst_node.position)
            nominal_range = self.max_link_distance_m if self.max_link_distance_m is not None else self.link_ranges_m.get(link_type, 0.0)
            rate = self._estimate_rate_mbps(link_type, distance, raw_links)
            latency = self._estimate_latency_s(link_type, distance, rate)
            if nominal_range and not math.isinf(nominal_range):
                range_ratio = distance / max(1e-9, nominal_range)
            else:
                range_ratio = 0.0
            out_of_nominal_range = bool(range_ratio > 1.0)
            distance_risk = max(0.0, min(1.0, range_ratio - 1.0))
            reliability = max(0.05, min(1.0, 1.0 - latency / 10.0 - 0.6 * distance_risk))
            edges.append(
                TopologyEdge(
                    src=src,
                    dst=dst,
                    link_type=link_type,
                    distance_m=distance,
                    rate_mbps=rate,
                    latency_s=latency,
                    reliability=reliability,
                    is_wireless=not link_type.endswith("2c") and not link_type.startswith("c2"),
                    metadata={
                        "nominal_range_m": nominal_range,
                        "range_ratio": range_ratio,
                        "out_of_nominal_range": out_of_nominal_range,
                        "distance_risk": distance_risk,
                    },
                )
            )
        return edges

    def _estimate_rate_mbps(self, link_type: str, distance_m: float, raw_links: Mapping[str, Any]) -> float:
        base = {
            "v2v": 20.0,
            "v2u": 35.0,
            "u2v": 35.0,
            "v2i": 50.0,
            "i2v": 50.0,
            "u2i": 80.0,
            "i2u": 80.0,
            "u2u": 60.0,
            "i2i": 200.0,
            "i2c": 1000.0,
            "c2i": 1000.0,
        }.get(link_type, 10.0)
        if math.isinf(distance_m):
            return base
        attenuation = 1.0 / (1.0 + max(0.0, distance_m) / 500.0)
        return max(0.1, base * attenuation)

    def _estimate_latency_s(self, link_type: str, distance_m: float, rate_mbps: float) -> float:
        propagation = 0.0 if math.isinf(distance_m) else distance_m / 3e8
        queue = 0.001 if rate_mbps >= 100 else 0.005
        if link_type in {"i2c", "c2i"}:
            queue += 0.02
        return float(propagation + queue)

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
