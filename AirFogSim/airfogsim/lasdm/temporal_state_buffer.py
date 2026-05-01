from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .topology_builder import DynamicTopology, TopologyNode


@dataclass(frozen=True)
class TemporalSummary:
    window_size: int
    time_span_s: float
    node_churn_rate: float
    edge_churn_rate: float
    mean_load_delta: float
    mean_speed_mps: float
    active_node_count: int
    active_edge_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "window_size": float(self.window_size),
            "time_span_s": float(self.time_span_s),
            "node_churn_rate": float(self.node_churn_rate),
            "edge_churn_rate": float(self.edge_churn_rate),
            "mean_load_delta": float(self.mean_load_delta),
            "mean_speed_mps": float(self.mean_speed_mps),
            "active_node_count": float(self.active_node_count),
            "active_edge_count": float(self.active_edge_count),
        }


class TemporalStateBuffer:
    """Maintains recent dynamic topologies and extracts temporal features."""

    def __init__(self, maxlen: int = 4):
        self.maxlen = max(1, int(maxlen))
        self._items: deque[DynamicTopology] = deque(maxlen=self.maxlen)

    def append(self, topology: DynamicTopology) -> None:
        self._items.append(topology)

    def clear(self) -> None:
        self._items.clear()

    def latest(self) -> Optional[DynamicTopology]:
        return self._items[-1] if self._items else None

    def items(self) -> List[DynamicTopology]:
        return list(self._items)

    def summarize(self) -> TemporalSummary:
        items = list(self._items)
        if not items:
            return TemporalSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        first, last = items[0], items[-1]
        time_span = max(0.0, float(last.time_s) - float(first.time_s))
        node_churn = self.node_churn_rate()
        edge_churn = self.edge_churn_rate()
        return TemporalSummary(
            window_size=len(items),
            time_span_s=time_span,
            node_churn_rate=node_churn,
            edge_churn_rate=edge_churn,
            mean_load_delta=self.mean_load_delta(),
            mean_speed_mps=self.mean_speed_mps(),
            active_node_count=len(last.nodes),
            active_edge_count=len(last.edges),
        )

    def node_velocity(self, node_id: str) -> float:
        items = [topology for topology in self._items if node_id in topology.nodes]
        if len(items) < 2:
            return 0.0
        a = items[-2]
        b = items[-1]
        dt = max(1e-9, float(b.time_s) - float(a.time_s))
        pos_a = a.nodes[node_id].position
        pos_b = b.nodes[node_id].position
        dist = sum((x - y) ** 2 for x, y in zip(pos_a, pos_b)) ** 0.5
        return float(dist / dt)

    def mean_speed_mps(self) -> float:
        latest = self.latest()
        if latest is None or len(self._items) < 2:
            return 0.0
        speeds = [self.node_velocity(node_id) for node_id in latest.nodes]
        return sum(speeds) / max(1, len(speeds))

    def node_churn_rate(self) -> float:
        if len(self._items) < 2:
            return 0.0
        prev = set(self._items[-2].nodes)
        cur = set(self._items[-1].nodes)
        union = prev | cur
        if not union:
            return 0.0
        return len(prev ^ cur) / float(len(union))

    def edge_churn_rate(self) -> float:
        if len(self._items) < 2:
            return 0.0
        prev = self._items[-2].edge_keys()
        cur = self._items[-1].edge_keys()
        union = prev | cur
        if not union:
            return 0.0
        return len(prev ^ cur) / float(len(union))

    def mean_load_delta(self) -> float:
        if len(self._items) < 2:
            return 0.0
        prev = self._items[-2].nodes
        cur = self._items[-1].nodes
        common = sorted(set(prev) & set(cur))
        if not common:
            return 0.0
        return sum(cur[node_id].load_ratio - prev[node_id].load_ratio for node_id in common) / float(len(common))

    def link_stability(self, src: str, dst: str) -> float:
        if not self._items:
            return 0.0
        count = 0
        for topology in self._items:
            if any(edge.src == src and edge.dst == dst for edge in topology.edges):
                count += 1
        return count / float(len(self._items))
