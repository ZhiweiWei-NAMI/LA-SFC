from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .service_spec import ServiceGraphSpec, ServiceIntent


@dataclass
class OrchestrationMetrics:
    cold_start_count: int = 0
    deployment_cost: float = 0.0
    payload_tx_mb: float = 0.0
    cross_region_forward_count: int = 0
    service_graph_finish_time: Dict[str, float] = field(default_factory=dict)
    service_graph_deadline_s: Dict[str, float] = field(default_factory=dict)
    wrong_service_composition_ratio: Dict[str, float] = field(default_factory=dict)
    missing_service_capability_ratio: Dict[str, float] = field(default_factory=dict)
    extra_service_capability_ratio: Dict[str, float] = field(default_factory=dict)
    graph_submit_time: Dict[str, float] = field(default_factory=dict)

    def record_graph_submit(self, graph_id: str, submit_time: float) -> None:
        self.graph_submit_time[graph_id] = submit_time

    def record_graph_completion(self, graph_id: str, finish_time: float) -> None:
        self.service_graph_finish_time[graph_id] = finish_time

    def record_cold_start(self, cost: float) -> None:
        self.cold_start_count += 1
        self.deployment_cost += cost

    def record_payload_tx(self, payload_mb: float) -> None:
        self.payload_tx_mb += max(0.0, payload_mb)

    def record_cross_region_forward(self) -> None:
        self.cross_region_forward_count += 1

    def record_composition(self, intent: ServiceIntent, graph: ServiceGraphSpec) -> None:
        self.service_graph_deadline_s[graph.graph_id] = intent.qos.deadline_s
        required = set(intent.required_capabilities)
        provided = set()
        for spec in graph.nodes.values():
            provided.update(spec.provides)
        missing = len([cap for cap in required if cap not in provided])
        extra = len([cap for cap in provided if cap not in required])
        denom = max(1, len(required))
        self.missing_service_capability_ratio[graph.graph_id] = missing / denom
        self.extra_service_capability_ratio[graph.graph_id] = extra / denom
        self.wrong_service_composition_ratio[graph.graph_id] = missing / denom

    def summary(self) -> Dict[str, object]:
        return {
            "cold_start_count": self.cold_start_count,
            "deployment_cost": self.deployment_cost,
            "payload_tx_mb": self.payload_tx_mb,
            "cross_region_forward_count": self.cross_region_forward_count,
            "service_graph_finish_time": dict(self.service_graph_finish_time),
            "service_graph_deadline_s": dict(self.service_graph_deadline_s),
            "wrong_service_composition_ratio": dict(self.wrong_service_composition_ratio),
            "missing_service_capability_ratio": dict(self.missing_service_capability_ratio),
            "extra_service_capability_ratio": dict(self.extra_service_capability_ratio),
            "graph_submit_time": dict(self.graph_submit_time),
        }
