from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ServiceCategory(str, Enum):
    CONNECTIVITY_C2 = "connectivity/C2"
    AIRSPACE_GOVERNANCE = "airspace_governance"
    SENSING_SITUATION = "sensing/situation_awareness"
    TRAJECTORY_CONTROL = "trajectory/cooperative_control"
    MISSION_APPLICATION = "mission_oriented_application"
    EDGE_INTELLIGENCE = "edge_intelligence/computing"
    CONTINUITY_RESILIENCE = "continuity/resilience"


@dataclass(frozen=True)
class PayloadProfile:
    semantic_type: str
    input_mb: float
    output_ratio: float = 1.0

    def output_mb(self) -> float:
        return max(0.0, self.input_mb * self.output_ratio)


@dataclass(frozen=True)
class QoSProfile:
    deadline_s: float
    priority: float = 1.0
    reliability_min: float = 0.0
    accuracy_min: float = 0.0
    max_energy_j: Optional[float] = None


@dataclass(frozen=True)
class MicroserviceSpec:
    ms_id: str
    category: ServiceCategory
    provides: List[str]
    cpu_per_mb: float
    memory_mb: float = 0.0
    storage_mb: float = 0.0
    input_semantic: str = "any"
    output_semantic: str = "any"
    output_ratio: float = 1.0
    allowed_node_types: List[str] = field(
        default_factory=lambda: ["vehicle", "uav", "rsu", "cloud_server"]
    )
    stateful: bool = False
    cold_start_s: float = 0.0
    deployment_cost: float = 0.0
    min_trust: float = 0.0


@dataclass(frozen=True)
class ServiceIntent:
    intent_id: str
    source_node_id: str
    category: ServiceCategory
    required_capabilities: List[str]
    payload: PayloadProfile
    qos: QoSProfile
    context: Dict[str, Any] = field(default_factory=dict)
    sink_node_id: Optional[str] = None


@dataclass
class ServiceGraphSpec:
    graph_id: str
    intent: ServiceIntent
    nodes: Dict[str, MicroserviceSpec]
    edges: List[Tuple[str, str]]

    def predecessors(self, ms_id: str) -> List[str]:
        return [u for u, v in self.edges if v == ms_id]

    def successors(self, ms_id: str) -> List[str]:
        return [v for u, v in self.edges if u == ms_id]

    def is_source_ms(self, ms_id: str) -> bool:
        return len(self.predecessors(ms_id)) == 0

    def is_terminal_ms(self, ms_id: str) -> bool:
        return len(self.successors(ms_id)) == 0
