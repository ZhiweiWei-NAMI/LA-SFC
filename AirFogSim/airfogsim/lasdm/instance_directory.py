from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ServiceInstanceStatus(str, Enum):
    STARTING = "starting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DRAINING = "draining"
    FAILED = "failed"


@dataclass
class ServiceInstance:
    instance_id: str
    service_id: str
    node_id: str
    capabilities: Tuple[str, ...]
    node_type: str
    region_id: str = "global"
    input_semantic: str = "any"
    output_semantic: str = "any"
    capacity: Dict[str, float] = field(default_factory=dict)
    used: Dict[str, float] = field(default_factory=dict)
    max_concurrency: int = 1
    current_load: int = 0
    status: ServiceInstanceStatus = ServiceInstanceStatus.ACTIVE
    health_score: float = 1.0
    reliability_score: float = 1.0
    accuracy_score: float = 1.0
    trust_score: float = 1.0
    cold_start_s: float = 0.0
    version: str = "v1"
    last_heartbeat: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ServiceInstance":
        status = ServiceInstanceStatus(raw.get("status", ServiceInstanceStatus.ACTIVE.value))
        return cls(
            instance_id=str(raw["instance_id"]),
            service_id=str(raw["service_id"]),
            node_id=str(raw["node_id"]),
            capabilities=tuple(raw.get("capabilities", [])),
            node_type=str(raw.get("node_type", "unknown")),
            region_id=str(raw.get("region_id", "global")),
            input_semantic=str(raw.get("input_semantic", "any")),
            output_semantic=str(raw.get("output_semantic", "any")),
            capacity={k: float(v) for k, v in raw.get("capacity", {}).items()},
            used={k: float(v) for k, v in raw.get("used", {}).items()},
            max_concurrency=int(raw.get("max_concurrency", 1)),
            current_load=int(raw.get("current_load", 0)),
            status=status,
            health_score=float(raw.get("health_score", 1.0)),
            reliability_score=float(raw.get("reliability_score", 1.0)),
            accuracy_score=float(raw.get("accuracy_score", 1.0)),
            trust_score=float(raw.get("trust_score", 1.0)),
            cold_start_s=float(raw.get("cold_start_s", 0.0)),
            version=str(raw.get("version", "v1")),
            last_heartbeat=float(raw.get("last_heartbeat", 0.0)),
            metadata=dict(raw.get("metadata", {})),
        )

    def available(self, resource: str) -> float:
        return float(self.capacity.get(resource, 0.0)) - float(self.used.get(resource, 0.0))

    def can_reserve(self, resource_request: Dict[str, float]) -> bool:
        if not self.is_selectable():
            return False
        if self.current_load >= self.max_concurrency:
            return False
        for key, value in resource_request.items():
            if self.available(key) + 1e-9 < float(value):
                return False
        return True

    def is_selectable(self) -> bool:
        return self.status in {ServiceInstanceStatus.ACTIVE, ServiceInstanceStatus.DEGRADED}

    def update_runtime_load(
        self,
        current_load: Optional[int] = None,
        used: Optional[Dict[str, float]] = None,
        capacity: Optional[Dict[str, float]] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        if max_concurrency is not None:
            self.max_concurrency = max(0, int(max_concurrency))
        if current_load is not None:
            self.current_load = max(0, int(current_load))
        if used is not None:
            self.used = {key: max(0.0, float(value)) for key, value in used.items()}
        if capacity is not None:
            self.capacity = {key: max(0.0, float(value)) for key, value in capacity.items()}

    def reserve(self, resource_request: Dict[str, float]) -> None:
        if not self.can_reserve(resource_request):
            raise ValueError(f"Instance {self.instance_id} cannot reserve requested resources")
        for key, value in resource_request.items():
            self.used[key] = float(self.used.get(key, 0.0)) + float(value)
        self.current_load += 1

    def release(self, resource_request: Dict[str, float]) -> None:
        for key, value in resource_request.items():
            self.used[key] = max(0.0, float(self.used.get(key, 0.0)) - float(value))
        self.current_load = max(0, self.current_load - 1)

    def load_ratio(self) -> float:
        if self.max_concurrency <= 0:
            return 1.0
        return min(1.0, self.current_load / float(self.max_concurrency))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "service_id": self.service_id,
            "node_id": self.node_id,
            "capabilities": list(self.capabilities),
            "node_type": self.node_type,
            "region_id": self.region_id,
            "input_semantic": self.input_semantic,
            "output_semantic": self.output_semantic,
            "capacity": dict(self.capacity),
            "used": dict(self.used),
            "max_concurrency": self.max_concurrency,
            "current_load": self.current_load,
            "status": self.status.value,
            "health_score": self.health_score,
            "reliability_score": self.reliability_score,
            "accuracy_score": self.accuracy_score,
            "trust_score": self.trust_score,
            "cold_start_s": self.cold_start_s,
            "version": self.version,
            "last_heartbeat": self.last_heartbeat,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ServiceSelectionQuery:
    required_capabilities: Tuple[str, ...]
    service_id: Optional[str] = None
    input_semantic: str = "any"
    allowed_node_types: Tuple[str, ...] = field(default_factory=tuple)
    allowed_region_ids: Tuple[str, ...] = field(default_factory=tuple)
    preferred_region_id: Optional[str] = None
    min_reliability: float = 0.0
    min_accuracy: float = 0.0
    min_trust: float = 0.0
    resource_request: Dict[str, float] = field(default_factory=dict)


class ServiceInstanceDirectory:
    """In-memory service instance registry for centralized or regional orchestration."""

    def __init__(self):
        self._instances: Dict[str, ServiceInstance] = {}

    def register(self, instance: ServiceInstance) -> ServiceInstance:
        if instance.instance_id in self._instances:
            raise ValueError(f"Duplicate service instance id: {instance.instance_id}")
        self._instances[instance.instance_id] = instance
        return instance

    def upsert(self, instance: ServiceInstance) -> ServiceInstance:
        self._instances[instance.instance_id] = instance
        return instance

    def get(self, instance_id: str) -> ServiceInstance:
        return self._instances[instance_id]

    def all(self) -> Iterable[ServiceInstance]:
        return self._instances.values()

    def heartbeat(self, instance_id: str, current_time: float, status: Optional[ServiceInstanceStatus] = None) -> None:
        instance = self.get(instance_id)
        instance.last_heartbeat = current_time
        if status is not None:
            instance.status = status

    def retire_stale(self, current_time: float, ttl_s: float) -> List[str]:
        retired: List[str] = []
        for instance in self._instances.values():
            if current_time - instance.last_heartbeat > ttl_s and instance.status != ServiceInstanceStatus.FAILED:
                instance.status = ServiceInstanceStatus.FAILED
                retired.append(instance.instance_id)
        return retired

    def candidates(self, query: ServiceSelectionQuery) -> List[ServiceInstance]:
        requested_caps = set(query.required_capabilities)
        allowed_types = set(query.allowed_node_types)
        allowed_regions = set(query.allowed_region_ids)
        result: List[ServiceInstance] = []
        for instance in self._instances.values():
            if query.service_id is not None and instance.service_id != query.service_id:
                continue
            if not requested_caps.issubset(set(instance.capabilities)):
                continue
            if allowed_types and instance.node_type not in allowed_types:
                continue
            if allowed_regions and instance.region_id not in allowed_regions:
                continue
            if not _semantic_compatible(query.input_semantic, instance.input_semantic):
                continue
            if instance.reliability_score < query.min_reliability:
                continue
            if instance.accuracy_score < query.min_accuracy:
                continue
            if instance.trust_score < query.min_trust:
                continue
            if not instance.can_reserve(query.resource_request):
                continue
            result.append(instance)
        return result

    def select_best(
        self,
        query: ServiceSelectionQuery,
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[ServiceInstance]:
        candidates = self.candidates(query)
        if not candidates:
            return None
        weights = weights or {}
        return max(candidates, key=lambda item: self.score(item, query, weights))

    def score(
        self,
        instance: ServiceInstance,
        query: ServiceSelectionQuery,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        weights = weights or {}
        same_region = 1.0 if query.preferred_region_id and instance.region_id == query.preferred_region_id else 0.0
        resource_headroom = 0.0
        for key, value in query.resource_request.items():
            capacity = max(1e-9, float(instance.capacity.get(key, 0.0)))
            resource_headroom += max(0.0, instance.available(key) - float(value)) / capacity
        return (
            weights.get("health", 2.0) * instance.health_score
            + weights.get("reliability", 2.0) * instance.reliability_score
            + weights.get("accuracy", 1.0) * instance.accuracy_score
            + weights.get("trust", 1.0) * instance.trust_score
            + weights.get("region", 0.5) * same_region
            + weights.get("resource_headroom", 0.5) * resource_headroom
            - weights.get("load", 1.0) * instance.load_ratio()
            - weights.get("cold_start", 0.1) * instance.cold_start_s
        )

    def reserve(self, instance_id: str, resource_request: Dict[str, float]) -> None:
        self.get(instance_id).reserve(resource_request)

    def release(self, instance_id: str, resource_request: Dict[str, float]) -> None:
        self.get(instance_id).release(resource_request)

    def snapshot(self) -> Dict[str, Any]:
        return {instance_id: instance.to_dict() for instance_id, instance in sorted(self._instances.items())}


def _semantic_compatible(requested: str, provided: str) -> bool:
    return requested in ("any", provided) or provided in ("any", requested)
