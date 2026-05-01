from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .instance_directory import ServiceInstance, ServiceInstanceDirectory, ServiceSelectionQuery
from .model import LASDMServiceChain, SFCFailureReason


@dataclass
class LASDMDecision:
    sfc_id: str
    assignments: Dict[str, str] = field(default_factory=dict)
    node_mapping: Dict[str, str] = field(default_factory=dict)
    routes: Dict[str, List[str]] = field(default_factory=dict)
    score: float = 0.0
    rejected_reason: Optional[SFCFailureReason] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sfc_id": self.sfc_id,
            "assignments": dict(self.assignments),
            "node_mapping": dict(self.node_mapping),
            "routes": {key: list(value) for key, value in self.routes.items()},
            "score": self.score,
            "rejected_reason": self.rejected_reason.value if self.rejected_reason else None,
            "diagnostics": dict(self.diagnostics),
        }


class LASDMOrchestrator:
    """Declarative SFC planner backed by the LASDM service instance directory."""

    def __init__(
        self,
        directory: ServiceInstanceDirectory,
        score_weights: Optional[Dict[str, float]] = None,
        commit_reservations: bool = True,
    ):
        self.directory = directory
        self.score_weights = score_weights or {}
        self.commit_reservations = commit_reservations

    def plan(self, chain: LASDMServiceChain) -> LASDMDecision:
        chain.validate()
        decision = LASDMDecision(sfc_id=chain.sfc_id)
        diagnostics: Dict[str, Any] = {
            "candidate_counts": {},
            "selected_scores": {},
            "candidate_regions": {},
            "selected_regions": {},
        }
        reservations: List[tuple[str, Dict[str, float]]] = []

        for sfc_node_id in chain.topological_order():
            sfc_node = chain.nodes[sfc_node_id]
            qos = sfc_node.effective_qos(chain.qos)
            query = ServiceSelectionQuery(
                service_id=sfc_node.service_type,
                required_capabilities=sfc_node.required_capabilities,
                input_semantic=sfc_node.input_semantic,
                allowed_node_types=_context_tuple(chain.context.get("allowed_node_types")),
                allowed_region_ids=_context_tuple(
                    chain.context.get("allowed_region_ids", chain.context.get("candidate_region_ids"))
                ),
                preferred_region_id=chain.context.get("preferred_region_id"),
                min_reliability=qos.reliability_min,
                min_accuracy=qos.accuracy_min,
                min_trust=float(chain.context.get("min_trust", 0.0)),
                resource_request=self._resource_request(sfc_node),
            )
            candidates = self.directory.candidates(query)
            diagnostics["candidate_counts"][sfc_node_id] = len(candidates)
            diagnostics["candidate_regions"][sfc_node_id] = _candidate_regions(candidates)
            selected = self._select_best(candidates, query)
            selection_query = query

            if selected is None:
                if sfc_node.optional:
                    diagnostics.setdefault("skipped_optional", []).append(sfc_node_id)
                    continue
                self._rollback(reservations)
                decision.rejected_reason = SFCFailureReason.NO_CANDIDATE
                decision.diagnostics = diagnostics
                return decision

            resource_request = self._resource_request(sfc_node)
            if self.commit_reservations:
                selected.reserve(resource_request)
                reservations.append((selected.instance_id, resource_request))
            score = self.directory.score(selected, selection_query, self.score_weights)
            decision.assignments[sfc_node_id] = selected.instance_id
            decision.node_mapping[sfc_node_id] = selected.node_id
            decision.routes[sfc_node_id] = self._route(self._route_source(chain, decision, sfc_node_id), selected)
            decision.score += score
            diagnostics["selected_scores"][sfc_node_id] = score
            diagnostics["selected_regions"][sfc_node_id] = selected.region_id
            if query.preferred_region_id and selected.region_id != query.preferred_region_id:
                diagnostics.setdefault("remote_serving_reason", {})[sfc_node_id] = "cross_region"

        decision.diagnostics = diagnostics
        return decision

    def _select_best(
        self,
        candidates: Sequence[ServiceInstance],
        query: ServiceSelectionQuery,
    ) -> Optional[ServiceInstance]:
        if not candidates:
            return None
        return max(candidates, key=lambda item: self.directory.score(item, query, self.score_weights))

    def _resource_request(self, sfc_node) -> Dict[str, float]:
        request = {
            "cpu": float(sfc_node.cpu_mb),
            "memory": float(sfc_node.memory_mb),
            "storage": float(sfc_node.storage_mb),
        }
        return {key: value for key, value in request.items() if value > 0}

    def _route(self, src_node_id: str, selected: ServiceInstance) -> List[str]:
        if src_node_id == selected.node_id:
            return [selected.node_id]
        return [src_node_id, selected.node_id]

    def _route_source(self, chain: LASDMServiceChain, decision: LASDMDecision, sfc_node_id: str) -> str:
        predecessors = chain.predecessors(sfc_node_id)
        if not predecessors:
            return chain.source_node_id
        mapped_predecessors = [decision.node_mapping[pred] for pred in predecessors if pred in decision.node_mapping]
        if mapped_predecessors:
            return mapped_predecessors[0]
        return chain.source_node_id

    def _rollback(self, reservations: List[tuple[str, Dict[str, float]]]) -> None:
        for instance_id, resource_request in reversed(reservations):
            self.directory.release(instance_id, resource_request)


def _context_tuple(value: Any, default: Iterable[str] = ()) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _candidate_regions(candidates: Iterable[ServiceInstance]) -> List[str]:
    return sorted({candidate.region_id for candidate in candidates})
