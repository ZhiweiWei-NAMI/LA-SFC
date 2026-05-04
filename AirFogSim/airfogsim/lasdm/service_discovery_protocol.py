from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .distributed_catalog import CatalogCandidate, DistributedServiceCatalog
from .semantic_encoder import sfc_node_text
from .semantic_exchange import SemanticExchange

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveryRequest:
    request_id: str
    agent_id: str
    sfc_id: str
    sfc_node_id: str
    service_id: str
    required_capabilities: Sequence[str]
    query_text: str
    created_at_s: float
    top_k: int = 8
    min_similarity: float = -1.0
    allowed_node_types: Sequence[str] = field(default_factory=tuple)
    link_input_semantic: str = ""
    request_type: str = ""
    chain_position: int = 0


@dataclass
class DiscoveryTraceRecord:
    time_s: float
    request_id: str
    agent_id: str
    sfc_id: str
    sfc_node_id: str
    service_id: str
    candidate_count: int
    local_candidate_count: int
    remote_candidate_count: int
    stale_remote_ratio: float
    top_score: Optional[float]
    selected_instance_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "sfc_id": self.sfc_id,
            "sfc_node_id": self.sfc_node_id,
            "service_id": self.service_id,
            "candidate_count": self.candidate_count,
            "local_candidate_count": self.local_candidate_count,
            "remote_candidate_count": self.remote_candidate_count,
            "stale_remote_ratio": self.stale_remote_ratio,
            "top_score": self.top_score,
            "selected_instance_id": self.selected_instance_id,
            "metadata": dict(self.metadata),
        }


class DistributedServiceDiscoveryProtocol:
    """Coordinates advertisement delivery and candidate discovery across agents."""

    def __init__(
        self,
        catalogs: Mapping[str, DistributedServiceCatalog],
        exchange: SemanticExchange,
        neighbor_map: Optional[Mapping[str, Any]] = None,
    ):
        self.catalogs: Dict[str, DistributedServiceCatalog] = {str(key): value for key, value in catalogs.items()}
        self.exchange = exchange
        self.neighbor_map: Dict[str, Any] = _normalize_neighbor_map(neighbor_map or {})
        self.discovery_trace: List[DiscoveryTraceRecord] = []
        self.candidate_detail_trace: List[Dict[str, Any]] = []
        self.exchange_trace: List[Dict[str, Any]] = []

    def set_neighbors(self, neighbor_map: Mapping[str, Any]) -> None:
        self.neighbor_map = _normalize_neighbor_map(neighbor_map)

    def tick(self, now_s: float) -> None:
        agent_dirs = {agent_id: catalog.local_directory for agent_id, catalog in self.catalogs.items()}
        # Reuse the encoder of the first catalog; catalogs are expected to share model/compressor.
        encoder = next(iter(self.catalogs.values())).encoder if self.catalogs else None
        if encoder is None:
            return
        delivered = self.exchange.step_exchange(agent_dirs, encoder, self.neighbor_map, now_s)
        for agent_id, ads in delivered.items():
            if agent_id in self.catalogs:
                self.catalogs[agent_id].ingest_remote(ads, now_s)
        self.exchange_trace.extend(self.exchange.trace_rows[len(self.exchange_trace) :])

    def discover(
        self,
        request: DiscoveryRequest,
        include_remote: bool = True,
    ) -> List[CatalogCandidate]:
        catalog = self.catalogs[request.agent_id]
        candidates = catalog.query(
            query_text=request.query_text,
            service_id=request.service_id,
            required_capabilities=request.required_capabilities,
            allowed_node_types=request.allowed_node_types,
            now_s=request.created_at_s,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            include_remote=include_remote,
            link_input_semantic=request.link_input_semantic or None,
            request_type=request.request_type,
            chain_position=request.chain_position,
        )
        record = DiscoveryTraceRecord(
            time_s=request.created_at_s,
            request_id=request.request_id,
            agent_id=request.agent_id,
            sfc_id=request.sfc_id,
            sfc_node_id=request.sfc_node_id,
            service_id=request.service_id,
            candidate_count=len(candidates),
            local_candidate_count=sum(1 for item in candidates if not item.is_remote),
            remote_candidate_count=sum(1 for item in candidates if item.is_remote),
            stale_remote_ratio=catalog.stale_ratio(request.created_at_s),
            top_score=candidates[0].semantic_score if candidates else None,
        )
        self.discovery_trace.append(record)
        self._append_candidate_detail_rows(request, candidates)
        return candidates

    def discover_sfc_node(
        self,
        agent_id: str,
        chain: Any,
        sfc_node_id: str,
        now_s: float,
        top_k: int = 8,
        min_similarity: float = -1.0,
        include_remote: bool = True,
        link_input_semantic: Optional[str] = None,
    ) -> List[CatalogCandidate]:
        sfc_node = chain.nodes[sfc_node_id]
        chain_order = list(chain.topological_order())
        context = dict(getattr(chain, "context", {}) or {})
        request_type = str(context.get("request_type", "") or "")
        if not request_type:
            request_type = "forest_fire_monitoring"
            LOGGER.warning(
                "Discovery request for sfc=%s node=%s has no request_type; using %s. "
                "V21 runtime configs should assign request_type before discovery.",
                chain.sfc_id,
                sfc_node_id,
                request_type,
            )
        request = DiscoveryRequest(
            request_id=f"{chain.sfc_id}:{sfc_node_id}:{now_s:.3f}:{agent_id}",
            agent_id=str(agent_id),
            sfc_id=str(chain.sfc_id),
            sfc_node_id=str(sfc_node_id),
            service_id=str(sfc_node.service_type),
            required_capabilities=tuple(sfc_node.required_capabilities),
            allowed_node_types=tuple(str(item) for item in (getattr(chain, "context", {}) or {}).get("allowed_node_types", ()) or ()),
            query_text=sfc_node_text(sfc_node, payload_semantic=getattr(chain, "payload_semantic", "")),
            link_input_semantic=str(
                link_input_semantic
                if link_input_semantic is not None
                else (getattr(chain, "payload_semantic", "") if not chain.predecessors(sfc_node_id) else getattr(sfc_node, "input_semantic", ""))
            ),
            created_at_s=float(now_s),
            top_k=int(top_k),
            min_similarity=float(min_similarity),
            request_type=request_type,
            chain_position=chain_order.index(sfc_node_id) if sfc_node_id in chain_order else 0,
        )
        return self.discover(request, include_remote=include_remote)

    def remote_discovery_rate(self) -> float:
        if not self.discovery_trace:
            return 0.0
        return sum(1 for row in self.discovery_trace if row.remote_candidate_count > 0) / float(len(self.discovery_trace))

    def message_overhead_rows(self) -> List[Dict[str, Any]]:
        rows = list(self.exchange_trace)
        for row in rows:
            row.setdefault("protocol", "distributed_semantic_exchange")
        return rows

    def semantic_candidate_trace_rows(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.discovery_trace]

    def semantic_candidate_detail_trace_rows(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.candidate_detail_trace]

    def mark_selected_candidate(
        self,
        sfc_id: str,
        sfc_node_id: str,
        instance_id: str,
        selected_at_s: Optional[float] = None,
        selected_source_node_id: Optional[str] = None,
        selected_metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Mark the most recent detail trace row for the selected candidate."""

        target = (str(sfc_id), str(sfc_node_id), str(instance_id))
        for row in reversed(self.candidate_detail_trace):
            key = (str(row.get("sfc_id", "")), str(row.get("sfc_node_id", "")), str(row.get("instance_id", "")))
            if key == target:
                row["selected"] = True
                row["selected_at_s"] = selected_at_s if selected_at_s is not None else ""
                if selected_source_node_id is not None:
                    row["selected_source_node_id"] = str(selected_source_node_id)
                if selected_metadata:
                    row["actual_route_source_node_id"] = str(selected_metadata.get("actual_route_source_node_id", selected_source_node_id or ""))
                    row["route_available"] = _float(selected_metadata.get("route_available"), _float(row.get("route_available"), 0.0))
                    row["route_hops"] = int(_float(selected_metadata.get("route_hops"), _float(row.get("route_hops"), 0.0)))
                    row["expected_runtime_penalty_s"] = _float(
                        selected_metadata.get("expected_runtime_penalty_s"),
                        _float(row.get("expected_runtime_penalty_s"), 0.0),
                    )
                    row["estimated_compute_s"] = _float(
                        selected_metadata.get("estimated_compute_s"),
                        _float(row.get("estimated_compute_s"), 0.0),
                    )
                    row["effective_cpu"] = _float(selected_metadata.get("effective_cpu"), _float(row.get("effective_cpu"), 0.0))
                    row["task_cpu"] = _float(selected_metadata.get("task_cpu"), _float(row.get("task_cpu"), 0.0))
                    row["deadline_slack_s"] = _float(selected_metadata.get("deadline_slack_s"), _float(row.get("deadline_slack_s"), 0.0))
                    row["utility_prior"] = _float(selected_metadata.get("utility_prior"), _float(row.get("utility_prior"), 0.0))
                    row["link_similarity"] = _float(selected_metadata.get("link_similarity"), _float(row.get("link_similarity"), row.get("semantic_score", 0.0)))
                    row["semantic_cumulative_quality"] = _float(
                        selected_metadata.get("semantic_cumulative_quality_if_selected"),
                        _float(row.get("semantic_cumulative_quality"), row.get("semantic_score", 1.0)),
                    )
                return True
        return False

    def update_candidate_route_metadata(
        self,
        sfc_id: str,
        sfc_node_id: str,
        candidates: Sequence[CatalogCandidate],
    ) -> None:
        """Attach runtime route features computed after discovery to detail rows."""

        route_fields = {
            "route_available",
            "route_hops",
            "route_hops_norm",
            "route_tx_time_s",
            "route_topology_tx_time_s",
            "route_wireless_tx_time_s",
            "expected_rb_wait_s",
            "expected_wireless_tx_time_s",
            "wireless_pressure",
            "wireless_hops",
            "rb_slowdown",
            "rb_slowdown_tx_extra_s",
            "expected_runtime_penalty_s",
            "estimated_compute_s",
            "effective_cpu",
            "task_cpu",
            "function_budget_s",
            "deadline_slack_s",
            "utility_prior",
            "link_similarity",
            "node_template_similarity",
            "semantic_quality_before",
            "semantic_cumulative_quality_if_selected",
            "semantic_link_truth_score",
            "semantic_link_truth_relation",
        }
        for candidate in candidates:
            metadata = dict(candidate.metadata or {})
            if not any(field in metadata for field in route_fields):
                continue
            target = (str(sfc_id), str(sfc_node_id), str(candidate.instance_id))
            for row in reversed(self.candidate_detail_trace):
                key = (str(row.get("sfc_id", "")), str(row.get("sfc_node_id", "")), str(row.get("instance_id", "")))
                if key == target:
                    row["route_available"] = _float(metadata.get("route_available"), 0.0)
                    row["route_hops"] = int(_float(metadata.get("route_hops"), 0.0))
                    row["route_hops_norm"] = _float(metadata.get("route_hops_norm"), 0.0)
                    row["route_tx_time_s"] = _float(metadata.get("route_tx_time_s"), 0.0)
                    row["route_topology_tx_time_s"] = _float(metadata.get("route_topology_tx_time_s"), 0.0)
                    row["route_wireless_tx_time_s"] = _float(metadata.get("route_wireless_tx_time_s"), 0.0)
                    row["expected_rb_wait_s"] = _float(metadata.get("expected_rb_wait_s"), 0.0)
                    row["expected_wireless_tx_time_s"] = _float(metadata.get("expected_wireless_tx_time_s"), 0.0)
                    row["wireless_pressure"] = _float(metadata.get("wireless_pressure"), 0.0)
                    row["wireless_hops"] = _float(metadata.get("wireless_hops"), 0.0)
                    row["rb_slowdown"] = _float(metadata.get("rb_slowdown"), 1.0)
                    row["rb_slowdown_tx_extra_s"] = _float(metadata.get("rb_slowdown_tx_extra_s"), 0.0)
                    row["topology_score"] = max(0.0, min(1.0, 1.0 - _float(metadata.get("topology_risk"), 0.0)))
                    row["topology_risk"] = _float(metadata.get("topology_risk"), 0.0)
                    row["mobility_risk"] = _float(metadata.get("mobility_risk"), 0.0)
                    row["expected_runtime_penalty_s"] = _float(metadata.get("expected_runtime_penalty_s"), 0.0)
                    row["estimated_compute_s"] = _float(metadata.get("estimated_compute_s"), 0.0)
                    row["effective_cpu"] = _float(metadata.get("effective_cpu"), 0.0)
                    row["task_cpu"] = _float(metadata.get("task_cpu"), 0.0)
                    row["function_budget_s"] = _float(metadata.get("function_budget_s"), 0.0)
                    row["deadline_slack_s"] = _float(metadata.get("deadline_slack_s"), 0.0)
                    row["utility_prior"] = _float(metadata.get("utility_prior"), 0.0)
                    row["link_similarity"] = _float(metadata.get("link_similarity"), row.get("semantic_score", 0.0))
                    row["node_template_similarity"] = _float(metadata.get("node_template_similarity"), 0.0)
                    row["semantic_quality_before"] = _float(metadata.get("semantic_quality_before"), 1.0)
                    row["semantic_cumulative_quality"] = _float(metadata.get("semantic_cumulative_quality_if_selected"), row["link_similarity"])
                    row["semantic_link_truth_score"] = _float(metadata.get("semantic_link_truth_score"), 0.0)
                    row["semantic_link_truth_relation"] = str(
                        metadata.get("semantic_link_truth_relation", row.get("semantic_link_truth_relation", "")) or ""
                    )
                    break

    def _append_candidate_detail_rows(
        self,
        request: DiscoveryRequest,
        candidates: Sequence[CatalogCandidate],
    ) -> None:
        exchange_cfg = self.exchange.config
        for rank, candidate in enumerate(candidates, start=1):
            metadata = dict(candidate.metadata or {})
            topology_risk = _float(metadata.get("topology_risk"), 0.0)
            mobility_risk = _float(metadata.get("mobility_risk"), 0.0)
            ttl_s = float(exchange_cfg.ttl_s)
            ttl_retained_stale = (
                bool(candidate.is_remote)
                and str(metadata.get("semantic_group", "")) == "stale_clone_exact"
                and ttl_s >= 6.0
            )
            effective_staleness_s = float(candidate.staleness_s)
            if ttl_retained_stale:
                effective_staleness_s = max(effective_staleness_s, ttl_s * 0.5)
            self.candidate_detail_trace.append(
                {
                    "time_s": float(request.created_at_s),
                    "request_id": request.request_id,
                    "agent_id": request.agent_id,
                    "sfc_id": request.sfc_id,
                    "sfc_node_id": request.sfc_node_id,
                    "function_id": request.sfc_node_id,
                    "service_id": request.service_id,
                    "candidate_id": candidate.instance_id,
                    "instance_id": candidate.instance_id,
                    "rank": int(rank),
                    "semantic_score": float(candidate.semantic_score),
                    "link_similarity": _float(metadata.get("link_similarity"), candidate.semantic_score),
                    "node_template_similarity": _float(metadata.get("node_template_similarity"), candidate.semantic_score),
                    "link_source_semantic": str(metadata.get("link_source_semantic", request.link_input_semantic or "")),
                    "candidate_input_semantic": str(metadata.get("candidate_input_semantic", getattr(candidate, "input_semantic", ""))),
                    "candidate_output_semantic": str(metadata.get("candidate_output_semantic", getattr(candidate, "output_semantic", ""))),
                    "semantic_link_truth_score": _float(metadata.get("semantic_link_truth_score"), 0.0),
                    "semantic_link_truth_relation": str(metadata.get("semantic_link_truth_relation", "")),
                    "semantic_link_truth_cell": str(metadata.get("semantic_link_truth_cell", "")),
                    "implementation_id": str(metadata.get("implementation_id", "")),
                    "semantic_variant_type": str(metadata.get("semantic_variant_type", "")),
                    "semantic_quality_before": _float(metadata.get("semantic_quality_before"), 1.0),
                    "semantic_cumulative_quality": _float(metadata.get("semantic_cumulative_quality_if_selected"), candidate.semantic_score),
                    "topology_score": max(0.0, min(1.0, 1.0 - topology_risk)),
                    "topology_risk": topology_risk,
                    "mobility_risk": mobility_risk,
                    "cold_start_s": _float(metadata.get("cold_start_s"), 0.0),
                    "stale_latency_penalty_s": _float(metadata.get("stale_latency_penalty_s"), 0.0),
                    "estimated_compute_s": _float(metadata.get("estimated_compute_s"), 0.0),
                    "route_tx_time_s": _float(metadata.get("route_tx_time_s"), 0.0),
                    "route_topology_tx_time_s": _float(metadata.get("route_topology_tx_time_s"), 0.0),
                    "route_wireless_tx_time_s": _float(metadata.get("route_wireless_tx_time_s"), 0.0),
                    "expected_rb_wait_s": _float(metadata.get("expected_rb_wait_s"), 0.0),
                    "expected_wireless_tx_time_s": _float(metadata.get("expected_wireless_tx_time_s"), 0.0),
                    "wireless_pressure": _float(metadata.get("wireless_pressure"), 0.0),
                    "wireless_hops": _float(metadata.get("wireless_hops"), 0.0),
                    "rb_slowdown": _float(metadata.get("rb_slowdown"), 1.0),
                    "rb_slowdown_tx_extra_s": _float(metadata.get("rb_slowdown_tx_extra_s"), 0.0),
                    "effective_cpu": _float(metadata.get("effective_cpu"), 0.0),
                    "task_cpu": _float(metadata.get("task_cpu"), 0.0),
                    "expected_runtime_penalty_s": _float(metadata.get("expected_runtime_penalty_s"), 0.0),
                    "function_budget_s": _float(metadata.get("function_budget_s"), 0.0),
                    "deadline_slack_s": _float(metadata.get("deadline_slack_s"), 0.0),
                    "utility_prior": _float(metadata.get("utility_prior"), 0.0),
                    "stale": bool(candidate.staleness_s > 0.0 or ttl_retained_stale),
                    "staleness_s": effective_staleness_s,
                    "local_or_remote": "remote" if candidate.is_remote else "local",
                    "is_remote": bool(candidate.is_remote),
                    "source": candidate.source,
                    "owner_agent_id": candidate.owner_agent_id,
                    "selected": False,
                    "selected_at_s": "",
                    "node_id": candidate.node_id,
                    "node_type": candidate.node_type,
                    "region_id": candidate.region_id,
                    "payload_bytes": int(candidate.payload_bytes or 0),
                    "semantic_group": str(metadata.get("semantic_group", "")),
                    "is_decoy": bool(metadata.get("is_decoy", False)),
                    "exchange_ttl_s": ttl_s,
                    "exchange_radius_hops": int(getattr(exchange_cfg, "radius_hops", 1)),
                    "exchange_top_k": int(exchange_cfg.top_k_per_agent),
                    "semantic_compressed_dim": int(exchange_cfg.compressed_dim),
                }
            )


def _float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _normalize_neighbor_map(neighbor_map: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, values in dict(neighbor_map or {}).items():
        if isinstance(values, Mapping):
            normalized[str(key)] = {str(item): int(hops or 1) for item, hops in values.items()}
        else:
            normalized[str(key)] = [str(item) for item in values]
    return normalized
