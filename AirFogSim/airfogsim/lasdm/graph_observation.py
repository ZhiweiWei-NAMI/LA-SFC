from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .distributed_catalog import CatalogCandidate
from .temporal_state_buffer import TemporalStateBuffer
from .topology_builder import DynamicTopology, TopologyEdge, TopologyNode


NODE_TYPES = ("vehicle", "uav", "rsu", "cloud_server", "unknown")
LINK_TYPES = ("v2v", "v2u", "u2v", "v2i", "i2v", "u2i", "i2u", "u2u", "i2i", "i2c", "c2i", "other")


@dataclass(frozen=True)
class GraphObservationConfig:
    max_nodes: int = 128
    max_edges: int = 512
    max_candidates: int = 16
    position_scale_m: float = 5000.0
    cpu_scale: float = 100.0
    memory_scale: float = 65536.0
    storage_scale: float = 1_000_000.0
    distance_scale_m: float = 5000.0
    rate_scale_mbps: float = 1000.0
    latency_scale_s: float = 1.0
    route_tx_scale_s: float = 20.0
    expected_penalty_scale_s: float = 40.0
    compute_scale_s: float = 20.0
    wireless_pressure_scale: float = 8.0
    rb_slowdown_scale: float = 8.0
    include_topology_features: bool = True
    include_temporal_features: bool = True
    include_semantic_features: bool = True


class GraphObservationBuilder:
    """Converts dynamic topology + candidate sets into fixed-size MARL observations."""

    def __init__(self, config: Optional[GraphObservationConfig] = None):
        self.config = config or GraphObservationConfig()

    def build(
        self,
        agent_id: str,
        topology: DynamicTopology,
        candidate_sets: Sequence[Mapping[str, Any]],
        temporal_buffer: Optional[TemporalStateBuffer] = None,
    ) -> Dict[str, Any]:
        node_ids = sorted(topology.nodes)[: self.config.max_nodes]
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        node_features = np.zeros((self.config.max_nodes, self.node_feature_dim), dtype=np.float32)
        node_mask = np.zeros((self.config.max_nodes,), dtype=np.float32)
        for idx, node_id in enumerate(node_ids):
            node_features[idx] = self.node_features(topology.nodes[node_id], agent_id)
            node_mask[idx] = 1.0

        edge_index = np.zeros((2, self.config.max_edges), dtype=np.int64)
        edge_features = np.zeros((self.config.max_edges, self.edge_feature_dim), dtype=np.float32)
        edge_mask = np.zeros((self.config.max_edges,), dtype=np.float32)
        edge_cursor = 0
        for edge in topology.edges:
            if edge.src not in node_index or edge.dst not in node_index:
                continue
            if edge_cursor >= self.config.max_edges:
                break
            edge_index[:, edge_cursor] = [node_index[edge.src], node_index[edge.dst]]
            edge_features[edge_cursor] = self.edge_features(edge)
            edge_mask[edge_cursor] = 1.0
            edge_cursor += 1

        encoded_candidate_sets = [self._encode_candidate_set(item) for item in candidate_sets]
        if not self.config.include_topology_features:
            node_features[:] = 0.0
            edge_index[:] = 0
            edge_features[:] = 0.0
            edge_mask[:] = 0.0
        temporal = temporal_buffer.summarize().to_dict() if (temporal_buffer is not None and self.config.include_temporal_features) else {}
        return {
            "agent_id": str(agent_id),
            "time_s": float(topology.time_s),
            "node_ids": node_ids,
            "node_features": node_features,
            "node_mask": node_mask,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_mask": edge_mask,
            "candidate_sets": encoded_candidate_sets,
            "temporal_features": temporal,
            "topology_metadata": dict(topology.metadata),
        }

    @property
    def node_feature_dim(self) -> int:
        return len(NODE_TYPES) + 11

    @property
    def edge_feature_dim(self) -> int:
        return len(LINK_TYPES) + 5

    @property
    def candidate_feature_dim(self) -> int:
        return 31

    def node_features(self, node: TopologyNode, agent_id: str) -> np.ndarray:
        one_hot = _one_hot(node.node_type, NODE_TYPES)
        pos = np.asarray(node.position, dtype=np.float32) / max(1e-9, self.config.position_scale_m)
        numeric = np.asarray(
            [
                node.cpu / self.config.cpu_scale,
                node.memory / self.config.memory_scale,
                node.storage / self.config.storage_scale,
                node.load_ratio,
                node.trust_score,
                min(1.0, node.energy_consumption / 1_000.0),
                min(1.0, node.service_count / 16.0),
                1.0 if node.region_id == str(agent_id) else 0.0,
                *pos[:3],
            ],
            dtype=np.float32,
        )
        return np.concatenate([one_hot, numeric], axis=0).astype(np.float32)

    def edge_features(self, edge: TopologyEdge) -> np.ndarray:
        one_hot = _one_hot(edge.link_type, LINK_TYPES, default_value="other")
        numeric = np.asarray(
            [
                min(1.0, edge.distance_m / self.config.distance_scale_m),
                min(1.0, edge.rate_mbps / self.config.rate_scale_mbps),
                min(1.0, edge.latency_s / self.config.latency_scale_s),
                edge.reliability,
                1.0 if edge.is_wireless else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([one_hot, numeric], axis=0).astype(np.float32)

    def candidate_features(self, candidate: CatalogCandidate, agent_id: str) -> np.ndarray:
        raw_semantic_score = float(candidate.semantic_score or 0.0)
        semantic_score = raw_semantic_score if self.config.include_semantic_features else 0.0
        staleness = min(1.0, candidate.staleness_s / 10.0) if self.config.include_temporal_features else 0.0
        metadata = dict(candidate.metadata or {})
        topology_enabled = self.config.include_topology_features
        temporal_enabled = self.config.include_temporal_features
        topology_risk = min(1.0, float(metadata.get("topology_risk", 0.0) or 0.0)) if topology_enabled else 0.0
        mobility_risk = min(1.0, float(metadata.get("mobility_risk", 0.0) or 0.0)) if temporal_enabled else 0.0
        cold_start_s = min(1.0, float(metadata.get("cold_start_s", 0.0) or 0.0) / 5.0) if topology_enabled else 0.0
        semantic_mismatch = max(0.0, 1.0 - semantic_score) if self.config.include_semantic_features else 0.0
        utility_value = float(metadata.get("utility_prior", 0.0) or 0.0)
        if topology_enabled and not self.config.include_semantic_features:
            utility_value -= raw_semantic_score
        utility_prior = max(-1.0, min(1.0, utility_value)) if topology_enabled else 0.0
        deadline_slack = max(-1.0, min(1.0, float(metadata.get("deadline_slack_s", 0.0) or 0.0) / 20.0)) if topology_enabled else 0.0
        route_tx_time = (
            min(1.0, float(metadata.get("route_tx_time_s", 0.0) or 0.0) / max(1e-9, self.config.route_tx_scale_s))
            if topology_enabled
            else 0.0
        )
        expected_penalty = min(
            1.0,
            float(metadata.get("expected_runtime_penalty_s", 0.0) or 0.0) / max(1e-9, self.config.expected_penalty_scale_s),
        ) if topology_enabled else 0.0
        estimated_compute = (
            min(1.0, float(metadata.get("estimated_compute_s", 0.0) or 0.0) / max(1e-9, self.config.compute_scale_s))
            if topology_enabled
            else 0.0
        )
        deadline_violation = 1.0 if topology_enabled and float(metadata.get("deadline_slack_s", 0.0) or 0.0) < 0.0 else 0.0
        expected_rb_wait = (
            min(1.0, float(metadata.get("expected_rb_wait_s", 0.0) or 0.0) / max(1e-9, self.config.route_tx_scale_s))
            if temporal_enabled
            else 0.0
        )
        wireless_pressure = min(
            1.0,
            float(metadata.get("wireless_pressure", 0.0) or 0.0) / max(1e-9, self.config.wireless_pressure_scale),
        ) if temporal_enabled else 0.0
        wireless_hops = min(1.0, float(metadata.get("wireless_hops", 0.0) or 0.0) / 4.0) if topology_enabled else 0.0
        rb_slowdown = (
            min(1.0, float(metadata.get("rb_slowdown", 1.0) or 1.0) / max(1e-9, self.config.rb_slowdown_scale))
            if temporal_enabled
            else 0.0
        )
        resource_available_ratio = (
            min(1.0, max(0.0, float(metadata.get("resource_available_ratio", 1.0) or 0.0)))
            if topology_enabled
            else 0.0
        )
        hops_from_prev = (
            min(1.0, max(0.0, float(metadata.get("hops_from_prev_function_norm", metadata.get("route_hops_norm", 1.0)) or 0.0)))
            if topology_enabled
            else 0.0
        )
        remaining_deadline_ratio = (
            min(1.0, max(0.0, float(metadata.get("remaining_deadline_ratio", 1.0) or 0.0)))
            if topology_enabled
            else 0.0
        )
        node_type = str(candidate.node_type)
        return np.asarray(
            [
                semantic_score,
                staleness,
                1.0 if candidate.is_remote else 0.0,
                min(1.0, float(candidate.metadata.get("freshness", 1.0))) if temporal_enabled else 0.0,
                min(1.0, float(candidate.metadata.get("health_score", 1.0))),
                min(1.0, float(candidate.metadata.get("load_ratio", 0.0))) if topology_enabled else 0.0,
                min(1.0, candidate.payload_bytes / 4096.0),
                1.0 if topology_enabled and node_type == "uav" else 0.0,
                1.0 if topology_enabled and node_type == "vehicle" else 0.0,
                1.0 if topology_enabled and node_type == "rsu" else 0.0,
                1.0 if topology_enabled and node_type == "cloud_server" else 0.0,
                topology_risk,
                mobility_risk,
                1.0 if topology_enabled and str(candidate.region_id) == str(agent_id) else 0.0,
                min(1.0, float(metadata.get("route_hops_norm", 1.0) or 0.0)) if topology_enabled else 0.0,
                min(1.0, float(metadata.get("route_available", 1.0) or 0.0)) if topology_enabled else 0.0,
                cold_start_s,
                semantic_mismatch,
                utility_prior,
                deadline_slack,
                route_tx_time,
                expected_penalty,
                estimated_compute,
                deadline_violation,
                expected_rb_wait,
                wireless_pressure,
                wireless_hops,
                rb_slowdown,
                resource_available_ratio,
                hops_from_prev,
                remaining_deadline_ratio,
            ],
            dtype=np.float32,
        )

    def _encode_candidate_set(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        candidates: Sequence[CatalogCandidate] = list(item.get("candidates", []))
        features = np.zeros((self.config.max_candidates, self.candidate_feature_dim), dtype=np.float32)
        mask = np.zeros((self.config.max_candidates,), dtype=np.float32)
        ids: List[str] = []
        for idx, candidate in enumerate(candidates[: self.config.max_candidates]):
            features[idx] = self.candidate_features(candidate, str(item.get("agent_id", "")))
            mask[idx] = 1.0
            ids.append(candidate.instance_id)
        return {
            "sfc_id": item.get("sfc_id", ""),
            "sfc_node_id": item.get("sfc_node_id", ""),
            "sfc_node_index": int(item.get("sfc_node_index", 0) or 0),
            "source_node_id": item.get("source_node_id", ""),
            "predecessor_node_ids": list(item.get("predecessor_node_ids", []) or []),
            "candidate_ids": ids,
            "candidate_features": features,
            "candidate_mask": mask,
            "raw_candidates": [candidate.to_dict() for candidate in candidates[: self.config.max_candidates]],
        }


def flatten_observation(observation: Mapping[str, Any]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    node_features = np.asarray(observation.get("node_features", []), dtype=np.float32)
    node_mask = np.asarray(observation.get("node_mask", []), dtype=np.float32).reshape(-1)
    edge_features = np.asarray(observation.get("edge_features", []), dtype=np.float32)
    edge_mask = np.asarray(observation.get("edge_mask", []), dtype=np.float32).reshape(-1)
    chunks.append(np.asarray([float(node_mask.sum()), float(edge_mask.sum())], dtype=np.float32))
    chunks.append(_feature_summary(node_features, node_mask))
    chunks.append(_feature_summary(edge_features, edge_mask))
    temporal = observation.get("temporal_features", {}) or {}
    if temporal:
        chunks.append(np.asarray([float(temporal[key]) for key in sorted(temporal)], dtype=np.float32))
    for candidate_set in observation.get("candidate_sets", []) or []:
        chunks.append(np.asarray(candidate_set.get("candidate_features", []), dtype=np.float32).reshape(-1))
        chunks.append(np.asarray(candidate_set.get("candidate_mask", []), dtype=np.float32).reshape(-1))
    if not chunks:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _feature_summary(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return np.zeros((4,), dtype=np.float32)
    if mask.size == arr.shape[0] and bool(np.any(mask > 0.0)):
        arr = arr[mask > 0.0]
    if arr.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.concatenate(
        [
            arr.mean(axis=0),
            arr.std(axis=0),
            arr.min(axis=0),
            arr.max(axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def _one_hot(value: str, choices: Sequence[str], default_value: str = "unknown") -> np.ndarray:
    value = str(value)
    if value not in choices:
        value = default_value if default_value in choices else choices[-1]
    arr = np.zeros((len(choices),), dtype=np.float32)
    arr[list(choices).index(value)] = 1.0
    return arr
