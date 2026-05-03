from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .instance_directory import ServiceInstance, ServiceInstanceDirectory
from .semantic_cache import SemanticAdvertisement, SemanticAdvertisementCache
from .semantic_encoder import SemanticEncoder, semantic_value_text, service_instance_text, sfc_node_text
from .semantic_exchange import SemanticCompressor


@dataclass(frozen=True)
class CatalogCandidate:
    instance_id: str
    service_id: str
    node_id: str
    region_id: str
    node_type: str
    source: str
    owner_agent_id: str
    semantic_score: float
    staleness_s: float
    is_remote: bool
    payload_bytes: int = 0
    input_semantic: str = "any"
    output_semantic: str = "any"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "service_id": self.service_id,
            "node_id": self.node_id,
            "region_id": self.region_id,
            "node_type": self.node_type,
            "source": self.source,
            "owner_agent_id": self.owner_agent_id,
            "semantic_score": self.semantic_score,
            "staleness_s": self.staleness_s,
            "is_remote": self.is_remote,
            "payload_bytes": self.payload_bytes,
            "input_semantic": self.input_semantic,
            "output_semantic": self.output_semantic,
            "metadata": dict(self.metadata),
        }


class DistributedServiceCatalog:
    """Per-agent catalog that only sees local instances plus exchanged summaries."""

    def __init__(
        self,
        agent_id: str,
        local_directory: Optional[ServiceInstanceDirectory] = None,
        encoder: Optional[SemanticEncoder] = None,
        compressor: Optional[SemanticCompressor] = None,
        respect_local_visibility: bool = True,
    ):
        self.agent_id = str(agent_id)
        self.local_directory = local_directory or ServiceInstanceDirectory()
        self.encoder = encoder or SemanticEncoder(backend="hash")
        self.compressor = compressor or SemanticCompressor(input_dim=self.encoder.embedding_dim)
        self.respect_local_visibility = bool(respect_local_visibility)
        self.remote_cache = SemanticAdvertisementCache(owner_agent_id=self.agent_id)
        self._local_embedding_by_instance: Dict[str, np.ndarray] = {}
        self.query_trace: List[Dict[str, Any]] = []

    def refresh_local_embeddings(self) -> None:
        for instance in self.local_directory.all():
            self._local_embedding_by_instance[instance.instance_id] = self.encoder.encode(service_instance_text(instance))

    def ingest_remote(self, ads: Iterable[SemanticAdvertisement], now_s: float) -> int:
        return self.remote_cache.upsert_many(ads, now_s)

    def query(
        self,
        query_text: str,
        service_id: Optional[str] = None,
        required_capabilities: Sequence[str] = (),
        allowed_node_types: Sequence[str] = (),
        now_s: float = 0.0,
        top_k: int = 8,
        min_similarity: float = -1.0,
        include_remote: bool = True,
        link_input_semantic: Optional[str] = None,
    ) -> List[CatalogCandidate]:
        self.refresh_local_embeddings()
        required = set(required_capabilities or [])
        allowed_types = {str(item) for item in allowed_node_types or ()}
        query_vec = self.encoder.encode(query_text)
        link_vec = self.encoder.encode(semantic_value_text(link_input_semantic)) if link_input_semantic is not None else None
        candidates: List[CatalogCandidate] = []

        for instance in self.local_directory.all():
            if not _instance_matches(instance, service_id, required):
                continue
            if allowed_types and str(instance.node_type) not in allowed_types:
                continue
            if self.respect_local_visibility and instance.metadata.get("local_catalog_visible") is False:
                continue
            vector = self._local_embedding_by_instance.get(instance.instance_id)
            if vector is None:
                vector = self.encoder.encode(service_instance_text(instance))
            template_score = _biased_semantic_score(
                float(self.encoder.similarity(query_vec, vector.reshape(1, -1))[0]),
                instance.metadata,
            )
            score = template_score
            if link_vec is not None:
                score = _semantic_fidelity(
                    _biased_semantic_score(
                        float(self.encoder.similarity(link_vec, self.encoder.encode(semantic_value_text(instance.input_semantic)).reshape(1, -1))[0]),
                        instance.metadata,
                    )
                )
            if score >= min_similarity:
                metadata = _candidate_metadata(instance)
                link_metadata = _semantic_link_metadata(link_input_semantic, instance.input_semantic, score)
                metadata.update(
                    {
                        "node_template_similarity": template_score,
                        "link_similarity": score,
                        "link_source_semantic": str(link_input_semantic or ""),
                        "candidate_input_semantic": str(instance.input_semantic),
                        "candidate_output_semantic": str(instance.output_semantic),
                        **link_metadata,
                    }
                )
                candidates.append(
                    CatalogCandidate(
                        instance_id=instance.instance_id,
                        service_id=instance.service_id,
                        node_id=instance.node_id,
                        region_id=instance.region_id,
                        node_type=instance.node_type,
                        source="local",
                        owner_agent_id=self.agent_id,
                        semantic_score=score,
                        staleness_s=0.0,
                        is_remote=False,
                        input_semantic=str(instance.input_semantic),
                        output_semantic=str(instance.output_semantic),
                        metadata=metadata,
                    )
                )

        if include_remote:
            for ad in self.remote_cache.fresh(now_s):
                if service_id is not None and ad.service_id != service_id:
                    continue
                if required and not required.issubset(set(ad.capabilities)):
                    continue
                if allowed_types and str(ad.node_type) not in allowed_types:
                    continue
                template_score = _biased_semantic_score(
                    self.compressor.compressed_similarity(query_vec, ad.compressed_embedding),
                    ad.metadata,
                )
                score = template_score
                if link_vec is not None:
                    score = _semantic_fidelity(
                        _biased_semantic_score(
                            float(self.encoder.similarity(link_vec, self.encoder.encode(semantic_value_text(ad.input_semantic)).reshape(1, -1))[0]),
                            ad.metadata,
                        )
                    )
                if score < min_similarity:
                    continue
                metadata = _remote_candidate_metadata(ad, now_s)
                link_metadata = _semantic_link_metadata(link_input_semantic, ad.input_semantic, score)
                metadata.update(
                    {
                        "node_template_similarity": template_score,
                        "link_similarity": score,
                        "link_source_semantic": str(link_input_semantic or ""),
                        "candidate_input_semantic": str(ad.input_semantic),
                        "candidate_output_semantic": str(ad.output_semantic),
                        **link_metadata,
                    }
                )
                candidates.append(
                    CatalogCandidate(
                        instance_id=ad.instance_id,
                        service_id=ad.service_id,
                        node_id=ad.node_id,
                        region_id=ad.region_id,
                        node_type=ad.node_type,
                        source="remote_advertisement",
                        owner_agent_id=ad.source_agent_id,
                        semantic_score=score,
                        staleness_s=ad.age_s(now_s),
                        is_remote=True,
                        payload_bytes=ad.payload_bytes,
                        input_semantic=str(ad.input_semantic),
                        output_semantic=str(ad.output_semantic),
                        metadata=metadata,
                    )
                )

        candidates.sort(key=lambda item: (-item.semantic_score, item.is_remote, item.staleness_s, item.instance_id))
        result = _diverse_top_k(candidates, max(0, int(top_k)))
        self.query_trace.append(
            {
                "agent_id": self.agent_id,
                "time_s": now_s,
                "service_id": service_id or "",
                "required_capabilities": sorted(required),
                "candidate_count": len(candidates),
                "returned_count": len(result),
                "remote_count": sum(1 for item in result if item.is_remote),
                "stale_ratio": self.remote_cache.stale_ratio(now_s),
                "top_score": result[0].semantic_score if result else None,
            }
        )
        return result

    def query_sfc_node(self, sfc_node: Any, payload_semantic: str = "", **kwargs: Any) -> List[CatalogCandidate]:
        return self.query(
            query_text=sfc_node_text(sfc_node, payload_semantic=payload_semantic),
            service_id=getattr(sfc_node, "service_type", None),
            required_capabilities=tuple(getattr(sfc_node, "required_capabilities", ()) or ()),
            **kwargs,
        )

    def stale_ratio(self, now_s: float) -> float:
        return self.remote_cache.stale_ratio(now_s)


def split_directory_by_region(directory: ServiceInstanceDirectory) -> Dict[str, ServiceInstanceDirectory]:
    per_region: Dict[str, ServiceInstanceDirectory] = {}
    for instance in directory.all():
        region = str(instance.region_id)
        per_region.setdefault(region, ServiceInstanceDirectory()).upsert(instance)
    return per_region


def _instance_matches(instance: ServiceInstance, service_id: Optional[str], required: set[str]) -> bool:
    if service_id is not None and instance.service_id != service_id:
        return False
    if required and not required.issubset(set(instance.capabilities)):
        return False
    if not instance.is_selectable():
        return False
    return True


def _biased_semantic_score(base_score: float, metadata: Mapping[str, Any]) -> float:
    """Apply scenario-controlled semantic bias for calibrated decoy candidates."""

    bias = 0.0
    try:
        bias = float(dict(metadata or {}).get("semantic_score_bias", 0.0) or 0.0)
    except Exception:
        bias = 0.0
    return max(-1.0, min(1.0, float(base_score) + bias))


def _semantic_fidelity(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _semantic_link_metadata(source_semantic: Optional[str], target_semantic: Any, score: float) -> Dict[str, Any]:
    source = _normalize_semantic_label(source_semantic)
    target = _normalize_semantic_label(target_semantic)
    fidelity = _semantic_fidelity(score)
    if source == target and source != "any":
        relation = "exact"
        label_score = 1.0
    elif target == "any":
        relation = "generic_accept"
        label_score = 0.90
    elif source == "any":
        relation = "generic_source"
        label_score = 0.75
    elif fidelity >= 0.85:
        relation = "strong"
        label_score = 0.85
    elif fidelity >= 0.65:
        relation = "compatible"
        label_score = 0.65
    elif fidelity >= 0.45:
        relation = "weak"
        label_score = 0.45
    else:
        relation = "mismatch"
        label_score = 0.0
    return {
        "semantic_link_relation": relation,
        "semantic_link_label_score": label_score,
        "semantic_link_matrix_cell": f"{source}->{target}:{relation}",
    }


def _normalize_semantic_label(value: Any) -> str:
    text = str(value or "any").strip().lower()
    return "_".join(text.replace("/", " ").replace("-", " ").split()) or "any"


def _candidate_metadata(instance: ServiceInstance) -> Dict[str, Any]:
    raw = dict(instance.metadata or {})
    load_ratio = _float(raw.get("load_ratio_override"), instance.load_ratio())
    raw.update(
        {
            "health_score": instance.health_score,
            "load_ratio": load_ratio,
            "capacity_cpu": _float(instance.capacity.get("cpu"), 0.0),
            "available_cpu": _float(instance.available("cpu"), 0.0),
            "max_concurrency": int(instance.max_concurrency),
            "current_load": int(instance.current_load),
            "topology_risk": _float(raw.get("topology_risk"), 0.0),
            "mobility_risk": _float(raw.get("mobility_risk"), 0.0),
            "semantic_group": str(raw.get("semantic_group", "")),
            "is_decoy": bool(raw.get("is_decoy", False)),
            "semantic_score_bias": _float(raw.get("semantic_score_bias"), 0.0),
            "cold_start_s": float(instance.cold_start_s),
        }
    )
    return raw


def _remote_candidate_metadata(ad: Any, now_s: float) -> Dict[str, Any]:
    raw = dict(getattr(ad, "metadata", {}) or {})
    if "load_ratio" not in raw and "load_ratio_override" in raw:
        raw["load_ratio"] = _float(raw.get("load_ratio_override"), 0.0)
    raw.setdefault("freshness", ad.freshness(now_s))
    raw.setdefault("topology_risk", _float(raw.get("topology_risk"), 0.0))
    raw.setdefault("mobility_risk", _float(raw.get("mobility_risk"), 0.0))
    raw.setdefault("semantic_group", str(raw.get("semantic_group", "")))
    raw.setdefault("is_decoy", bool(raw.get("is_decoy", False)))
    raw.setdefault("semantic_score_bias", _float(raw.get("semantic_score_bias"), 0.0))
    return raw


def _diverse_top_k(candidates: Sequence[CatalogCandidate], top_k: int) -> List[CatalogCandidate]:
    """Return semantically ranked candidates while preserving topology trade-offs.

    The discovery layer is semantic-first, but a pure semantic top-k can remove
    all low-risk candidates before topology-aware policies or MARL observe them.
    This selector keeps the best semantic candidates and then fills the remaining
    budget with the best candidates from each calibrated semantic/topology group
    and low-risk candidates. It does not change the selected action; it only
    exposes the trade-off to the policy.
    """

    if top_k <= 0:
        return []
    ranked = list(candidates)
    if len(ranked) <= top_k:
        return ranked
    selected: List[CatalogCandidate] = []
    seen: set[str] = set()

    def add(candidate: CatalogCandidate) -> None:
        if len(selected) >= top_k or candidate.instance_id in seen:
            return
        selected.append(candidate)
        seen.add(candidate.instance_id)

    semantic_quota = max(1, min(len(ranked), top_k // 2))
    for candidate in ranked[:semantic_quota]:
        add(candidate)

    group_order = [
        "semantic_high_topology_bad",
        "semantic_medium_topology_good",
        "semantic_low_topology_good",
        "stale_remote_candidates",
    ]
    for group in group_order:
        group_candidates = [
            item
            for item in ranked
            if str(dict(item.metadata or {}).get("semantic_group", "")) == group
        ]
        group_candidates.sort(
            key=lambda item: (
                _float(dict(item.metadata or {}).get("topology_risk"), 0.0),
                _float(dict(item.metadata or {}).get("load_ratio"), 0.0),
                -item.semantic_score,
                item.instance_id,
            )
        )
        if group_candidates:
            add(group_candidates[0])

    low_risk = sorted(
        ranked,
        key=lambda item: (
            _float(dict(item.metadata or {}).get("topology_risk"), 0.0),
            _float(dict(item.metadata or {}).get("mobility_risk"), 0.0),
            _float(dict(item.metadata or {}).get("load_ratio"), 0.0),
            -item.semantic_score,
            item.instance_id,
        ),
    )
    for candidate in low_risk:
        add(candidate)
    for candidate in ranked:
        add(candidate)
    return selected[:top_k]


def _float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)
