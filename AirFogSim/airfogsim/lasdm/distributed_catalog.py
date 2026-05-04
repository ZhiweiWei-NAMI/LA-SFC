from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .instance_directory import ServiceInstance, ServiceInstanceDirectory
from .semantic_link_matrix import SemanticLinkMatrix
from .semantic_link_predictor import SemanticLinkScorer
from .semantic_cache import SemanticAdvertisement, SemanticAdvertisementCache
from .semantic_encoder import SemanticEncoder, service_instance_text, sfc_node_text
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
        semantic_scorer: Optional[SemanticLinkScorer] = None,
        semantic_matrix: Optional[SemanticLinkMatrix] = None,
    ):
        self.agent_id = str(agent_id)
        self.local_directory = local_directory or ServiceInstanceDirectory()
        self.encoder = encoder or SemanticEncoder(backend="hash")
        self.compressor = compressor or SemanticCompressor(input_dim=self.encoder.embedding_dim)
        self.respect_local_visibility = bool(respect_local_visibility)
        self.semantic_scorer = semantic_scorer
        self.semantic_matrix = semantic_matrix
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
        request_type: str = "",
        chain_position: int = 0,
    ) -> List[CatalogCandidate]:
        self.refresh_local_embeddings()
        required = set(required_capabilities or [])
        allowed_types = {str(item) for item in allowed_node_types or ()}
        query_vec = self.encoder.encode(query_text)
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
            template_score = _semantic_fidelity(float(self.encoder.similarity(query_vec, vector.reshape(1, -1))[0]))
            score = self._score_candidate_for_discovery(
                query_text=query_text,
                candidate_metadata=instance.metadata,
                service_id=instance.service_id,
                node_type=instance.node_type,
                region_id=instance.region_id,
                is_remote=False,
                chain_position=chain_position,
                template_score=template_score,
            )
            if score >= min_similarity:
                metadata = _candidate_metadata(instance)
                metadata.update(
                    {
                        "node_template_similarity": template_score,
                        "link_similarity": score,
                        "link_source_semantic": str(link_input_semantic or ""),
                        "candidate_input_semantic": str(instance.input_semantic),
                        "candidate_output_semantic": str(instance.output_semantic),
                        "request_context_text": str(query_text),
                        "chain_position": int(chain_position),
                    }
                )
                metadata.update(self._truth_metadata(request_type, instance.service_id, metadata, link_input_semantic))
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
                template_score = _semantic_fidelity(self.compressor.compressed_similarity(query_vec, ad.compressed_embedding))
                score = self._score_candidate_for_discovery(
                    query_text=query_text,
                    candidate_metadata=ad.metadata,
                    service_id=ad.service_id,
                    node_type=ad.node_type,
                    region_id=ad.region_id,
                    is_remote=True,
                    chain_position=chain_position,
                    template_score=template_score,
                    staleness_s=ad.age_s(now_s),
                )
                if score < min_similarity:
                    continue
                metadata = _remote_candidate_metadata(ad, now_s)
                metadata.update(
                    {
                        "node_template_similarity": template_score,
                        "link_similarity": score,
                        "link_source_semantic": str(link_input_semantic or ""),
                        "candidate_input_semantic": str(ad.input_semantic),
                        "candidate_output_semantic": str(ad.output_semantic),
                        "request_context_text": str(query_text),
                        "chain_position": int(chain_position),
                    }
                )
                metadata.update(self._truth_metadata(request_type, ad.service_id, metadata, link_input_semantic))
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

    def _score_candidate_for_discovery(
        self,
        query_text: str,
        candidate_metadata: Mapping[str, Any],
        service_id: str,
        node_type: str,
        region_id: str,
        is_remote: bool,
        chain_position: int,
        template_score: float,
        staleness_s: float = 0.0,
    ) -> float:
        scorer = self.semantic_scorer
        if scorer is None:
            return _semantic_fidelity(template_score)
        service_idx = scorer.service_type_index(service_id, dict(candidate_metadata or {}).get("service_type_idx"))
        profile_text = str(dict(candidate_metadata or {}).get("profile_text", "") or "")
        if not profile_text:
            profile_text = str(dict(candidate_metadata or {}).get("semantic_description", "") or service_id)
        return _semantic_fidelity(
            scorer.score_for_discovery(
                request_context_text=query_text,
                instance_profile_text=profile_text,
                service_type_idx=service_idx,
                chain_position=int(chain_position),
                node_type=str(node_type),
                is_remote=bool(is_remote),
                is_same_region=str(region_id) == self.agent_id,
                staleness_s=float(staleness_s),
                topology_risk=float(dict(candidate_metadata or {}).get("topology_risk", 0.0) or 0.0),
            )
        )

    def _truth_metadata(
        self,
        request_type: str,
        service_id: str,
        candidate_metadata: Mapping[str, Any],
        link_input_semantic: Optional[str],
    ) -> Dict[str, Any]:
        if self.semantic_matrix is None:
            return {}
        return self.semantic_matrix.truth_for_candidate(
            request_type=str(request_type or ""),
            service_type=str(service_id),
            candidate_metadata=candidate_metadata,
            link_input_semantic=link_input_semantic,
        )

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


def _semantic_fidelity(score: float) -> float:
    value = float(score)
    if value < 0.0:
        # Encoder similarities are cosine-like in [-1, 1]; map them into the same
        # [0, 1] fidelity range as learned scorer probabilities without discarding
        # weak negative evidence through hard clipping.
        value = 0.5 * (value + 1.0)
    return max(0.0, min(1.0, value))


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
        "remote_exact",
        "remote_compatible",
        "borderline_weak",
        "hard_negative_mismatch",
        "stale_clone_exact",
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
