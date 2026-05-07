from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .instance_directory import ServiceInstanceDirectory
from .semantic_cache import SemanticAdvertisement
from .semantic_encoder import SemanticEncoder, service_instance_text


@dataclass(frozen=True)
class SemanticExchangeConfig:
    ttl_s: float = 5.0
    radius_hops: int = 1
    fixed_delay_s: float = 0.1
    per_hop_delay_s: float = 0.02
    top_k_per_agent: int = 32
    embedding_dim: int = 384
    include_metadata: bool = True
    min_advertised_health: float = 0.0


@dataclass
class SemanticMessage:
    sender_agent_id: str
    receiver_agent_id: str
    advertisements: List[SemanticAdvertisement]
    send_time_s: float
    deliver_time_s: float
    payload_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_trace_row(self) -> Dict[str, Any]:
        return {
            "sender_agent_id": self.sender_agent_id,
            "receiver_agent_id": self.receiver_agent_id,
            "send_time_s": self.send_time_s,
            "deliver_time_s": self.deliver_time_s,
            "delay_s": max(0.0, self.deliver_time_s - self.send_time_s),
            "advertisement_count": len(self.advertisements),
            "payload_bytes": self.payload_bytes,
            **self.metadata,
        }


class SemanticExchange:
    """Delay-aware message bus for local semantic advertisement exchange."""

    def __init__(self, config: Optional[SemanticExchangeConfig] = None):
        self.config = config or SemanticExchangeConfig()
        self.pending: List[SemanticMessage] = []
        self.delivered_messages: List[SemanticMessage] = []
        self.trace_rows: List[Dict[str, Any]] = []
        self.total_payload_bytes = 0
        self.total_message_count = 0

    def build_advertisements(
        self,
        agent_id: str,
        directory: ServiceInstanceDirectory,
        encoder: SemanticEncoder,
        now_s: float,
        top_k: Optional[int] = None,
    ) -> List[SemanticAdvertisement]:
        limit = int(top_k if top_k is not None else self.config.top_k_per_agent)
        instances = [instance for instance in directory.all() if instance.health_score >= self.config.min_advertised_health]
        instances.sort(key=lambda item: (item.region_id, item.service_id, item.instance_id))
        ads: List[SemanticAdvertisement] = []
        for instance in instances[: max(0, limit)]:
            text = service_instance_text(instance)
            vector = np.asarray(encoder.encode(text), dtype=np.float32).reshape(-1)
            if vector.size != int(self.config.embedding_dim):
                raise ValueError(f"semantic embedding dimension mismatch: expected {int(self.config.embedding_dim)}, got {vector.size}")
            metadata = dict(instance.metadata or {}) if self.config.include_metadata else {}
            if self.config.include_metadata:
                metadata.setdefault("cold_start_s", float(instance.cold_start_s))
                metadata.setdefault("capacity_cpu", float(instance.capacity.get("cpu", 0.0) or 0.0))
                metadata.setdefault("available_cpu", float(instance.available("cpu")))
                metadata.setdefault("max_concurrency", int(instance.max_concurrency))
                metadata.setdefault("current_load", int(instance.current_load))
            ad = SemanticAdvertisement(
                source_agent_id=str(agent_id),
                instance_id=instance.instance_id,
                service_id=instance.service_id,
                node_id=instance.node_id,
                region_id=instance.region_id,
                node_type=instance.node_type,
                capabilities=tuple(instance.capabilities),
                input_semantic=instance.input_semantic,
                output_semantic=instance.output_semantic,
                semantic_embedding=[float(item) for item in vector.tolist()],
                created_at_s=float(now_s),
                ttl_s=float(self.config.ttl_s),
                version=instance.version,
                metadata=metadata,
            )
            ad.payload_bytes = self.estimate_advertisement_bytes(ad)
            ads.append(ad)
        return ads

    def broadcast(
        self,
        sender_agent_id: str,
        receiver_agent_ids: Iterable[str],
        advertisements: Sequence[SemanticAdvertisement],
        now_s: float,
        hop_count: int = 1,
    ) -> List[SemanticMessage]:
        created: List[SemanticMessage] = []
        for receiver in receiver_agent_ids:
            receiver = str(receiver)
            if receiver == sender_agent_id:
                continue
            payload_bytes = sum(ad.payload_bytes or self.estimate_advertisement_bytes(ad) for ad in advertisements)
            delay = float(self.config.fixed_delay_s) + float(self.config.per_hop_delay_s) * max(0, int(hop_count) - 1)
            message = SemanticMessage(
                sender_agent_id=str(sender_agent_id),
                receiver_agent_id=receiver,
                advertisements=list(advertisements),
                send_time_s=float(now_s),
                deliver_time_s=float(now_s) + delay,
                payload_bytes=int(payload_bytes),
                metadata={
                    "hop_count": int(hop_count),
                    "exchange_ttl_s": float(self.config.ttl_s),
                    "exchange_radius_hops": int(self.config.radius_hops),
                    "exchange_top_k": int(self.config.top_k_per_agent),
                    "semantic_embedding_dim": int(self.config.embedding_dim),
                },
            )
            self.pending.append(message)
            self.trace_rows.append(message.to_trace_row())
            self.total_payload_bytes += int(payload_bytes)
            self.total_message_count += 1
            created.append(message)
        return created

    def deliver(self, now_s: float) -> Dict[str, List[SemanticAdvertisement]]:
        ready: Dict[str, List[SemanticAdvertisement]] = {}
        still_pending: List[SemanticMessage] = []
        for message in self.pending:
            if message.deliver_time_s <= now_s + 1e-9:
                ready.setdefault(message.receiver_agent_id, []).extend(message.advertisements)
                self.delivered_messages.append(message)
            else:
                still_pending.append(message)
        self.pending = still_pending
        return ready

    def step_exchange(
        self,
        agent_directories: Mapping[str, ServiceInstanceDirectory],
        encoder: SemanticEncoder,
        neighbor_map: Mapping[str, Any],
        now_s: float,
    ) -> Dict[str, List[SemanticAdvertisement]]:
        for agent_id, directory in agent_directories.items():
            ads = self.build_advertisements(agent_id, directory, encoder, now_s)
            neighbors = neighbor_map.get(agent_id, [])
            if isinstance(neighbors, Mapping):
                neighbor_items = sorted((str(receiver), int(hop_count or 1)) for receiver, hop_count in neighbors.items())
                if neighbor_items and all(hop <= 1 for _receiver, hop in neighbor_items):
                    neighbor_items = neighbor_items[: max(1, int(self.config.radius_hops or 1))]
                for receiver, hop_count in neighbor_items:
                    self.broadcast(agent_id, [receiver], ads, now_s, hop_count=int(hop_count or 1))
            else:
                # Some runtime snapshots expose an unordered control-plane neighbor
                # list rather than explicit hop counts. In that case the configured
                # exchange radius is modeled as a bounded fanout so the TTL/radius
                # sweep changes actual discovery coverage instead of only annotating
                # trace rows.
                max_receivers = max(1, int(self.config.radius_hops or 1))
                selected_neighbors = sorted(str(item) for item in neighbors)[:max_receivers]
                self.broadcast(agent_id, selected_neighbors, ads, now_s)
        return self.deliver(now_s)

    def estimate_advertisement_bytes(self, ad: SemanticAdvertisement) -> int:
        # JSON length is a conservative, reproducible proxy for message overhead.
        raw = ad.to_dict()
        return len(json.dumps(raw, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def overhead_summary(self) -> Dict[str, float]:
        return {
            "message_count": float(self.total_message_count),
            "payload_bytes": float(self.total_payload_bytes),
            "pending_message_count": float(len(self.pending)),
            "delivered_message_count": float(len(self.delivered_messages)),
        }
