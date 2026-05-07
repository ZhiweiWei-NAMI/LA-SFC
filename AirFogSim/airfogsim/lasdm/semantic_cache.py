from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


@dataclass
class CachedEmbedding:
    key: str
    vector: List[float]
    model_name: str
    created_at_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_numpy(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


class EmbeddingCache:
    """Small JSONL embedding cache for service/request descriptions."""

    def __init__(self):
        self._items: Dict[str, CachedEmbedding] = {}

    def get(self, key: str) -> Optional[np.ndarray]:
        item = self._items.get(str(key))
        return None if item is None else item.as_numpy()

    def put(
        self,
        key: str,
        vector: np.ndarray,
        model_name: str,
        created_at_s: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._items[str(key)] = CachedEmbedding(
            key=str(key),
            vector=np.asarray(vector, dtype=np.float32).tolist(),
            model_name=str(model_name),
            created_at_s=float(created_at_s),
            metadata=dict(metadata or {}),
        )

    def keys(self) -> List[str]:
        return sorted(self._items)

    def save_jsonl(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as file:
            for key in self.keys():
                file.write(json.dumps(asdict(self._items[key]), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str) -> "EmbeddingCache":
        cache = cls()
        source = Path(path)
        if not source.exists():
            return cache
        with source.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                raw = json.loads(line)
                cache._items[str(raw["key"])] = CachedEmbedding(**raw)
        return cache


@dataclass
class SemanticAdvertisement:
    """Full semantic service advertisement exchanged among regional agents."""

    source_agent_id: str
    instance_id: str
    service_id: str
    node_id: str
    region_id: str
    node_type: str
    capabilities: Tuple[str, ...]
    input_semantic: str
    output_semantic: str
    semantic_embedding: List[float]
    created_at_s: float
    ttl_s: float
    version: str = "v1"
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload_bytes: int = 0

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - float(self.created_at_s))

    def is_stale(self, now_s: float) -> bool:
        return self.age_s(now_s) > float(self.ttl_s)

    def freshness(self, now_s: float) -> float:
        if self.ttl_s <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.age_s(now_s) / float(self.ttl_s)))

    def key(self) -> Tuple[str, str]:
        return (self.source_agent_id, self.instance_id)

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["capabilities"] = list(self.capabilities)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SemanticAdvertisement":
        data = dict(raw)
        data["capabilities"] = tuple(data.get("capabilities", ()))
        return cls(**data)


class SemanticAdvertisementCache:
    """TTL-aware cache of remote advertisements visible to a single agent."""

    def __init__(self, owner_agent_id: str):
        self.owner_agent_id = str(owner_agent_id)
        self._ads: Dict[Tuple[str, str], SemanticAdvertisement] = {}
        self.received_count = 0
        self.dropped_stale_count = 0

    def upsert(self, ad: SemanticAdvertisement, now_s: float) -> bool:
        if ad.source_agent_id == self.owner_agent_id:
            return False
        if ad.is_stale(now_s):
            self.dropped_stale_count += 1
            return False
        self._ads[ad.key()] = ad
        self.received_count += 1
        return True

    def upsert_many(self, ads: Iterable[SemanticAdvertisement], now_s: float) -> int:
        accepted = 0
        for ad in ads:
            accepted += int(self.upsert(ad, now_s))
        return accepted

    def prune(self, now_s: float) -> int:
        stale_keys = [key for key, ad in self._ads.items() if ad.is_stale(now_s)]
        for key in stale_keys:
            self._ads.pop(key, None)
        self.dropped_stale_count += len(stale_keys)
        return len(stale_keys)

    def fresh(self, now_s: float) -> List[SemanticAdvertisement]:
        self.prune(now_s)
        return list(self._ads.values())

    def all(self) -> List[SemanticAdvertisement]:
        return list(self._ads.values())

    def stale_ratio(self, now_s: float) -> float:
        total = len(self._ads)
        if total == 0:
            return 0.0
        stale = sum(1 for ad in self._ads.values() if ad.is_stale(now_s))
        return stale / float(total)

    def to_trace_rows(self, now_s: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ad in self._ads.values():
            rows.append(
                {
                    "owner_agent_id": self.owner_agent_id,
                    "source_agent_id": ad.source_agent_id,
                    "instance_id": ad.instance_id,
                    "service_id": ad.service_id,
                    "region_id": ad.region_id,
                    "age_s": ad.age_s(now_s),
                    "ttl_s": ad.ttl_s,
                    "freshness": ad.freshness(now_s),
                    "stale": ad.is_stale(now_s),
                    "payload_bytes": ad.payload_bytes,
                }
            )
        return rows
