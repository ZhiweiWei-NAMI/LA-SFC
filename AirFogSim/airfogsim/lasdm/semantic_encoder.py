from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


@dataclass(frozen=True)
class SemanticTextRecord:
    """Text item that should be represented by the semantic encoder."""

    record_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticMatch:
    """Similarity result returned by SemanticEncoder.topk."""

    record_id: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticEncoder:
    """Semantic text encoder with explicit SBERT or deterministic hash backends."""

    _GLOBAL_CACHE: Dict[str, np.ndarray] = {}
    _GLOBAL_MODELS: Dict[Tuple[str, str], Any] = {}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str = "hash",
        normalize: bool = True,
        hash_dim: int = 384,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.normalize = bool(normalize)
        self.hash_dim = int(hash_dim)
        self.batch_size = int(batch_size)
        self.device = device
        self._model: Any = None
        self._backend_active: Optional[str] = None
        self._cache = self._GLOBAL_CACHE

    @property
    def embedding_dim(self) -> int:
        if self._cache:
            return int(next(iter(self._cache.values())).shape[0])
        if self._backend_active == "sbert" and self._model is not None:
            try:
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return self.hash_dim

    @property
    def backend_active(self) -> str:
        if self._backend_active is None:
            self._ensure_backend()
        return str(self._backend_active)

    def encode(self, texts: Union[str, Sequence[str]], use_cache: bool = True) -> np.ndarray:
        """Return an array of L2-normalized embeddings with shape [N, D]."""

        single = isinstance(texts, str)
        items = [str(texts)] if single else [str(item) for item in texts]
        if not items:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        vectors: List[Optional[np.ndarray]] = [None] * len(items)
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        for idx, text in enumerate(items):
            key = self._cache_key(text)
            if use_cache and key in self._cache:
                vectors[idx] = self._cache[key]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            encoded = self._encode_uncached(missing_texts)
            for idx, text, vector in zip(missing_indices, missing_texts, encoded):
                vector = vector.astype(np.float32, copy=False)
                if self.normalize:
                    vector = normalize_vector(vector)
                if use_cache:
                    self._cache[self._cache_key(text)] = vector
                vectors[idx] = vector

        result = np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32, copy=False)
        return result[0] if single else result

    def encode_records(self, records: Sequence[SemanticTextRecord]) -> Dict[str, np.ndarray]:
        texts = [record.text for record in records]
        vectors = self.encode(texts)
        return {record.record_id: vectors[idx] for idx, record in enumerate(records)}

    def similarity(self, query: Union[str, np.ndarray], candidates: Union[Sequence[str], np.ndarray]) -> np.ndarray:
        if isinstance(query, str):
            query_vec = np.asarray(self.encode(query), dtype=np.float32)
        else:
            query_vec = np.asarray(query, dtype=np.float32)
        if isinstance(candidates, np.ndarray):
            cand_vecs = np.asarray(candidates, dtype=np.float32)
        else:
            cand_vecs = np.asarray(self.encode(list(candidates)), dtype=np.float32)
        if cand_vecs.ndim == 1:
            cand_vecs = cand_vecs.reshape(1, -1)
        return cosine_similarity(query_vec.reshape(1, -1), cand_vecs).reshape(-1)

    def topk(
        self,
        query: Union[str, np.ndarray],
        records: Union[Mapping[str, str], Sequence[SemanticTextRecord]],
        top_k: int = 10,
        threshold: float = -1.0,
    ) -> List[SemanticMatch]:
        if isinstance(records, Mapping):
            normalized_records = [SemanticTextRecord(str(key), str(value)) for key, value in records.items()]
        else:
            normalized_records = list(records)
        if not normalized_records:
            return []
        scores = self.similarity(query, [record.text for record in normalized_records])
        order = np.argsort(-scores)
        matches: List[SemanticMatch] = []
        for rank, idx in enumerate(order[: max(0, int(top_k))], start=1):
            score = float(scores[idx])
            if score < threshold:
                continue
            record = normalized_records[int(idx)]
            matches.append(SemanticMatch(record.record_id, score, rank, dict(record.metadata)))
        return matches

    def clear_cache(self) -> None:
        self._cache.clear()

    def _encode_uncached(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_backend()
        if self._backend_active == "sbert":
            embeddings = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        return np.vstack([hash_text_embedding(text, dim=self.hash_dim) for text in texts]).astype(np.float32)

    def _ensure_backend(self) -> None:
        if self._backend_active is not None:
            return
        backend = self.backend.lower()
        if backend not in {"sbert", "hash"}:
            raise ValueError("SemanticEncoder backend must be one of: sbert, hash")
        if backend == "sbert":
            try:
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("USE_FLAX", "0")
                os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                from sentence_transformers import SentenceTransformer  # type: ignore

                kwargs = {}
                if self.device:
                    kwargs["device"] = self.device
                model_key = (self.model_name, str(self.device or ""))
                if model_key not in self._GLOBAL_MODELS:
                    self._GLOBAL_MODELS[model_key] = SentenceTransformer(self.model_name, **kwargs)
                self._model = self._GLOBAL_MODELS[model_key]
                self._backend_active = "sbert"
                return
            except Exception as exc:
                raise RuntimeError(
                    "sentence-transformers is required for backend='sbert'. "
                    "Install it or explicitly configure backend='hash'."
                ) from exc
        self._backend_active = "hash"

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        return f"{self.model_name}:{self.backend}:{digest}"


def service_instance_text(instance: Any) -> str:
    metadata = dict(getattr(instance, "metadata", {}) or {})
    parts = [
        f"service {getattr(instance, 'service_id', '')}",
        f"capabilities {' '.join(getattr(instance, 'capabilities', []) or [])}",
        f"input {getattr(instance, 'input_semantic', '')}",
        f"output {getattr(instance, 'output_semantic', '')}",
        f"node_type {getattr(instance, 'node_type', '')}",
        f"region {getattr(instance, 'region_id', '')}",
        str(metadata.get("description", "")),
        str(metadata.get("semantic_description", "")),
    ]
    return " ; ".join(part for part in parts if part and part.strip())


def sfc_node_text(sfc_node: Any, payload_semantic: str = "") -> str:
    metadata = dict(getattr(sfc_node, "metadata", {}) or {})
    parts = [
        f"request service {getattr(sfc_node, 'service_type', '')}",
        f"required capabilities {' '.join(getattr(sfc_node, 'required_capabilities', []) or [])}",
        f"input {getattr(sfc_node, 'input_semantic', payload_semantic)}",
        f"output {getattr(sfc_node, 'output_semantic', '')}",
        str(metadata.get("description", "")),
        str(metadata.get("semantic_description", "")),
    ]
    return " ; ".join(part for part in parts if part and part.strip())


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < eps:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True).T
    denom = np.maximum(eps, a_norm @ b_norm)
    return (a @ b.T) / denom


def hash_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic lexical embedding for explicitly configured hash backend."""

    dim = int(dim)
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _TOKEN_RE.findall(text.lower()) or [text.lower()]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=16).digest()
        for offset in range(0, len(digest), 4):
            raw = int.from_bytes(digest[offset : offset + 4], "little", signed=False)
            idx = raw % dim
            sign = 1.0 if (raw >> 9) & 1 else -1.0
            weight = 1.0 + ((raw >> 16) & 0xFF) / 255.0
            vec[idx] += sign * weight
    return normalize_vector(vec)
