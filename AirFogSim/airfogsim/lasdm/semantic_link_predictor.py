from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .semantic_encoder import SemanticEncoder


NODE_TYPE_ORDER = ("vehicle", "uav", "rsu", "cloud_server")


class SemanticLinkScorer(nn.Module):
    """End-to-end scalar semantic scorer shared by discovery and policy.

    The SBERT/hash text encoder is frozen. Only the small MLP, service-type
    embeddings, and position embeddings are trainable. Discovery calls
    ``score_for_discovery`` under ``no_grad``; policy code calls
    ``score_with_grad`` so actor losses can update the scorer parameters.
    """

    def __init__(
        self,
        service_type_to_idx: Optional[Mapping[str, int]] = None,
        encoder: Optional[SemanticEncoder] = None,
        num_service_types: int = 16,
        service_embedding_dim: int = 32,
        max_chain_position: int = 8,
        position_embedding_dim: int = 16,
        hidden_dim: int = 256,
        embedding_dim: int = 384,
    ):
        super().__init__()

        class _TorchScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.service_embedding = nn.Embedding(max(1, int(num_service_types)), int(service_embedding_dim))
                self.position_embedding = nn.Embedding(max(1, int(max_chain_position)), int(position_embedding_dim))
                input_dim = int(embedding_dim) * 2 + int(service_embedding_dim) + int(position_embedding_dim) + 8
                self.scorer = nn.Sequential(
                    nn.Linear(input_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), max(8, int(hidden_dim) // 2)),
                    nn.ReLU(),
                    nn.Linear(max(8, int(hidden_dim) // 2), 1),
                    nn.Sigmoid(),
                )

            def forward(
                self,
                request_embedding: Any,
                profile_embedding: Any,
                service_type_idx: Any,
                chain_position: Any,
                meta_features: Any,
            ) -> Any:
                service_idx = service_type_idx.clamp(0, self.service_embedding.num_embeddings - 1).long()
                position_idx = chain_position.clamp(0, self.position_embedding.num_embeddings - 1).long()
                service = self.service_embedding(service_idx).reshape(-1)
                position = self.position_embedding(position_idx).reshape(-1)
                vector = torch.cat(
                    [
                        request_embedding.reshape(-1),
                        profile_embedding.reshape(-1),
                        service,
                        position,
                        meta_features.reshape(-1),
                    ],
                    dim=-1,
                )
                return self.scorer(vector).reshape(())

        self.torch = torch
        self.encoder = encoder or SemanticEncoder(backend="hash", hash_dim=int(embedding_dim))
        self.embedding_dim = int(embedding_dim)
        self.service_type_to_idx = {str(key): int(value) for key, value in dict(service_type_to_idx or {}).items()}
        self.module = _TorchScorer()
        self._text_tensor_cache: Dict[tuple[str, str, str], Any] = {}

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        self.module.load_state_dict(state, strict=strict)

    def to(self, device: Any) -> "SemanticLinkScorer":
        self.module.to(device)
        return self

    def train(self, mode: bool = True) -> "SemanticLinkScorer":
        self.module.train(mode)
        return self

    def eval(self) -> "SemanticLinkScorer":
        self.module.eval()
        return self

    @property
    def device(self) -> Any:
        return next(self.module.parameters()).device

    def service_type_index(self, service_type: Any, explicit_idx: Any = None) -> int:
        if explicit_idx not in (None, ""):
            try:
                return int(explicit_idx)
            except (TypeError, ValueError):
                pass
        return int(self.service_type_to_idx.get(str(service_type), 0))

    def encode_text_tensor(self, text: Any, dtype: Any = None, device: Any = None) -> Any:
        target_device = device if device is not None else self.device
        target_dtype = dtype if dtype is not None else self.torch.float32
        text_key = str(text or "")
        cache_key = (text_key, str(target_device), str(target_dtype))
        cached = self._text_tensor_cache.get(cache_key)
        if cached is not None:
            return cached
        vector = np.asarray(self.encoder.encode(text_key), dtype=np.float32).reshape(-1)
        if vector.size < self.embedding_dim:
            vector = np.pad(vector, (0, self.embedding_dim - vector.size), mode="constant")
        elif vector.size > self.embedding_dim:
            vector = vector[: self.embedding_dim]
        tensor = self.torch.as_tensor(vector, dtype=target_dtype, device=target_device)
        self._text_tensor_cache[cache_key] = tensor
        return tensor

    @property
    def no_grad(self):
        return self.torch.no_grad

    def score_for_discovery(
        self,
        request_context_text: str,
        instance_profile_text: str,
        service_type_idx: int,
        chain_position: int,
        node_type: str,
        is_remote: bool,
        is_same_region: bool,
        staleness_s: float = 0.0,
        topology_risk: float = 0.0,
    ) -> float:
        with self.torch.no_grad():
            value = self.score_with_grad(
                request_context_text=request_context_text,
                instance_profile_text=instance_profile_text,
                service_type_idx=service_type_idx,
                chain_position=chain_position,
                node_type=node_type,
                is_remote=float(bool(is_remote)),
                is_same_region=float(bool(is_same_region)),
                staleness_s=staleness_s,
                topology_risk=topology_risk,
            )
            return float(value.detach().cpu().item())

    def score_with_grad(
        self,
        request_context_text: Optional[str] = None,
        instance_profile_text: Optional[str] = None,
        request_context_emb: Any = None,
        instance_profile_emb: Any = None,
        service_type_idx: int = 0,
        chain_position: int = 0,
        node_type: str = "unknown",
        node_type_onehot: Any = None,
        is_remote: float = 0.0,
        is_same_region: float = 0.0,
        staleness_s: float = 0.0,
        topology_risk: float = 0.0,
        reference: Any = None,
    ) -> Any:
        dtype = getattr(reference, "dtype", self.torch.float32)
        device = getattr(reference, "device", self.device)
        if request_context_emb is None:
            request_context_emb = self.encode_text_tensor(request_context_text or "", dtype=dtype, device=device)
        else:
            request_context_emb = self.torch.as_tensor(request_context_emb, dtype=dtype, device=device)
        if instance_profile_emb is None:
            instance_profile_emb = self.encode_text_tensor(instance_profile_text or "", dtype=dtype, device=device)
        else:
            instance_profile_emb = self.torch.as_tensor(instance_profile_emb, dtype=dtype, device=device)
        if node_type_onehot is None:
            node_type_onehot = self._node_type_onehot(node_type, dtype=dtype, device=device)
        else:
            node_type_onehot = self.torch.as_tensor(node_type_onehot, dtype=dtype, device=device).reshape(-1)
            if node_type_onehot.numel() < 4:
                node_type_onehot = self.torch.cat(
                    [node_type_onehot, self.torch.zeros((4 - node_type_onehot.numel(),), dtype=dtype, device=device)],
                    dim=0,
                )
            node_type_onehot = node_type_onehot[:4]
        meta = self.torch.cat(
            [
                node_type_onehot,
                self.torch.tensor(
                    [
                        float(is_remote),
                        float(is_same_region),
                        min(1.0, max(0.0, float(staleness_s) / 10.0)),
                        min(1.0, max(0.0, float(topology_risk))),
                    ],
                    dtype=dtype,
                    device=device,
                ),
            ],
            dim=0,
        )
        return self.module(
            request_context_emb,
            instance_profile_emb,
            self.torch.tensor(int(service_type_idx), dtype=self.torch.long, device=device),
            self.torch.tensor(int(chain_position), dtype=self.torch.long, device=device),
            meta,
        )

    def score_candidate_tensor(
        self,
        candidate: Mapping[str, Any],
        candidate_set: Mapping[str, Any],
        reference: Any,
    ) -> Any:
        metadata = dict(candidate.get("metadata", {}) or {})
        request_text = str(metadata.get("request_context_text", candidate_set.get("request_context_text", "")) or "")
        profile_text = str(metadata.get("profile_text", "") or "")
        service_idx = self.service_type_index(
            candidate.get("service_id", metadata.get("service_type", "")),
            metadata.get("service_type_idx"),
        )
        same_region = str(candidate.get("region_id", "")) == str(candidate_set.get("agent_id", ""))
        return self.score_with_grad(
            request_context_text=request_text,
            instance_profile_text=profile_text,
            service_type_idx=service_idx,
            chain_position=int(candidate_set.get("sfc_node_index", 0) or 0),
            node_type=str(candidate.get("node_type", "unknown")),
            is_remote=float(bool(candidate.get("is_remote", False))),
            is_same_region=float(same_region),
            staleness_s=float(candidate.get("staleness_s", 0.0) or 0.0),
            topology_risk=float(metadata.get("topology_risk", 0.0) or 0.0),
            reference=reference,
        )

    def _node_type_onehot(self, node_type: str, dtype: Any, device: Any) -> Any:
        values = [1.0 if str(node_type) == item else 0.0 for item in NODE_TYPE_ORDER]
        return self.torch.tensor(values, dtype=dtype, device=device)


def build_semantic_scorer(
    service_type_to_idx: Mapping[str, int],
    encoder: Optional[SemanticEncoder] = None,
    embedding_dim: int = 384,
) -> SemanticLinkScorer:
    return SemanticLinkScorer(
        service_type_to_idx=service_type_to_idx,
        encoder=encoder,
        num_service_types=max(1, len(service_type_to_idx)),
        embedding_dim=int(embedding_dim),
    )
