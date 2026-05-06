from __future__ import annotations

import csv
import contextlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


DEFAULT_SEMANTIC_OUTPUT_ROOT = Path("experiment_artifacts/raw_data/semantic_profiles_artifacts")

VARIANT_TYPE_SCORE = {
    "exact": 1.00,
    "equivalent": 0.86,
    "compatible": 0.68,
    "weak": 0.36,
    "mismatch": 0.04,
}

TRUTH_RELATION_THRESHOLDS = (
    (0.88, "exact"),
    (0.76, "equivalent"),
    (0.58, "compatible"),
    (0.28, "weak"),
    (0.00, "mismatch"),
)

SERVICE_IO_DEFAULTS = {
    "aerial_video_preprocessing": ("video", "keyframes"),
    "thermal_frame_preprocessing": ("thermal_frame", "calibrated_thermal_frame"),
    "multi_spectral_fusion_preprocessing": ("multi_sensor_frames", "fused_scene"),
    "fire_candidate_detection": ("keyframes", "fire_candidates"),
    "vehicle_detection_and_tracking": ("keyframes", "vehicle_tracks"),
    "person_detection_and_reid": ("keyframes", "person_tracklets"),
    "industrial_defect_detection": ("fused_scene", "defect_candidates"),
    "smoke_fire_fusion": ("fire_candidates", "fused_fire_evidence"),
    "multi_sensor_event_fusion": ("fused_scene", "event_evidence"),
    "trajectory_risk_assessment": ("vehicle_tracks", "risk_assessment"),
    "fire_event_verification": ("event_evidence", "verified_event"),
    "identity_verification": ("person_tracklets", "verified_identity"),
    "defect_classification": ("defect_candidates", "classified_defects"),
    "emergency_alert_publish": ("verified_event", "alert_record"),
    "surveillance_event_archive": ("verified_event", "archived_event"),
    "inspection_report_generation": ("classified_defects", "inspection_report"),
}


@dataclass(frozen=True)
class SemanticImplementation:
    raw: Dict[str, Any]

    @property
    def implementation_id(self) -> str:
        return str(self.raw["implementation_id"])

    @property
    def service_type(self) -> str:
        return str(self.raw["service_type"])

    @property
    def variant_type(self) -> str:
        return str(self.raw.get("semantic_variant_type", "compatible"))

    @property
    def profile_text(self) -> str:
        explicit = str(self.raw.get("profile_text", "") or "").strip()
        if explicit:
            return explicit
        profile = dict(self.raw.get("semantic_profile", {}) or {})
        domain = dict(self.raw.get("domain", {}) or {})
        performance = dict(self.raw.get("performance_profile", {}) or {})
        capabilities = [str(item).replace("_", " ") for item in self.raw.get("capabilities", []) or []]
        primary_domains = [str(item).replace("_", " ") for item in domain.get("primary_domains", []) or []]
        secondary_domains = [str(item).replace("_", " ") for item in domain.get("secondary_domains", []) or []]
        fragments = [
            str(profile.get("description", "")),
            _semantic_description(profile.get("input_semantic", "")),
            _semantic_description(profile.get("output_semantic", "")),
            str(domain.get("scene_limitations", "")),
            str(performance.get("robustness_notes", "")),
            (
                f"It is cataloged as a {self.variant_type} implementation for {self.service_type.replace('_', ' ')}. "
                f"The primary deployment semantics cover {', '.join(primary_domains)}, with secondary coverage for "
                f"{', '.join(secondary_domains) if secondary_domains else 'nearby operational cases'}. "
                f"Its declared capabilities are {', '.join(capabilities) if capabilities else 'service-specific inference'}, "
                "so the scheduler can compare it against alternatives that may share an API but differ in domain fit, "
                "sensor modality, and downstream meaning."
            ),
        ]
        return " ".join(fragment.strip() for fragment in fragments if fragment and fragment.strip())

    @property
    def input_semantic_label(self) -> str:
        default, _output_default = SERVICE_IO_DEFAULTS.get(self.service_type, ("any", "any"))
        profile = dict(self.raw.get("semantic_profile", {}) or {})
        input_semantic = profile.get("input_semantic", {})
        if isinstance(input_semantic, Mapping):
            return str(input_semantic.get("format") or input_semantic.get("modality") or self.raw.get("input_semantic") or default)
        return str(self.raw.get("input_semantic") or default)

    @property
    def output_semantic_label(self) -> str:
        _input_default, default = SERVICE_IO_DEFAULTS.get(self.service_type, ("any", "any"))
        profile = dict(self.raw.get("semantic_profile", {}) or {})
        raw_output = profile.get("output_semantic", {})
        if not isinstance(raw_output, Mapping):
            return str(self.raw.get("output_semantic") or default)
        output_semantic = dict(raw_output or {})
        label_space = output_semantic.get("label_space")
        if isinstance(label_space, Sequence) and not isinstance(label_space, str) and label_space:
            return "_".join(str(item) for item in label_space[:3])
        return str(output_semantic.get("format") or self.raw.get("output_semantic") or default)

    @property
    def domain_tags(self) -> Tuple[str, ...]:
        domain = dict(self.raw.get("domain", {}) or {})
        tags: List[str] = []
        for key in ("primary_domains", "secondary_domains"):
            values = domain.get(key, ())
            if isinstance(values, str):
                values = [values]
            tags.extend(str(item) for item in values or ())
        return tuple(dict.fromkeys(tag for tag in tags if tag))

    @property
    def compatible_node_types(self) -> Tuple[str, ...]:
        values = self.raw.get("compatible_node_types", ())
        if isinstance(values, str):
            values = [values]
        return tuple(str(item) for item in values or ())


class SemanticLinkMatrix:
    """Semantic truth matrix built from reviewed YAML profiles.

    The matrix is an environment artifact. It computes audit-only truth fields
    from request type, implementation profile, and link input semantics. Runtime
    ranking uses the learned scalar scorer, not these truth fields.
    """

    def __init__(self, config_root: str | Path):
        self.config_root = Path(config_root)
        if not self.config_root.exists():
            raise FileNotFoundError(f"Semantic config root does not exist: {self.config_root}")
        self.request_types = self._load_named_items(self.config_root / "request_types.yaml", "request_types")
        self.service_types = self._load_named_items(self.config_root / "service_types.yaml", "service_types")
        self.implementations = self._load_implementations(self.config_root / "implementations")
        if not self.implementations:
            raise ValueError(f"No semantic implementations found under {self.config_root / 'implementations'}")
        self._by_service: Dict[str, List[SemanticImplementation]] = defaultdict(list)
        self._by_id: Dict[str, SemanticImplementation] = {}
        for implementation in self.implementations:
            self._by_service[implementation.service_type].append(implementation)
            if implementation.implementation_id in self._by_id:
                raise ValueError(f"Duplicate implementation_id: {implementation.implementation_id}")
            self._by_id[implementation.implementation_id] = implementation
        self._service_type_to_idx = {
            service_type: index for index, service_type in enumerate(sorted(self._by_service))
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], method_root: str | Path, workspace_root: str | Path) -> "SemanticLinkMatrix":
        raw = dict(config.get("semantic_profiles", {}) or {})
        configured_root = raw.get("config_root")
        if configured_root:
            root = Path(str(configured_root))
            if not root.is_absolute():
                root = Path(workspace_root) / root
        else:
            root = Path(method_root) / "configs" / "semantic"
        return cls(root)

    @property
    def service_type_to_idx(self) -> Dict[str, int]:
        return dict(self._service_type_to_idx)

    def service_type_index(self, service_type: str) -> int:
        return int(self._service_type_to_idx.get(str(service_type), 0))

    def list_service_types(self) -> List[str]:
        return sorted(self._by_service)

    def list_implementations_for(
        self,
        service_type: str,
        compatible_node_type: Optional[str] = None,
        include_mismatch: bool = True,
    ) -> List[SemanticImplementation]:
        items = list(self._by_service.get(str(service_type), []))
        if not include_mismatch:
            items = [item for item in items if item.variant_type != "mismatch"]
        if compatible_node_type:
            node_type = str(compatible_node_type)
            scoped = [item for item in items if not item.compatible_node_types or node_type in item.compatible_node_types]
            items = scoped
        return items

    def implementation(self, implementation_id: str) -> SemanticImplementation:
        return self._by_id[str(implementation_id)]

    def implementation_or_none(self, implementation_id: Any) -> Optional[SemanticImplementation]:
        return self._by_id.get(str(implementation_id))

    def request_type_for_context(self, context: Mapping[str, Any], default: str = "forest_fire_monitoring") -> str:
        request_type = str(context.get("request_type", "") or "")
        if request_type in self.request_types:
            return request_type
        task_class = str(context.get("task_class", "") or "")
        if task_class in self.request_types:
            return task_class
        return default if default in self.request_types else sorted(self.request_types)[0]

    def truth_for_candidate(
        self,
        request_type: str,
        service_type: str,
        candidate_metadata: Mapping[str, Any],
        link_input_semantic: Optional[str] = None,
    ) -> Dict[str, Any]:
        implementation = self.implementation_or_none(candidate_metadata.get("implementation_id"))
        if implementation is None:
            implementation = self._best_profile_for_unknown_candidate(service_type, candidate_metadata)
        s_type = self._type_match_score(request_type, implementation)
        s_io = self._io_score(link_input_semantic, implementation.input_semantic_label)
        s_domain = self._domain_score(request_type, implementation)
        score = max(0.0, min(1.0, 0.40 * s_type + 0.35 * s_io + 0.25 * s_domain))
        relation = self._relation(score)
        return {
            "semantic_link_truth_score": score,
            "semantic_link_truth_relation": relation,
            "semantic_link_truth_cell": f"{_norm(link_input_semantic or 'any')}->{_norm(implementation.input_semantic_label)}:{relation}",
            "semantic_type_score": s_type,
            "semantic_io_score": s_io,
            "semantic_domain_score": s_domain,
        }

    def candidate_metadata_from_implementation(self, implementation: SemanticImplementation) -> Dict[str, Any]:
        raw = dict(implementation.raw)
        profile = dict(raw.get("semantic_profile", {}) or {})
        input_semantic = dict(profile.get("input_semantic", {}) or {}) if isinstance(profile.get("input_semantic", {}), Mapping) else {}
        output_semantic = dict(profile.get("output_semantic", {}) or {}) if isinstance(profile.get("output_semantic", {}), Mapping) else {}
        domain = dict(raw.get("domain", {}) or {})
        label_space = output_semantic.get("label_space", ())
        if isinstance(label_space, str):
            label_space = [label_space]
        return {
            "implementation_id": implementation.implementation_id,
            "service_type_idx": self.service_type_index(implementation.service_type),
            "model_family": str(raw.get("model_family", "")),
            "model_version": str(raw.get("model_version", "")),
            "modality": str(input_semantic.get("modality", _modality_from_domains(implementation.domain_tags))),
            "domain_tags": list(implementation.domain_tags),
            "label_space": [str(item) for item in label_space or ()],
            "profile_text": implementation.profile_text,
            "input_semantic_profile": _semantic_description(profile.get("input_semantic", "")),
            "output_semantic_profile": _semantic_description(profile.get("output_semantic", "")),
            "semantic_variant_type": implementation.variant_type,
            "semantic_group": implementation.variant_type,
            "implementation_profile": raw,
            "primary_domains": list(domain.get("primary_domains", []) or []),
            "secondary_domains": list(domain.get("secondary_domains", []) or []),
        }

    def export_artifacts(
        self,
        output_root: str | Path,
        materialized_instances: Optional[Iterable[Mapping[str, Any]]] = None,
        encoder_config: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        with _file_lock(root / ".semantic_artifacts.lock"):
            config_target = root / "configs" / "semantic"
            if config_target.exists():
                shutil.rmtree(config_target)
            shutil.copytree(self.config_root, config_target)
            self._write_truth_csv(root / "semantic_link_truth.csv")
            self._write_io_csv(root / "io_compatibility_matrix.csv")
            self._write_profile_pool(root / "semantic_profile_pool.jsonl")
            if encoder_config is not None:
                self._write_embedding_artifacts(root, encoder_config)
            if materialized_instances is not None:
                self._write_materialized_profiles(root / "materialized_instance_profiles.jsonl", materialized_instances)
            audit = self.audit(materialized_instances=materialized_instances)
            (root / "semantic_dataset_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
            return audit

    def audit(self, materialized_instances: Optional[Iterable[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        per_service = {key: len(value) for key, value in sorted(self._by_service.items())}
        variant_counts = Counter(item.variant_type for item in self.implementations)
        model_families = sorted({str(item.raw.get("model_family", "")) for item in self.implementations if item.raw.get("model_family")})
        modalities = sorted(
            {
                _modality_from_domains(item.domain_tags)
                for item in self.implementations
                if _modality_from_domains(item.domain_tags)
            }
        )
        profile_texts = [item.profile_text for item in self.implementations]
        duplicate_profiles = len(profile_texts) - len(set(profile_texts))
        word_counts = [_word_count(text) for text in profile_texts]
        materialized = list(materialized_instances or [])
        materialized_impl_counts = Counter(
            str(dict(item.get("metadata", {}) or {}).get("implementation_id", "")) for item in materialized
        )
        materialized_group_counts = Counter(
            str(dict(item.get("metadata", {}) or {}).get("semantic_group", "") or "")
            for item in materialized
        )
        remote_positive_groups = {"remote_exact", "remote_compatible", "stale_clone_exact"}
        local_positive_count = 0
        remote_positive_count = 0
        for item in materialized:
            metadata = dict(item.get("metadata", {}) or {})
            variant = str(metadata.get("semantic_variant_type", metadata.get("semantic_group", "")) or "")
            if variant not in {"exact", "equivalent", "compatible"}:
                continue
            group = str(metadata.get("semantic_group", "") or "")
            if group in remote_positive_groups:
                remote_positive_count += 1
            else:
                local_positive_count += 1
        return {
            "schema_version": "v21",
            "service_type_count": len(self._by_service),
            "implementation_count": len(self.implementations),
            "implementations_per_service_type": per_service,
            "variant_counts": dict(sorted(variant_counts.items())),
            "model_family_count": len(model_families),
            "model_families": model_families,
            "modalities": modalities,
            "duplicate_profile_count": duplicate_profiles,
            "profile_word_count_min": min(word_counts) if word_counts else 0,
            "profile_word_count_max": max(word_counts) if word_counts else 0,
            "profile_word_count_mean": sum(word_counts) / len(word_counts) if word_counts else 0.0,
            "positive_pair_count": int(sum(count for key, count in variant_counts.items() if key in {"exact", "equivalent", "compatible"})),
            "negative_pair_count": int(sum(count for key, count in variant_counts.items() if key in {"weak", "mismatch"})),
            "semantic_group_coverage": dict(sorted(variant_counts.items())),
            "materialized_semantic_group_coverage": dict(sorted((key, value) for key, value in materialized_group_counts.items() if key)),
            "local_positive_count": int(local_positive_count),
            "remote_positive_count": int(remote_positive_count),
            "local_positive_to_remote_positive_ratio": float(local_positive_count) / max(1.0, float(remote_positive_count)),
            "lexical_shortcut_ceiling": _lexical_shortcut_ceiling(self.implementations),
            "materialized_instance_count": len(materialized),
            "materialized_implementation_count": len([key for key in materialized_impl_counts if key]),
            "quality_gates": {
                "service_type_count_ge_12": len(self._by_service) >= 12,
                "implementation_per_service_type_ge_5": all(count >= 5 for count in per_service.values()),
                "total_implementations_ge_80": len(self.implementations) >= 80,
                "model_family_diversity_ge_10": len(model_families) >= 10,
                "modality_coverage_ge_4": len(modalities) >= 4,
                "no_duplicate_profiles": duplicate_profiles == 0,
                "profile_text_min_100_words": bool(word_counts) and min(word_counts) >= 100,
                "profile_text_max_300_words": bool(word_counts) and max(word_counts) <= 300,
            },
        }

    def _type_match_score(self, request_type: str, implementation: SemanticImplementation) -> float:
        request = dict(self.request_types.get(str(request_type), {}) or {})
        service_sequence = {
            str(item)
            for item in request.get("service_sequence", request.get("representative_service_chain", [])) or ()
        }
        if service_sequence and implementation.service_type not in service_sequence:
            return min(0.20, VARIANT_TYPE_SCORE.get(implementation.variant_type, 0.50))
        score = VARIANT_TYPE_SCORE.get(implementation.variant_type, 0.50)
        preferred = request.get("preferred_implementations", {})
        if isinstance(preferred, Mapping) and implementation.implementation_id in preferred:
            score = float(preferred[implementation.implementation_id])
        return max(0.0, min(1.0, score))

    def _domain_score(self, request_type: str, implementation: SemanticImplementation) -> float:
        request = dict(self.request_types.get(str(request_type), {}) or {})
        request_domains = _tokens(request.get("domain_tags", request.get("primary_domains", ())))
        impl_domains = set(implementation.domain_tags)
        if not request_domains or not impl_domains:
            return 0.50
        overlap = request_domains & impl_domains
        union = request_domains | impl_domains
        score = len(overlap) / max(1, len(union))
        domain = dict(implementation.raw.get("domain", {}) or {})
        unsuitable = _tokens(domain.get("unsuitable_domains", ()))
        if request_domains & unsuitable:
            score *= 0.35
        return max(0.0, min(1.0, score))

    def _io_score(self, source_semantic: Optional[str], target_semantic: str) -> float:
        source = _norm(source_semantic or "any")
        target = _norm(target_semantic or "any")
        if source == "any" or target == "any":
            return 0.70
        if source == target:
            return 1.00
        source_tokens = set(source.split("_"))
        target_tokens = set(target.split("_"))
        if source_tokens & target_tokens:
            return 0.72
        compatibility = {
            ("video", "rgb"): 0.92,
            ("keyframes", "rgb"): 0.95,
            ("keyframes", "multispectral"): 0.82,
            ("thermal_frame", "thermal"): 0.96,
            ("thermal_frame", "calibrated_thermal_frame"): 0.96,
            ("calibrated_thermal_frame", "multi_sensor_frames"): 0.84,
            ("keyframes", "multi_sensor_frames"): 0.78,
            ("multi_sensor_frames", "fused_scene"): 0.94,
            ("fused_scene", "fire_candidates"): 0.76,
            ("fused_scene", "defect_candidates"): 0.92,
            ("fire_candidates", "fused_fire_evidence"): 0.92,
            ("fused_fire_evidence", "event_evidence"): 0.88,
            ("event_evidence", "verified_event"): 0.94,
            ("keyframes", "vehicle_tracks"): 0.90,
            ("vehicle_tracks", "risk_assessment"): 0.92,
            ("keyframes", "person_tracklets"): 0.90,
            ("person_tracklets", "verified_identity"): 0.92,
            ("defect_candidates", "classified_defects"): 0.94,
            ("classified_defects", "inspection_report"): 0.95,
            ("verified_event", "alert_record"): 0.93,
            ("verified_event", "archived_event"): 0.90,
            ("detection_result", "detections"): 0.92,
            ("detection_result", "event_candidates"): 0.86,
            ("verified_event", "event_record"): 0.90,
            ("verified_event", "structured_event"): 0.88,
        }
        return compatibility.get((source, target), compatibility.get((target, source), 0.42))

    def _best_profile_for_unknown_candidate(
        self,
        service_type: str,
        metadata: Mapping[str, Any],
    ) -> SemanticImplementation:
        candidates = self.list_implementations_for(service_type)
        if not candidates:
            return self.implementations[0]
        variant = str(metadata.get("semantic_variant_type", "compatible"))
        for item in candidates:
            if item.variant_type == variant:
                return item
        return candidates[0]

    def _relation(self, score: float) -> str:
        value = max(0.0, min(1.0, float(score)))
        for threshold, relation in TRUTH_RELATION_THRESHOLDS:
            if value >= threshold:
                return relation
        return "mismatch"

    def _write_truth_csv(self, path: Path) -> None:
        fieldnames = [
            "request_type",
            "service_type",
            "implementation_id",
            "semantic_variant_type",
            "semantic_link_truth_score",
            "semantic_link_truth_relation",
            "semantic_type_score",
            "semantic_domain_score",
        ]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for request_type in sorted(self.request_types):
                for implementation in sorted(self.implementations, key=lambda item: item.implementation_id):
                    truth = self.truth_for_candidate(
                        request_type,
                        implementation.service_type,
                        {"implementation_id": implementation.implementation_id},
                    )
                    writer.writerow(
                        {
                            "request_type": request_type,
                            "service_type": implementation.service_type,
                            "implementation_id": implementation.implementation_id,
                            "semantic_variant_type": implementation.variant_type,
                            "semantic_link_truth_score": truth["semantic_link_truth_score"],
                            "semantic_link_truth_relation": truth["semantic_link_truth_relation"],
                            "semantic_type_score": truth["semantic_type_score"],
                            "semantic_domain_score": truth["semantic_domain_score"],
                        }
                    )

    def _write_io_csv(self, path: Path) -> None:
        labels = sorted({item.input_semantic_label for item in self.implementations} | {item.output_semantic_label for item in self.implementations})
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["prev_output", "this_input", "compatibility"])
            writer.writeheader()
            for source in labels:
                for target in labels:
                    writer.writerow({"prev_output": source, "this_input": target, "compatibility": self._io_score(source, target)})

    def _write_profile_pool(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as file:
            for item in sorted(self.implementations, key=lambda impl: impl.implementation_id):
                payload = {
                    "implementation_id": item.implementation_id,
                    "service_type": item.service_type,
                    "semantic_variant_type": item.variant_type,
                    "profile_text": item.profile_text,
                    "metadata": self.candidate_metadata_from_implementation(item),
                }
                file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_embedding_artifacts(self, root: Path, encoder_config: Mapping[str, Any]) -> None:
        from .semantic_encoder import SemanticEncoder

        records = sorted(self.implementations, key=lambda impl: impl.implementation_id)
        encoder = SemanticEncoder(
            model_name=str(encoder_config.get("sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            backend=str(encoder_config.get("encoder_backend", encoder_config.get("backend", "sbert"))),
            hash_dim=int(encoder_config.get("hash_dim", 384)),
            batch_size=int(encoder_config.get("encoder_batch_size", 64)),
            device=str(encoder_config.get("encoder_device", "") or "") or None,
        )
        embeddings = np.asarray(encoder.encode([item.profile_text for item in records]), dtype=np.float32)
        report = _embedding_distance_report(records, embeddings, encoder.backend_active, encoder.model_name)
        (root / "semantic_embedding_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        import torch

        torch.save(
            {
                "implementation_ids": [item.implementation_id for item in records],
                "service_types": [item.service_type for item in records],
                "semantic_variant_types": [item.variant_type for item in records],
                "embeddings": torch.as_tensor(embeddings, dtype=torch.float32),
                "backend": encoder.backend_active,
                "model_name": encoder.model_name,
            },
            root / "semantic_embedding_cache.pt",
        )

    def _write_materialized_profiles(self, path: Path, instances: Iterable[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for instance in instances:
                metadata = dict(instance.get("metadata", {}) or {})
                payload = {
                    "instance_id": str(instance.get("instance_id", "")),
                    "service_id": str(instance.get("service_id", "")),
                    "node_id": str(instance.get("node_id", "")),
                    "node_type": str(instance.get("node_type", "")),
                    "region_id": str(instance.get("region_id", "")),
                    "implementation_id": str(metadata.get("implementation_id", "")),
                    "semantic_variant_type": str(metadata.get("semantic_variant_type", "")),
                    "profile_text": str(metadata.get("profile_text", "")),
                }
                file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _load_named_items(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        items = raw.get(key, raw)
        if isinstance(items, Mapping):
            iterable = [dict(value, **{"name": name}) if isinstance(value, Mapping) else {"name": name} for name, value in items.items()]
        else:
            iterable = list(items or [])
        result: Dict[str, Dict[str, Any]] = {}
        for item in iterable:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("request_type") or item.get("service_type") or item.get("name") or "")
            if name:
                result[name] = dict(item)
        return result

    @staticmethod
    def _load_implementations(path: Path) -> List[SemanticImplementation]:
        if not path.exists():
            return []
        records: List[SemanticImplementation] = []
        for file_path in sorted(path.rglob("*.yaml")):
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
            items = raw.get("implementations") if isinstance(raw, Mapping) else None
            if items is None and isinstance(raw, Mapping) and raw.get("implementation_id"):
                items = [raw]
            for item in items or []:
                if isinstance(item, Mapping) and item.get("implementation_id") and item.get("service_type"):
                    records.append(SemanticImplementation(dict(item)))
        return records


def _tokens(values: Any) -> set[str]:
    if isinstance(values, str):
        values = [values]
    return {_norm(item) for item in values or () if _norm(item)}


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    return "_".join(part for part in "".join(chars).split("_") if part) or "any"


def _word_count(text: str) -> int:
    return len([part for part in str(text).replace("/", " ").replace("-", " ").split() if part.strip()])


def _semantic_description(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("description") or value.get("text") or value.get("format") or value.get("modality") or "")
    return str(value or "")


def _modality_from_domains(tags: Sequence[str]) -> str:
    normalized = {_norm(tag) for tag in tags}
    for modality in ("rgb", "thermal", "multispectral", "infrared", "acoustic"):
        if modality in normalized:
            return modality
    if "aerial" in normalized or "traffic" in normalized or "security" in normalized:
        return "rgb"
    return "structured"


def _embedding_distance_report(
    records: Sequence[SemanticImplementation],
    embeddings: np.ndarray,
    backend: str,
    model_name: str,
) -> Dict[str, Any]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("Embedding cache shape does not match semantic implementation records")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    by_variant: Dict[str, List[float]] = defaultdict(list)
    by_pair: Dict[str, List[float]] = defaultdict(list)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            distance = float(1.0 - float(np.dot(normalized[i], normalized[j])))
            left = records[i].variant_type
            right = records[j].variant_type
            if left == right:
                by_variant[left].append(distance)
            pair_key = "__".join(sorted((left, right)))
            by_pair[pair_key].append(distance)
    return {
        "backend": str(backend),
        "model_name": str(model_name),
        "embedding_count": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "distance_metric": "cosine_distance",
        "within_variant_distance": {key: _quantiles(values) for key, values in sorted(by_variant.items())},
        "variant_pair_distance": {key: _quantiles(values) for key, values in sorted(by_pair.items())},
    }


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def _lexical_shortcut_ceiling(records: Sequence[SemanticImplementation]) -> Dict[str, float]:
    token_sets = [
        (
            item.implementation_id,
            item.service_type,
            {
                token
                for token in _norm(item.profile_text).split("_")
                if len(token) >= 4 and token not in {"this", "that", "with", "from", "into", "model", "service"}
            },
        )
        for item in records
    ]
    same_service: List[float] = []
    cross_service: List[float] = []
    for index, (_left_id, left_service, left_tokens) in enumerate(token_sets):
        for _right_id, right_service, right_tokens in token_sets[index + 1 :]:
            if not left_tokens or not right_tokens:
                continue
            jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
            if left_service == right_service:
                same_service.append(jaccard)
            else:
                cross_service.append(jaccard)
    return {
        "same_service_p95_jaccard": float(np.quantile(np.asarray(same_service, dtype=np.float32), 0.95)) if same_service else 0.0,
        "cross_service_p95_jaccard": float(np.quantile(np.asarray(cross_service, dtype=np.float32), 0.95)) if cross_service else 0.0,
        "max_cross_service_jaccard": float(max(cross_service)) if cross_service else 0.0,
    }


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
