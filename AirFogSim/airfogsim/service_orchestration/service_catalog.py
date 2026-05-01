from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import yaml

from .service_spec import MicroserviceSpec, ServiceCategory

VALID_NODE_TYPES = {"vehicle", "uav", "rsu", "cloud_server"}


class ServiceCatalog:
    def __init__(self, microservices: Dict[str, MicroserviceSpec]):
        self.microservices = microservices

    @classmethod
    def from_yaml(cls, path: str) -> "ServiceCatalog":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        specs: Dict[str, MicroserviceSpec] = {}
        for item in raw.get("microservices", []):
            allowed_node_types = list(
                item.get(
                    "allowed_node_types",
                    ["vehicle", "uav", "rsu", "cloud_server"],
                )
            )
            invalid_node_types = sorted(set(allowed_node_types) - VALID_NODE_TYPES)
            if invalid_node_types:
                raise ValueError(
                    f"Invalid allowed_node_types for {item.get('ms_id', '<unknown>')}: "
                    f"{', '.join(invalid_node_types)}"
                )
            spec = MicroserviceSpec(
                ms_id=item["ms_id"],
                category=ServiceCategory(item["category"]),
                provides=list(item.get("provides", [])),
                cpu_per_mb=float(item["cpu_per_mb"]),
                memory_mb=float(item.get("memory_mb", 0.0)),
                storage_mb=float(item.get("storage_mb", 0.0)),
                input_semantic=item.get("input_semantic", "any"),
                output_semantic=item.get("output_semantic", "any"),
                output_ratio=float(item.get("output_ratio", 1.0)),
                allowed_node_types=allowed_node_types,
                stateful=bool(item.get("stateful", False)),
                cold_start_s=float(item.get("cold_start_s", 0.0)),
                deployment_cost=float(item.get("deployment_cost", 0.0)),
                min_trust=float(item.get("min_trust", 0.0)),
            )
            if spec.ms_id in specs:
                raise ValueError(f"Duplicate microservice id: {spec.ms_id}")
            if not spec.provides:
                raise ValueError(f"Microservice {spec.ms_id} must provide at least one capability")
            if spec.cpu_per_mb < 0 or spec.memory_mb < 0 or spec.storage_mb < 0:
                raise ValueError(f"Microservice {spec.ms_id} has negative resource demand")
            if spec.output_ratio < 0:
                raise ValueError(f"Microservice {spec.ms_id} has negative output_ratio")
            if spec.cold_start_s < 0 or spec.deployment_cost < 0:
                raise ValueError(f"Microservice {spec.ms_id} has negative deployment cost")
            if not 0.0 <= spec.min_trust <= 1.0:
                raise ValueError(f"Microservice {spec.ms_id} min_trust must be in [0, 1]")
            specs[spec.ms_id] = spec
        catalog = cls(specs)
        catalog.validate()
        return catalog

    def find_by_capability(self, cap: str) -> List[MicroserviceSpec]:
        return [m for m in self.microservices.values() if cap in m.provides]

    def get(self, ms_id: str) -> MicroserviceSpec:
        return self.microservices[ms_id]

    def all(self) -> Iterable[MicroserviceSpec]:
        return self.microservices.values()

    def provided_capabilities(self) -> set[str]:
        capabilities = set()
        for ms in self.microservices.values():
            capabilities.update(ms.provides)
        return capabilities

    def missing_capabilities(self, capabilities: Sequence[str]) -> List[str]:
        provided = self.provided_capabilities()
        return sorted({capability for capability in capabilities if capability not in provided})

    def validate_required_capabilities(self, capabilities: Sequence[str]) -> None:
        missing = self.missing_capabilities(capabilities)
        if missing:
            raise ValueError(f"Catalog missing required capabilities: {', '.join(missing)}")

    def validate(self) -> None:
        categories = {spec.category for spec in self.microservices.values()}
        missing_categories = sorted(
            category.value for category in ServiceCategory if category not in categories
        )
        if missing_categories:
            raise ValueError(f"Catalog missing service categories: {', '.join(missing_categories)}")
