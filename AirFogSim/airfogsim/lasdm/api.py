from __future__ import annotations

from typing import List

import yaml

from .instance_directory import ServiceInstance, ServiceInstanceDirectory
from .manager import LASDMManager
from .model import LASDMServiceChain
from .orchestrator import LASDMOrchestrator


def load_chains_from_yaml(path: str) -> List[LASDMServiceChain]:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return [LASDMServiceChain.from_dict(item) for item in raw.get("service_chains", [])]


def load_instances_from_yaml(path: str) -> ServiceInstanceDirectory:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    directory = ServiceInstanceDirectory()
    for item in raw.get("service_instances", []):
        directory.register(ServiceInstance.from_dict(item))
    return directory


def build_manager_from_yaml(path: str) -> LASDMManager:
    with open(path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    directory = ServiceInstanceDirectory()
    for item in raw.get("service_instances", []):
        directory.register(ServiceInstance.from_dict(item))
    weights = raw.get("orchestrator", {}).get("score_weights", {})
    manager = LASDMManager(directory=directory, orchestrator=LASDMOrchestrator(directory, score_weights=weights))
    for chain_raw in raw.get("service_chains", []):
        manager.submit(LASDMServiceChain.from_dict(chain_raw), current_time=0.0)
    return manager
