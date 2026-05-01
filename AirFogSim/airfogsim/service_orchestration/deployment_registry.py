from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

from .service_spec import MicroserviceSpec


AIRFOGSIM_TYPE_MAP = {
    "V": "vehicle",
    "U": "uav",
    "I": "rsu",
    "C": "cloud_server",
}


class DeploymentRegistry:
    """Track which microservices are already deployed on which nodes."""

    def __init__(self, lazy_deploy: bool = True):
        self.lazy_deploy = lazy_deploy
        self.deployed: Dict[str, Set[str]] = defaultdict(set)

    def is_deployed(self, node_id: str, ms_id: str) -> bool:
        return ms_id in self.deployed.get(node_id, set())

    def deploy(self, node_id: str, ms_id: str) -> None:
        self.deployed[node_id].add(ms_id)

    def can_host(self, env, node_id: str, ms: MicroserviceSpec) -> bool:
        node_type = AIRFOGSIM_TYPE_MAP.get(env._getNodeTypeById(node_id))
        if node_type not in ms.allowed_node_types:
            return False

        node = env._getNodeById(node_id)
        if node is None:
            return False

        fog_profile = node.getFogProfile() if hasattr(node, "getFogProfile") else {}
        if fog_profile.get("cpu", 0.0) <= 0:
            return False
        if fog_profile.get("memory", float("inf")) < ms.memory_mb:
            return False
        if fog_profile.get("storage", float("inf")) < ms.storage_mb:
            return False
        if hasattr(env, "getNodeTrustScore") and env.getNodeTrustScore(node_id) < ms.min_trust:
            return False
        return self.is_deployed(node_id, ms.ms_id) or self.lazy_deploy
