import os
import unittest

from airfogsim.service_orchestration.deployment_registry import DeploymentRegistry
from airfogsim.service_orchestration.metrics import OrchestrationMetrics
from airfogsim.service_orchestration.service_catalog import ServiceCatalog
from airfogsim.service_orchestration.service_orchestrator import ServiceOrchestrator
from airfogsim.service_orchestration.testing import build_smoke_env


ROOT = os.path.dirname(os.path.dirname(__file__))


class DeploymentOrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ServiceCatalog.from_yaml(os.path.join(ROOT, "examples", "service_catalog.yaml"))

    def test_trust_and_host_filters(self):
        env = build_smoke_env()
        deployment = DeploymentRegistry(lazy_deploy=True)
        remote_id = self.catalog.get("remote_id_check")
        self.assertFalse(deployment.can_host(env, "Vehicle_0", remote_id))
        self.assertTrue(deployment.can_host(env, "RSU_0", remote_id))

    def test_orchestrator_deploys_and_records_cold_start(self):
        env = build_smoke_env()
        metrics = OrchestrationMetrics()
        deployment = DeploymentRegistry(lazy_deploy=True)
        orchestrator = ServiceOrchestrator(deployment, serving_policy="proposed", metrics=metrics)
        ms = self.catalog.get("edge_object_verification")
        decision = orchestrator.select_serving_node(
            env,
            {"task_node_id": "UAV_0", "task_size": 1.0, "task_cpu": 1.0},
            ms,
        )
        self.assertIsNotNone(decision)
        target_node_id, route = decision
        self.assertIn(target_node_id, {"RSU_0", "cloudServer_0"})
        self.assertGreaterEqual(len(route), 1)
        self.assertEqual(metrics.cold_start_count, 1)

    def test_cloud_route_uses_rsu_hop(self):
        env = build_smoke_env()
        metrics = OrchestrationMetrics()
        deployment = DeploymentRegistry(lazy_deploy=True)
        orchestrator = ServiceOrchestrator(deployment, serving_policy="cats_style_score", metrics=metrics)
        route = orchestrator._build_route(env, "UAV_0", "cloudServer_0")
        self.assertEqual(route, ["RSU_0", "cloudServer_0"])


if __name__ == "__main__":
    unittest.main()
