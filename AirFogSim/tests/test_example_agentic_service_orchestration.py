import os
import subprocess
import sys
import unittest

from airfogsim.service_orchestration.agentic_service_algorithm import AgenticServiceAlgorithmModule
from airfogsim.service_orchestration.testing import build_smoke_env


ROOT = os.path.dirname(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.dirname(ROOT)


class ExampleSmokeTests(unittest.TestCase):
    def test_communication_filter_skips_local_and_non_wireless_hops(self):
        env = build_smoke_env()
        algorithm = AgenticServiceAlgorithmModule(
            catalog_path=os.path.join(ROOT, "examples", "service_catalog.yaml"),
            intent_generator=lambda env: [],
        )
        algorithm.initialize(env)

        self.assertFalse(
            algorithm._needs_wireless_rb(
                env,
                {
                    "task_id": "local",
                    "task_node_id": "UAV_0",
                    "current_node_id": "UAV_0",
                    "to_offload_route": ["UAV_0"],
                    "executed_locally": True,
                },
            )
        )
        self.assertFalse(
            algorithm._needs_wireless_rb(
                env,
                {
                    "task_id": "cloud_hop",
                    "task_node_id": "RSU_0",
                    "current_node_id": "RSU_0",
                    "to_offload_route": ["cloudServer_0"],
                    "executed_locally": False,
                },
            )
        )
        self.assertTrue(
            algorithm._needs_wireless_rb(
                env,
                {
                    "task_id": "wireless",
                    "task_node_id": "UAV_0",
                    "current_node_id": "UAV_0",
                    "to_offload_route": ["RSU_0"],
                    "executed_locally": False,
                },
            )
        )

    def test_example_smoke(self):
        script = os.path.join(ROOT, "examples", "example_agentic_service_orchestration.py")
        result = subprocess.run(
            [sys.executable, script, "--smoke-test"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("Smoke test passed.", result.stdout)

    def test_regional_benchmark_smoke(self):
        script = os.path.join(
            WORKSPACE_ROOT,
            "methods_baselines",
            "benchmarks",
            "agentic_service_orchestration",
            "main_agentic_service_orchestration.py",
        )
        result = subprocess.run(
            [sys.executable, script, "--baseline", "regional_distributed_greedy"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn('"baseline": "regional_distributed_greedy"', result.stdout)


if __name__ == "__main__":
    unittest.main()
