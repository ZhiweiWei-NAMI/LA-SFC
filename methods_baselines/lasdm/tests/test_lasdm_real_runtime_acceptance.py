import csv
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
BENCHMARK_ROOT = os.path.join(
    WORKSPACE_ROOT,
    "methods_baselines",
    "benchmarks",
    "agentic_service_orchestration",
)
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.benchmark_adapter import SUMMARY_FIELDS as LASDM_SUMMARY_FIELDS


LASDM_SCRIPT = os.path.join(METHOD_ROOT, "run_lasdm_benchmark.py")
ASO_SCRIPT = os.path.join(BENCHMARK_ROOT, "main_agentic_service_orchestration.py")
PROTOCOL_PATH = os.path.join(WORKSPACE_ROOT, "docs", "experiment_protocol.md")

PAPER_SEEDS = list(range(10))
PAPER_SCENARIOS = [
    "normal",
    "high_mobility",
    "load_burst",
    "link_fault",
    "airspace_event",
]
PAPER_BASELINES = [
    "fixed_sfc",
    "deployment_only",
    "nearest_edge",
    "cats_style",
    "centralized_greedy",
    "regional_distributed_greedy",
    "edge_only",
    "uav_only",
    "proposed",
]


def _load_agentic_benchmark_runner():
    spec = importlib.util.spec_from_file_location(
        "agentic_service_benchmark_runner",
        os.path.join(BENCHMARK_ROOT, "benchmark_runner.py"),
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load agentic service benchmark_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_output_dir(stdout: str) -> str:
    match = re.search(r'"output_dir":\s*"([^"]+)"', stdout)
    if match is None:
        raise AssertionError(f"output_dir not found in stdout:\n{stdout}")
    return match.group(1)


class LASDMRealRuntimeAcceptanceTests(unittest.TestCase):
    def test_paper_experiment_matrix_contract(self):
        self.assertEqual(PAPER_SEEDS, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(
            PAPER_SCENARIOS,
            ["normal", "high_mobility", "load_burst", "link_fault", "airspace_event"],
        )
        self.assertEqual(
            PAPER_BASELINES,
            [
                "fixed_sfc",
                "deployment_only",
                "nearest_edge",
                "cats_style",
                "centralized_greedy",
                "regional_distributed_greedy",
                "edge_only",
                "uav_only",
                "proposed",
            ],
        )

    def test_metric_contract_covers_graph_task_latency_and_cost_fields(self):
        agentic_runner = _load_agentic_benchmark_runner()
        agentic_fields = set(agentic_runner.SUMMARY_FIELDS)
        lasdm_fields = set(LASDM_SUMMARY_FIELDS)

        self.assertLessEqual(
            {
                "graph_completion_ratio",
                "deadline_satisfaction_ratio",
                "avg_graph_finish_time",
                "p95_graph_finish_time",
                "cold_start_count",
            },
            lasdm_fields,
        )
        self.assertIn("task_success_ratio", agentic_fields)

    def test_protocol_separates_unit_smoke_and_sumo_traci_commands(self):
        with open(PROTOCOL_PATH, "r", encoding="utf-8") as file:
            protocol = file.read()

        self.assertIn("Default non-SUMO checks", protocol)
        self.assertIn("SUMO/TraCI gated integration checks", protocol)
        self.assertIn("RUN_LASDM_SUMO_INTEGRATION=1", protocol)
        self.assertIn("pytest -q methods_baselines/lasdm/tests", protocol)
        self.assertIn("main_agentic_service_orchestration.py", protocol)

    def test_lasdm_smoke_command_writes_required_artifact_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    LASDM_SCRIPT,
                    "--baseline",
                    "lasdm_greedy",
                    "--scenario",
                    "normal",
                    "--seed",
                    "0",
                    "--output-root",
                    temp_dir,
                ],
                cwd=WORKSPACE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            output_dir = _extract_output_dir(result.stdout)
            summary_path = os.path.join(output_dir, "summary.csv")
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "manifest.json")))
            self.assertTrue(os.path.isfile(summary_path))

            with open(summary_path, "r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline"], "lasdm_greedy")
            self.assertEqual(rows[0]["scenario"], "normal")
            self.assertLessEqual(
                {
                    "graph_completion_ratio",
                    "deadline_satisfaction_ratio",
                    "avg_graph_finish_time",
                    "p95_graph_finish_time",
                    "cold_start_count",
                    "deployment_cost",
                },
                set(rows[0].keys()),
            )

    def test_sumo_traci_airfogsim_integration_command_is_opt_in(self):
        if os.environ.get("RUN_LASDM_SUMO_INTEGRATION") != "1":
            self.skipTest("Set RUN_LASDM_SUMO_INTEGRATION=1 to run SUMO/TraCI integration.")
        if shutil.which("sumo") is None:
            self.fail("SUMO binary is required for RUN_LASDM_SUMO_INTEGRATION=1.")
        try:
            import traci  # noqa: F401
        except ImportError as exc:
            self.fail(f"traci Python package is required for SUMO/TraCI integration: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "airfogsim_sumo_integration.yaml")
            with open(config_path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        """
                        experiment:
                          name: lasdm_real_runtime_sumo_check
                          mode: airfogsim
                          baselines: [proposed]
                          seeds: [0]
                          catalog_path: examples/service_catalog.yaml
                          intents_path: examples/task_intents.yaml
                          airfogsim_config_path: examples/config_local_map_agentic_service.yaml
                          airfogsim_max_steps: 3
                        intent_source:
                          airfogsim:
                            type: poisson_low_altitude
                            arrival_prob: 0.1
                        output:
                          write_manifest: true
                          write_per_run_json: true
                          write_summary_csv: true
                          write_aggregate_csv: true
                        """
                    ).strip()
                )

            result = subprocess.run(
                [
                    sys.executable,
                    ASO_SCRIPT,
                    "--config",
                    config_path,
                    "--output-root",
                    temp_dir,
                ],
                cwd=WORKSPACE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            output_dir = _extract_output_dir(result.stdout)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "summary.csv")))


if __name__ == "__main__":
    unittest.main()
