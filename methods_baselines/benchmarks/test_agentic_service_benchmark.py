import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

TEST_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(TEST_DIR, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
BENCHMARK_ROOT = os.path.join(WORKSPACE_ROOT, "methods_baselines", "benchmarks", "agentic_service_orchestration")
for path in (AIRFOGSIM_ROOT, BENCHMARK_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_runner import (
    load_benchmark_config,
    validate_airfogsim_setup,
)


SCRIPT = os.path.join(BENCHMARK_ROOT, "main_agentic_service_orchestration.py")


def extract_output_dir(stdout: str) -> str:
    match = re.search(r'"output_dir":\s*"([^"]+)"', stdout)
    if match is None:
        raise AssertionError(f"output_dir not found in stdout:\n{stdout}")
    return match.group(1)


def run_benchmark(*args):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=WORKSPACE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


class AgenticServiceBenchmarkTests(unittest.TestCase):
    def test_single_smoke_run_writes_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_benchmark("--baseline", "proposed", "--output-root", temp_dir)
            self.assertEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn('"baseline": "proposed"', result.stdout)

            output_dir = extract_output_dir(result.stdout)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "manifest.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "aggregate.csv")))
            self.assertTrue(
                os.path.isfile(
                    os.path.join(output_dir, "runs", "proposed__seed_0.json"),
                )
            )

            with open(os.path.join(output_dir, "summary.csv"), "r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline"], "proposed")

    def test_batch_smoke_run_writes_all_baselines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "batch_smoke.yaml")
            with open(config_path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        """
                        experiment:
                          name: smoke_suite
                          mode: smoke
                          baselines:
                            - fixed_sfc
                            - deployment_only
                            - nearest_edge
                            - cats_style
                            - centralized_greedy
                            - regional_distributed_greedy
                            - edge_only
                            - uav_only
                            - proposed
                          seeds: [0]
                          catalog_path: examples/service_catalog.yaml
                          intents_path: examples/task_intents.yaml
                          airfogsim_config_path: examples/config_agentic_service.yaml
                          smoke_max_rounds: 12
                          airfogsim_max_steps: 10
                        intent_source:
                          smoke:
                            type: static_yaml
                          airfogsim:
                            type: poisson_low_altitude
                            arrival_prob: 0.3
                        output:
                          root_dir: experiment_artifacts/raw_data/agentic_service_orchestration
                          write_manifest: true
                          write_per_run_json: true
                          write_summary_csv: true
                          write_aggregate_csv: true
                          overwrite: false
                        """
                    ).strip()
                )

            result = run_benchmark("--config", config_path, "--output-root", temp_dir)
            self.assertEqual(result.returncode, 0, msg=result.stdout)

            output_dir = extract_output_dir(result.stdout)
            with open(os.path.join(output_dir, "summary.csv"), "r", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))
            with open(os.path.join(output_dir, "aggregate.csv"), "r", encoding="utf-8") as file:
                aggregate_rows = list(csv.DictReader(file))

            self.assertEqual(len(summary_rows), 9)
            self.assertEqual(len(aggregate_rows), 9)

    def test_per_run_json_contains_raw_and_flattened_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_benchmark("--baseline", "regional_distributed_greedy", "--output-root", temp_dir)
            self.assertEqual(result.returncode, 0, msg=result.stdout)

            output_dir = extract_output_dir(result.stdout)
            run_path = os.path.join(
                output_dir,
                "runs",
                "regional_distributed_greedy__seed_0.json",
            )
            with open(run_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            self.assertEqual(payload["baseline"], "regional_distributed_greedy")
            self.assertIn("raw_metrics", payload)
            self.assertIn("graph_submit_count", payload)
            self.assertIn("graph_complete_count", payload)
            self.assertIn("task_done_num", payload)
            self.assertIn("simulation_time_end", payload)
            self.assertIn("p95_graph_finish_time", payload)
            self.assertIn("deadline_satisfaction_ratio", payload)
            self.assertIn("missing_service_capability_mean", payload)
            self.assertIn("extra_service_capability_mean", payload)

    def test_service_plane_baselines_are_runnable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for baseline in ["edge_only", "uav_only"]:
                result = run_benchmark("--baseline", baseline, "--output-root", temp_dir)
                self.assertEqual(result.returncode, 0, msg=result.stdout)
                self.assertIn(f'"baseline": "{baseline}"', result.stdout)

    def test_airfogsim_validation_failure_writes_error_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "invalid_airfogsim.yaml")
            with open(config_path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        """
                        experiment:
                          name: invalid_airfogsim
                          mode: airfogsim
                          baselines: [proposed]
                          seeds: [7]
                          catalog_path: examples/service_catalog.yaml
                          intents_path: examples/task_intents.yaml
                          airfogsim_config_path: missing_configs/nope.yaml
                          smoke_max_rounds: 12
                          airfogsim_max_steps: 10
                        intent_source:
                          smoke:
                            type: static_yaml
                          airfogsim:
                            type: poisson_low_altitude
                            arrival_prob: 0.3
                        output:
                          root_dir: experiment_artifacts/raw_data/agentic_service_orchestration
                          write_manifest: true
                          write_per_run_json: true
                          write_summary_csv: true
                          write_aggregate_csv: true
                          overwrite: false
                        """
                    ).strip()
                )

            result = run_benchmark("--config", config_path, "--output-root", temp_dir)
            self.assertNotEqual(result.returncode, 0, msg=result.stdout)
            self.assertIn("airfogsim_config_path not found", result.stdout)

            output_dir = extract_output_dir(result.stdout)
            run_path = os.path.join(output_dir, "runs", "proposed__seed_7.json")
            with open(run_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

            self.assertFalse(payload["completed"])
            self.assertEqual(payload["exit_code"], 1)
            self.assertIn("airfogsim_config_path not found", payload["error"])

    def test_config_load_rejects_missing_intent_capability(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            intents_path = os.path.join(temp_dir, "bad_intents.yaml")
            config_path = os.path.join(temp_dir, "bad_capability.yaml")
            with open(intents_path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        """
                        intents:
                          - intent_id: bad
                            source_node_id: UAV_0
                            category: mission_oriented_application
                            required_capabilities: [not_in_catalog]
                            payload: {semantic_type: video, input_mb: 1.0}
                            qos: {deadline_s: 1.0}
                        """
                    ).strip()
                )
            with open(config_path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        f"""
                        experiment:
                          name: bad_capability
                          mode: smoke
                          baselines: [proposed]
                          seeds: [0]
                          catalog_path: examples/service_catalog.yaml
                          intents_path: {intents_path}
                          airfogsim_config_path: examples/config_agentic_service.yaml
                        """
                    ).strip()
                )

            with self.assertRaisesRegex(ValueError, "not_in_catalog"):
                load_benchmark_config(config_path)

    def test_airfogsim_validation_rejects_placeholder_uav_traffic_path(self):
        config = {
            "sumo": {
                "sumo_net": __file__,
                "sumo_config": __file__,
            },
            "traffic": {
                "traffic_mode": "SUMO",
                "uav_traffic_file": "path/to/uav_traffic_file.csv",
            },
        }

        with self.assertRaisesRegex(FileNotFoundError, "placeholder path"):
            validate_airfogsim_setup(__file__, config)


if __name__ == "__main__":
    unittest.main()
