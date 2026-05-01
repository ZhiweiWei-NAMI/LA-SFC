import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.baselines import baseline_names, get_baseline_policy
from airfogsim.lasdm.benchmark_adapter import (
    SUMMARY_FIELDS,
    SENSITIVITY_SUMMARY_FIELDS,
    load_lasdm_sensitivity_config,
    load_lasdm_benchmark_config,
    run_lasdm_benchmark_suite,
    run_lasdm_sensitivity_suite,
)


CONFIG_PATH = os.path.join(METHOD_ROOT, "configs", "lasdm_airfogsim.yaml")
SENSITIVITY_CONFIG_PATH = os.path.join(METHOD_ROOT, "configs", "lasdm_sensitivity.yaml")
SCRIPT_PATH = os.path.join(METHOD_ROOT, "run_lasdm_benchmark.py")
SENSITIVITY_SCRIPT_PATH = os.path.join(METHOD_ROOT, "run_lasdm_sensitivity.py")


def extract_output_dir(stdout: str) -> str:
    match = re.search(r'"output_dir":\s*"([^"]+)"', stdout)
    if match is None:
        raise AssertionError(f"output_dir not found in stdout:\n{stdout}")
    return match.group(1)


class LASDMBenchmarkAdapterTests(unittest.TestCase):
    def test_config_loads_existing_lasdm_experiment_matrix(self):
        config = load_lasdm_benchmark_config(CONFIG_PATH)
        self.assertEqual(config["experiment"]["seeds"], list(range(10)))
        self.assertEqual(config["experiment"]["mode"], "offline")
        self.assertEqual(config["runtime"]["env_adapter_import_path"], "airfogsim.lasdm.env_adapter.LASDMEnvAdapter")
        self.assertIn("proposed", config["experiment"]["baselines"])
        self.assertEqual(
            [scenario["name"] for scenario in config["experiment"]["scenarios"]],
            ["normal", "high_mobility", "load_burst", "link_fault", "airspace_event"],
        )
        self.assertIn("deployment_only", baseline_names())
        self.assertIn("nearest_edge", baseline_names())
        self.assertIn("cats_style", baseline_names())
        self.assertIn("uav_only", baseline_names())
        self.assertIn("edge_only", baseline_names())
        self.assertIn("lasdm_greedy", baseline_names())
        self.assertEqual(get_baseline_policy("edge_only").allowed_node_types, ("rsu", "cloud_server"))
        self.assertEqual(get_baseline_policy("lasdm_greedy").name, "lasdm_greedy")

    def test_single_run_writes_manifest_per_run_summary_and_aggregate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, payload = run_lasdm_benchmark_suite(
                CONFIG_PATH,
                baseline_override="proposed",
                seed_override=0,
                scenario_override="normal",
                output_root_override=temp_dir,
            )
            self.assertEqual(exit_code, 0, msg=payload)
            output_dir = payload["output_dir"]
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "manifest.json")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "aggregate.csv")))
            run_path = os.path.join(output_dir, "runs", "proposed__normal__seed_0.json")
            self.assertTrue(os.path.isfile(run_path))

            with open(run_path, "r", encoding="utf-8") as file:
                run_payload = json.load(file)
            self.assertEqual(run_payload["baseline"], "proposed")
            self.assertEqual(run_payload["scenario"], "normal")
            self.assertEqual(run_payload["mode"], "offline")
            self.assertTrue(run_payload["completed"])
            self.assertIn("raw_metrics", run_payload)
            self.assertIn("decisions", run_payload)
            self.assertEqual(run_payload["graph_submit_count"], 4)
            self.assertEqual(run_payload["accepted_decision_count"], 4)

            with open(os.path.join(output_dir, "summary.csv"), "r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline"], "proposed")
            self.assertEqual(rows[0]["scenario"], "normal")
            self.assertEqual(rows[0]["status"], "succeeded")
            self.assertEqual(rows[0]["graph_submitted_count"], "4")
            self.assertEqual(rows[0]["failure_reason"], "none")
            self.assertIn("avg_graph_finish_time", rows[0])
            self.assertEqual(list(rows[0].keys()), SUMMARY_FIELDS)
            with open(os.path.join(output_dir, "manifest.json"), "r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["mode"], "offline")

    def test_batch_run_aggregates_by_baseline_and_scenario(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "mini_lasdm.yaml")
            with open(CONFIG_PATH, "r", encoding="utf-8") as source:
                base_config = source.read()
            with open(config_path, "w", encoding="utf-8") as file:
                file.write(
                    base_config
                    + textwrap.dedent(
                        """

                        output:
                          write_manifest: true
                          write_per_run_json: true
                          write_summary_csv: true
                          write_aggregate_csv: true
                          overwrite: false
                        """
                    )
                )

            exit_code, payload = run_lasdm_benchmark_suite(
                config_path,
                baseline_override="proposed",
                scenario_override="normal",
                output_root_override=temp_dir,
            )
            self.assertEqual(exit_code, 0, msg=payload)
            output_dir = payload["output_dir"]
            with open(os.path.join(output_dir, "aggregate.csv"), "r", encoding="utf-8") as file:
                aggregate_rows = list(csv.DictReader(file))
            self.assertEqual(len(aggregate_rows), 1)
            self.assertEqual(aggregate_rows[0]["baseline"], "proposed")
            self.assertEqual(aggregate_rows[0]["scenario"], "normal")
            self.assertEqual(aggregate_rows[0]["run_count"], "1")
            self.assertIn("graph_completion_ratio_mean", aggregate_rows[0])

    def test_sensitivity_config_expands_deadline_network_load_matrix(self):
        config = load_lasdm_sensitivity_config(SENSITIVITY_CONFIG_PATH)
        scenarios = config["experiment"]["scenarios"]
        combos = {
            (scenario["deadline_level"], scenario["network_level"], scenario["load_level"])
            for scenario in scenarios
        }
        self.assertEqual(len(scenarios), 27)
        self.assertEqual(len(combos), 27)
        self.assertIn(("tight", "stable", "low"), combos)
        self.assertIn(("loose", "burst", "high"), combos)
        self.assertEqual(config["experiment"]["mode"], "offline")
        self.assertEqual(config["experiment"]["baselines"], ["lasdm_greedy", "proposed"])
        self.assertEqual(scenarios[0]["deadline_s"], 1.2)
        self.assertEqual(scenarios[0]["network_fault"], "none")
        self.assertEqual(scenarios[0]["load_multiplier"], 0.7)

    def test_sensitivity_run_writes_dimension_summary_without_airfogsim(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, payload = run_lasdm_sensitivity_suite(
                SENSITIVITY_CONFIG_PATH,
                baseline_override="proposed",
                seed_override=0,
                output_root_override=temp_dir,
            )
            self.assertEqual(exit_code, 0, msg=payload)
            self.assertEqual(payload["run_count"], 27)
            summary_path = os.path.join(payload["output_dir"], "sensitivity_summary.csv")
            self.assertEqual(payload["sensitivity_summary_path"], summary_path)
            self.assertTrue(os.path.isfile(summary_path))

            with open(summary_path, "r", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 27)
            self.assertEqual(list(rows[0].keys()), SENSITIVITY_SUMMARY_FIELDS)
            self.assertEqual({row["baseline"] for row in rows}, {"proposed"})
            self.assertEqual({row["deadline"] for row in rows}, {"tight", "medium", "loose"})
            self.assertEqual({row["network"] for row in rows}, {"stable", "degraded", "burst"})
            self.assertEqual({row["load"] for row in rows}, {"low", "medium", "high"})
            self.assertTrue(all(row["run_count"] == "1" for row in rows))
            tight_rows = [row for row in rows if row["deadline"] == "tight"]
            self.assertTrue(all(row["deadline_s"] == "1.2" for row in tight_rows))
            self.assertIn("deadline_satisfaction_ratio_mean", rows[0])

    def test_cli_runner_prints_standard_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    SCRIPT_PATH,
                    "--baseline",
                    "lasdm_greedy",
                    "--scenario",
                    "normal",
                    "--seed",
                    "0",
                    "--mode",
                    "offline",
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
            self.assertIn('"baseline": "lasdm_greedy"', result.stdout)
            output_dir = extract_output_dir(result.stdout)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "manifest.json")))

    def test_sensitivity_cli_runner_prints_summary_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    SENSITIVITY_SCRIPT_PATH,
                    "--baseline",
                    "proposed",
                    "--scenario",
                    "deadline_tight__network_stable__load_low",
                    "--seed",
                    "0",
                    "--mode",
                    "offline",
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
            self.assertIn('"sensitivity_summary_path":', result.stdout)
            output_dir = extract_output_dir(result.stdout)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "sensitivity_summary.csv")))

    def test_uav_only_records_no_candidate_without_runner_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, payload = run_lasdm_benchmark_suite(
                CONFIG_PATH,
                baseline_override="uav_only",
                seed_override=0,
                scenario_override="normal",
                output_root_override=temp_dir,
            )
            self.assertEqual(exit_code, 0, msg=payload)
            self.assertEqual(payload["completed"], True)
            self.assertEqual(payload["graph_failed_count"], 4)
            self.assertEqual(payload["rejected_decision_count"], 4)
            self.assertIn("no_candidate", payload["failure_reason_count"])

    def test_airfogsim_mode_uses_env_adapter_and_writes_standard_outputs(self):
        module_name = "airfogsim.lasdm.env_adapter"
        module = types.ModuleType(module_name)
        calls = []
        test_case = self

        class FakeLASDMEnvAdapter:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def run_benchmark(self, **kwargs):
                calls.append(("run", kwargs))
                test_case.assertEqual(kwargs["baseline_name"], "proposed")
                test_case.assertEqual(kwargs["scenario"]["name"], "normal")
                return {
                    "completed": True,
                    "raw_metrics": {
                        "graph_submitted_count": 1,
                        "graph_complete_count": 1,
                        "graph_failed_count": 0,
                        "graph_timeout_count": 0,
                        "deadline_satisfaction_ratio": 1.0,
                        "avg_graph_finish_time": 1.25,
                        "accepted_decision_count": 1,
                        "payload_tx_mb": 2.0,
                    },
                    "decisions": [
                        {
                            "sfc_id": "runtime_sfc",
                            "assignments": {"preprocess": "uav0_preprocess_0"},
                            "routes": {"preprocess": ["UAV_0", "UAV_0"]},
                            "diagnostics": {"candidate_counts": {"preprocess": 1}},
                            "score": 0.9,
                        }
                    ],
                }

        module.LASDMEnvAdapter = FakeLASDMEnvAdapter
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                exit_code, payload = run_lasdm_benchmark_suite(
                    CONFIG_PATH,
                    baseline_override="proposed",
                    seed_override=0,
                    scenario_override="normal",
                    output_root_override=temp_dir,
                    mode_override="airfogsim",
                )
                self.assertEqual(exit_code, 0, msg=payload)
                self.assertEqual(payload["mode"], "airfogsim")
                self.assertEqual(payload["graph_submitted_count"], 1)
                self.assertEqual(payload["graph_complete_count"], 1)
                output_dir = payload["output_dir"]
                self.assertTrue(os.path.isfile(os.path.join(output_dir, "manifest.json")))
                self.assertTrue(os.path.isfile(os.path.join(output_dir, "summary.csv")))
                self.assertTrue(os.path.isfile(os.path.join(output_dir, "aggregate.csv")))
                with open(os.path.join(output_dir, "manifest.json"), "r", encoding="utf-8") as file:
                    manifest = json.load(file)
                self.assertEqual(manifest["mode"], "airfogsim")
                with open(os.path.join(output_dir, "summary.csv"), "r", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(rows[0]["status"], "succeeded")
                self.assertEqual(rows[0]["graph_submitted_count"], "1")
                self.assertEqual(rows[0]["avg_graph_finish_time"], "1.25")
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        self.assertEqual([name for name, _ in calls], ["init", "run"])


if __name__ == "__main__":
    unittest.main()
