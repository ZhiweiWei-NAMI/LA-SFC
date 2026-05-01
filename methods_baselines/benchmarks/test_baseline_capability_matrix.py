import ast
import csv
import os
import subprocess
import sys
import tempfile
import unittest


TEST_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(TEST_DIR, "../.."))
SCRIPT_DIR = os.path.join(WORKSPACE_ROOT, "analysis", "scripts")
SCRIPT_PATH = os.path.join(SCRIPT_DIR, "generate_baseline_capability_matrix.py")
CSV_PATH = os.path.join(WORKSPACE_ROOT, "analysis", "baseline_capability_matrix.csv")
LASDM_BASELINES_PATH = os.path.join(WORKSPACE_ROOT, "AirFogSim", "airfogsim", "lasdm", "baselines.py")
AGENTIC_RUNNER_PATH = os.path.join(
    WORKSPACE_ROOT,
    "methods_baselines",
    "benchmarks",
    "agentic_service_orchestration",
    "benchmark_runner.py",
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from generate_baseline_capability_matrix import FIELDNAMES, MATRIX_ROWS


def registry_keys_from_ast(path):
    with open(path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=path)

    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BASELINE_REGISTRY" for target in node.targets
        ):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "BASELINE_REGISTRY":
                value = node.value
        if value is None:
            continue
        if not isinstance(value, ast.Dict):
            raise AssertionError(f"BASELINE_REGISTRY is not a literal dict in {path}")
        return {key.value for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    raise AssertionError(f"BASELINE_REGISTRY not found in {path}")


class BaselineCapabilityMatrixTests(unittest.TestCase):
    def test_static_data_covers_current_lasdm_and_agentic_baselines(self):
        rows_by_key = {(row["family"], row["baseline"]): row for row in MATRIX_ROWS}

        expected_lasdm = registry_keys_from_ast(LASDM_BASELINES_PATH)
        expected_agentic = registry_keys_from_ast(AGENTIC_RUNNER_PATH)

        self.assertEqual(
            {baseline for family, baseline in rows_by_key if family == "lasdm"},
            expected_lasdm,
        )
        self.assertEqual(
            {baseline for family, baseline in rows_by_key if family == "agentic"},
            expected_agentic,
        )

        for key in [
            ("lasdm", "edge_only"),
            ("lasdm", "uav_only"),
            ("agentic", "edge_only"),
            ("agentic", "uav_only"),
            ("agentic", "regional_distributed_greedy"),
        ]:
            self.assertEqual(rows_by_key[key]["intentionally_limited"], "true")

        self.assertEqual(rows_by_key[("agentic", "uav_only")]["limitation_axis"], "candidate_scope_soft")
        self.assertEqual(rows_by_key[("lasdm", "proposed")]["intentionally_limited"], "false")
        self.assertEqual(rows_by_key[("agentic", "proposed")]["intentionally_limited"], "false")

    def test_generator_writes_expected_columns_and_check_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "baseline_capability_matrix.csv")
            result = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--output", output_path],
                cwd=WORKSPACE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout)

            with open(output_path, "r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(rows[0].keys(), dict.fromkeys(FIELDNAMES).keys())
            self.assertEqual(len(rows), len(MATRIX_ROWS))

            check_result = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--output", output_path, "--check"],
                cwd=WORKSPACE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(check_result.returncode, 0, msg=check_result.stdout)

    def test_checked_in_csv_matches_generator(self):
        result = subprocess.run(
            [sys.executable, SCRIPT_PATH, "--output", CSV_PATH, "--check"],
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)


if __name__ == "__main__":
    unittest.main()
