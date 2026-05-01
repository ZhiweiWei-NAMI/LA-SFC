import csv
import json
import os
import sys
import tempfile
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from airfogsim.enum_const import EnumerateConstants
from methods_baselines.lasdm.scripts.generate_failure_trace import (
    FAILURE_CONTRACT,
    generate_failure_trace,
)


class LASDMFailureTraceTests(unittest.TestCase):
    def test_failure_contract_covers_required_families(self):
        required = {
            "no_candidate",
            "link_disconnect",
            "energy_exhausted",
            "deadline_too_small",
            "node_moved_away",
        }
        self.assertEqual(set(FAILURE_CONTRACT), required)
        self.assertEqual(FAILURE_CONTRACT["no_candidate"]["canonical_reason"], "no_candidate")
        self.assertEqual(FAILURE_CONTRACT["energy_exhausted"]["canonical_reason"], "energy_violation")
        self.assertEqual(FAILURE_CONTRACT["deadline_too_small"]["canonical_reason"], "deadline_missed")
        self.assertEqual(FAILURE_CONTRACT["link_disconnect"]["native_airfogsim_support"], "partial")
        self.assertEqual(FAILURE_CONTRACT["node_moved_away"]["native_airfogsim_support"], "partial")
        self.assertTrue(FAILURE_CONTRACT["energy_exhausted"]["needs_native_runtime_hook"])

    def test_generates_failure_trace_from_run_json_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = os.path.join(tmpdir, "runs")
            os.makedirs(runs_dir)
            run_path = os.path.join(runs_dir, "proposed__stress__seed_7.json")
            run_payload = {
                "baseline": "proposed",
                "scenario": "stress",
                "seed": 7,
                "mode": "airfogsim",
                "completed": True,
                "exit_code": 0,
                "status": "failed",
                "raw_metrics": {
                    "failure_reason_count": {"no_candidate": 1},
                    "events": [
                        {
                            "event_type": "failed",
                            "entity_id": "sfc_link",
                            "payload": {
                                "reason": "deadline_missed",
                                "airfogsim_failure_code": EnumerateConstants.TASK_FAIL_OUT_OF_TTI,
                                "airfogsim_failure_reason": "Task fails due to transmission timeout.",
                                "task_id": "task_link",
                            },
                        },
                        {
                            "event_type": "failed",
                            "entity_id": "sfc_deadline",
                            "payload": {
                                "reason": "deadline_missed",
                                "airfogsim_failure_code": EnumerateConstants.TASK_FAIL_OUT_OF_DDL,
                                "airfogsim_failure_reason": "Task fails due to out of deadline.",
                                "task_id": "task_deadline",
                            },
                        },
                        {
                            "event_type": "failed",
                            "entity_id": "sfc_energy",
                            "payload": {
                                "reason": "energy_violation",
                                "airfogsim_status": "battery exhausted",
                            },
                        },
                        {
                            "event_type": "failed",
                            "entity_id": "sfc_moved",
                            "payload": {
                                "reason": "no_candidate",
                                "airfogsim_status": "node_moved_out_of_coverage",
                            },
                        },
                    ],
                },
                "decisions": [
                    {
                        "sfc_id": "sfc_no_candidate",
                        "rejected_reason": "no_candidate",
                        "diagnostics": {"candidate_counts": {"detect": 0}},
                    }
                ],
            }
            with open(run_path, "w", encoding="utf-8") as file:
                json.dump(run_payload, file)

            with open(os.path.join(tmpdir, "summary.csv"), "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["baseline", "scenario", "seed", "status", "failure_reason_count"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "baseline": "proposed",
                        "scenario": "stress",
                        "seed": 7,
                        "status": "failed",
                        "failure_reason_count": json.dumps({"no_candidate": 1}),
                    }
                )

            output_path = os.path.join(tmpdir, "failure_trace.json")
            trace = generate_failure_trace(tmpdir, output_path=output_path)

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(trace["schema_version"], "lasdm_failure_trace_v1")
            self.assertEqual(trace["source"]["run_count"], 1)
            self.assertGreaterEqual(trace["reason_histogram"]["no_candidate"], 2)
            self.assertEqual(trace["reason_histogram"]["energy_violation"], 1)
            self.assertIn("link_disconnect", trace["family_histogram"])
            self.assertIn("deadline_too_small", trace["family_histogram"])
            self.assertIn("energy_exhausted", trace["family_histogram"])
            self.assertIn("node_moved_away", trace["family_histogram"])
            self.assertEqual(trace["coverage"]["energy_exhausted"]["native_airfogsim_support"], "no")
            self.assertTrue(trace["coverage"]["node_moved_away"]["needs_native_runtime_hook"])


if __name__ == "__main__":
    unittest.main()
