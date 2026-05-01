import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
SCRIPTS_ROOT = os.path.join(WORKSPACE_ROOT, "scripts")
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from generate_paper_validation_artifacts import failure_trace_rows, resource_rows  # noqa: E402


class PaperValidationArtifactTests(unittest.TestCase):
    def test_failure_trace_requires_per_record_evidence(self):
        records = [
            {
                "baseline": "proposed",
                "scenario": "normal",
                "seed": 0,
                "raw_metrics": {
                    "failure_trace": [
                        {
                            "task_id": "task_1",
                            "airfogsim_reason": "deadline_missed",
                            "lasdm_reason": "deadline_missed",
                            "timestamp": 3.0,
                        }
                    ]
                },
            },
            {
                "baseline": "edge_only",
                "scenario": "normal",
                "seed": 0,
                "failure_reason": "no_candidate",
                "raw_metrics": {"current_time": 0.0},
                "decisions": [{"sfc_id": "sfc_0"}],
            },
        ]

        rows = failure_trace_rows(records)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "task_1")

    def test_resource_rows_expand_runtime_snapshot_by_node(self):
        records = [
            {
                "baseline": "proposed",
                "scenario": "normal",
                "seed": 0,
                "raw_metrics": {
                    "resource_usage_timeseries": [
                        {
                            "time_s": 1.0,
                            "cpu_utilization_by_node": {"RSU_0": 0.25, "UAV_0": 0.5},
                            "computing_tasks_by_node": {"RSU_0": 1},
                            "rb_utilization": 0.5,
                            "backhaul_usage_bytes": 2000000,
                            "energy_consumed_step": 3.0,
                            "energy_by_uav": {"UAV_0": 93.0},
                        }
                    ]
                },
            }
        ]

        rows = resource_rows(records)

        self.assertEqual({row["node_id"] for row in rows}, {"RSU_0", "UAV_0"})
        rsu = next(row for row in rows if row["node_id"] == "RSU_0")
        uav = next(row for row in rows if row["node_id"] == "UAV_0")
        self.assertEqual(rsu["cpu_utilization"], 0.25)
        self.assertEqual(rsu["backhaul_usage_mb"], 2.0)
        self.assertEqual(uav["energy_consumption"], 3.0)


if __name__ == "__main__":
    unittest.main()
