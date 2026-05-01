import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
SCRIPTS_ROOT = os.path.join(WORKSPACE_ROOT, "scripts")
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from runtime_significance_analysis import significance_rows  # noqa: E402


class RuntimeSignificanceAnalysisTests(unittest.TestCase):
    def test_run_metrics_report_significant_completion_gap(self):
        records = []
        for seed in range(4):
            records.append(
                {
                    "baseline": "proposed",
                    "scenario": "load_burst",
                    "seed": seed,
                    "graph_completion_ratio": 1.0,
                    "deadline_satisfaction_ratio": 1.0,
                    "task_success_ratio": 1.0,
                    "avg_graph_finish_time": 5.0,
                }
            )
            records.append(
                {
                    "baseline": "nearest_edge",
                    "scenario": "load_burst",
                    "seed": seed,
                    "graph_completion_ratio": 0.0,
                    "deadline_satisfaction_ratio": 0.0,
                    "task_success_ratio": 1.0,
                    "avg_graph_finish_time": 0.0,
                }
            )

        rows = significance_rows([], records)
        completion = next(
            row
            for row in rows
            if row["metric"] == "graph_completion_ratio" and row["baseline_b"] == "nearest_edge"
        )

        self.assertEqual(completion["significant"], "True")
        self.assertEqual(float(completion["p_value"]), 0.0)


if __name__ == "__main__":
    unittest.main()
