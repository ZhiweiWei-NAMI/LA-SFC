import csv
import os
import sys
import tempfile
import unittest


TEST_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(TEST_DIR, "../.."))
PLOTTING_ROOT = os.path.join(WORKSPACE_ROOT, "experiment_artifacts", "plotting")
if PLOTTING_ROOT not in sys.path:
    sys.path.insert(0, PLOTTING_ROOT)

from compare_lasdm_agentic import (  # noqa: E402
    COMPARISON_FIELDS,
    CONTINUITY_FIELDS,
    build_comparison_rows,
    build_continuity_rows,
    write_csv,
)


def write_summary(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LASDMAgenticAlignmentTests(unittest.TestCase):
    def test_continuity_rows_normalize_lasdm_and_agentic_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lasdm_summary = os.path.join(tmpdir, "lasdm", "summary.csv")
            agentic_summary = os.path.join(tmpdir, "agentic", "summary.csv")
            write_summary(
                lasdm_summary,
                [
                    "baseline",
                    "scenario",
                    "seed",
                    "mode",
                    "status",
                    "graph_submitted_count",
                    "graph_complete_count",
                    "graph_failed_count",
                    "graph_timeout_count",
                    "deadline_satisfaction_ratio",
                    "task_success_ratio",
                    "avg_graph_finish_time",
                    "payload_tx_mb",
                ],
                [
                    {
                        "baseline": "proposed",
                        "scenario": "normal",
                        "seed": 0,
                        "mode": "offline",
                        "status": "succeeded",
                        "graph_submitted_count": 2,
                        "graph_complete_count": 1,
                        "graph_failed_count": 1,
                        "graph_timeout_count": 0,
                        "deadline_satisfaction_ratio": 0.5,
                        "task_success_ratio": 1.0,
                        "avg_graph_finish_time": 3.0,
                        "payload_tx_mb": 2.0,
                    }
                ],
            )
            write_summary(
                agentic_summary,
                [
                    "baseline",
                    "seed",
                    "mode",
                    "completed",
                    "graph_submit_count",
                    "graph_complete_count",
                    "deadline_satisfaction_ratio",
                    "task_success_ratio",
                    "avg_graph_finish_time",
                    "payload_tx_mb",
                ],
                [
                    {
                        "baseline": "proposed",
                        "seed": 0,
                        "mode": "smoke",
                        "completed": True,
                        "graph_submit_count": 2,
                        "graph_complete_count": 2,
                        "deadline_satisfaction_ratio": 1.0,
                        "task_success_ratio": 1.0,
                        "avg_graph_finish_time": 1.0,
                        "payload_tx_mb": 4.0,
                    }
                ],
            )

            continuity_rows = build_continuity_rows(lasdm_summary, agentic_summary)
            self.assertEqual(len(continuity_rows), 2)
            lasdm_row = next(row for row in continuity_rows if row["source"] == "lasdm")
            self.assertEqual(lasdm_row["scenario"], "normal")
            self.assertAlmostEqual(lasdm_row["graph_completion_ratio"], 0.5)
            self.assertAlmostEqual(lasdm_row["failure_terminal_ratio"], 0.5)
            self.assertAlmostEqual(lasdm_row["continuity_score"], 0.25)

            comparison_rows = build_comparison_rows(continuity_rows)
            by_metric = {row["metric"]: row for row in comparison_rows}
            self.assertEqual(by_metric["continuity_score"]["better_source"], "agentic")
            self.assertEqual(by_metric["payload_tx_mb"]["better_source"], "lasdm")
            self.assertEqual(
                by_metric["continuity_score"]["alignment_status"],
                "shared_seed_descriptive_scenario_mismatch",
            )

    def test_writers_emit_stable_headers_for_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            continuity_path = os.path.join(tmpdir, "continuity_metrics.csv")
            comparison_path = os.path.join(tmpdir, "comparison_lasdm_vs_agentic.csv")
            write_csv(continuity_path, [], CONTINUITY_FIELDS)
            write_csv(comparison_path, [], COMPARISON_FIELDS)
            with open(continuity_path, "r", encoding="utf-8") as file:
                self.assertTrue(file.readline().startswith("source,suite_dir,baseline"))
            with open(comparison_path, "r", encoding="utf-8") as file:
                self.assertTrue(file.readline().startswith("baseline,scenario,metric"))


if __name__ == "__main__":
    unittest.main()
