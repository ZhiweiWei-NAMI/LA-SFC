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

from airfogsim.lasdm.event_logger import (
    LASDM_EVENT_SCHEMA_VERSION,
    LASDMEventLogger,
    LASDMFunctionMetrics,
    LASDMSFCMetrics,
    read_jsonl,
)
from airfogsim.lasdm.metrics import LASDMMetrics
from airfogsim.lasdm.model import GraphStatus, SFCFailureReason


class LASDMEventLoggerTests(unittest.TestCase):
    def test_logger_writes_standard_sfc_and_function_jsonl(self):
        logger = LASDMEventLogger()
        logger.record_sfc_event(
            "sfc_a",
            "succeeded",
            3.5,
            status=GraphStatus.SUCCEEDED,
            metrics=LASDMSFCMetrics(sfc_id="sfc_a", latency_s=1.5, qos_hit=True),
            payload={"baseline": "lasdm_greedy"},
        )
        logger.record_function_event(
            "sfc_a",
            "det",
            "function_completed",
            3.0,
            status="succeeded",
            metrics=LASDMFunctionMetrics(
                sfc_id="sfc_a",
                function_id="det",
                service_type="detect",
                instance_id="rsu_detect",
                node_id="RSU_0",
                processing_time_s=0.25,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.jsonl")
            logger.write_jsonl(path)
            rows = read_jsonl(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["schema_version"], LASDM_EVENT_SCHEMA_VERSION)
        self.assertEqual(rows[0]["entity_type"], "sfc")
        self.assertEqual(rows[0]["sfc_id"], "sfc_a")
        self.assertIsNone(rows[0]["function_id"])
        self.assertEqual(rows[0]["status"], "succeeded")
        self.assertEqual(rows[0]["metrics"]["latency_s"], 1.5)
        self.assertTrue(rows[0]["metrics"]["qos_hit"])
        self.assertEqual(rows[1]["entity_type"], "function_node")
        self.assertEqual(rows[1]["function_id"], "det")
        self.assertEqual(rows[1]["metrics"]["service_type"], "detect")

    def test_metrics_summary_is_unchanged_and_events_are_standardized(self):
        metrics = LASDMMetrics()
        metrics.record_submit("sfc_b", 0.0)
        metrics.record_success("sfc_b", submit_time=0.0, finish_time=2.0, qos_hit=False)
        before = metrics.summary()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics_events.jsonl")
            metrics.write_events_jsonl(path)
            with open(path, "r", encoding="utf-8") as file:
                rows = [json.loads(line) for line in file]

        self.assertEqual(before["submitted"], 1)
        self.assertEqual(before["succeeded"], 1)
        self.assertEqual(before["qos_hit_ratio"], 0.0)
        self.assertEqual(before, metrics.summary())
        self.assertEqual(rows[1]["schema_version"], LASDM_EVENT_SCHEMA_VERSION)
        self.assertEqual(rows[1]["metrics"]["latency_s"], 2.0)
        self.assertFalse(rows[1]["metrics"]["qos_hit"])

    def test_failure_events_expose_reason_and_terminal_status_metrics(self):
        metrics = LASDMMetrics()
        metrics.record_failure(
            "sfc_c",
            time_s=5.0,
            reason=SFCFailureReason.DEADLINE_MISSED,
            status=GraphStatus.TIMED_OUT,
            details={"node_id": "det"},
        )

        record = metrics.event_records()[0]
        self.assertEqual(record["status"], "timed_out")
        self.assertEqual(record["metrics"]["failure_reason"], "deadline_missed")
        self.assertEqual(record["payload"]["node_id"], "det")
        self.assertEqual(metrics.summary()["timed_out"], 1)


if __name__ == "__main__":
    unittest.main()
