import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.enum_const import EnumerateConstants
from airfogsim.entities.task import Task
from airfogsim.lasdm.failure_mapping import (
    build_failure_event_payload,
    build_failure_event_record,
    map_airfogsim_task_failure,
)
from airfogsim.lasdm.model import GraphStatus, SFCFailureReason


class LASDMFailureMappingTests(unittest.TestCase):
    def test_maps_airfogsim_failure_codes_to_sfc_reasons(self):
        cases = {
            EnumerateConstants.TASK_FAIL_OUT_OF_DDL: SFCFailureReason.DEADLINE_MISSED,
            EnumerateConstants.TASK_FAIL_OUT_OF_TTI: SFCFailureReason.DEADLINE_MISSED,
            EnumerateConstants.TASK_FAIL_OUT_OF_NODE: SFCFailureReason.NO_CANDIDATE,
            EnumerateConstants.TASK_FAIL_PARENT_FAILED: SFCFailureReason.TASK_FAILED,
            EnumerateConstants.TASK_FAIL_MALICIOUS_RESULT: SFCFailureReason.RELIABILITY_VIOLATION,
        }
        for code, expected in cases.items():
            with self.subTest(code=code):
                self.assertEqual(map_airfogsim_task_failure(failure_code=code), expected)

    def test_all_native_task_failure_codes_map_to_a_lasdm_reason(self):
        for name, code in vars(EnumerateConstants).items():
            if not name.startswith("TASK_FAIL_"):
                continue
            with self.subTest(name=name, code=code):
                self.assertIsInstance(map_airfogsim_task_failure(failure_code=code), SFCFailureReason)

    def test_maps_task_failure_code_and_ignores_default_non_failure_code(self):
        task = Task("task_1", "UAV_0", 1.0, 0.5, 2.0, 1.0, 0.0)
        self.assertIsNone(map_airfogsim_task_failure(task=task))

        task.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_DDL)
        self.assertEqual(map_airfogsim_task_failure(task=task), SFCFailureReason.DEADLINE_MISSED)

    def test_maps_status_and_error_strings(self):
        self.assertEqual(map_airfogsim_task_failure(status=GraphStatus.TIMED_OUT), SFCFailureReason.DEADLINE_MISSED)
        self.assertIsNone(map_airfogsim_task_failure(status="finished"))
        self.assertEqual(map_airfogsim_task_failure(status="accuracy violation"), SFCFailureReason.ACCURACY_VIOLATION)
        self.assertEqual(map_airfogsim_task_failure(error="resource unavailable"), SFCFailureReason.NO_CANDIDATE)
        self.assertEqual(map_airfogsim_task_failure(error=RuntimeError("compute exception")), SFCFailureReason.TASK_FAILED)

    def test_native_airfogsim_lifecycle_statuses_are_not_failures(self):
        statuses = (
            "to_generate",
            "to_offload",
            "offloading",
            "computing",
            "computed",
            "to_return",
            "returning",
            "finished",
            GraphStatus.PENDING,
            GraphStatus.SUCCEEDED,
        )
        for status in statuses:
            with self.subTest(status=status):
                self.assertIsNone(map_airfogsim_task_failure(status=status))

    def test_native_airfogsim_failure_descriptions_map_to_lasdm_reasons(self):
        cases = {
            "Task fails due to out of deadline.": SFCFailureReason.DEADLINE_MISSED,
            "Task fails due to transmission timeout.": SFCFailureReason.DEADLINE_MISSED,
            "Task fails due to out of node.": SFCFailureReason.NO_CANDIDATE,
            "Task fails due to parent task failed.": SFCFailureReason.TASK_FAILED,
            "Task fails due to malicious result detected.": SFCFailureReason.RELIABILITY_VIOLATION,
        }
        for status, expected in cases.items():
            with self.subTest(status=status):
                self.assertEqual(map_airfogsim_task_failure(status=status), expected)

    def test_builds_standard_failure_event_payload_from_task(self):
        task = Task("task_2", "UAV_0", 1.0, 0.5, 2.0, 1.0, 0.0)
        task.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_TTI)

        payload = build_failure_event_payload(
            task=task,
            status="out_of_tti",
            sfc_node_id="detect",
            service_type="object_detection",
            details={"phase": "offload"},
        )

        self.assertEqual(payload["reason"], SFCFailureReason.DEADLINE_MISSED.value)
        self.assertEqual(payload["status"], GraphStatus.FAILED.value)
        self.assertEqual(payload["task_id"], "task_2")
        self.assertEqual(payload["task_node_id"], "UAV_0")
        self.assertEqual(payload["sfc_node_id"], "detect")
        self.assertEqual(payload["service_type"], "object_detection")
        self.assertEqual(payload["airfogsim_failure_code"], EnumerateConstants.TASK_FAIL_OUT_OF_TTI)
        self.assertIn("transmission timeout", payload["airfogsim_failure_reason"])
        self.assertEqual(payload["details"], {"phase": "offload"})

    def test_builds_event_record_compatible_with_lasdm_event_fields(self):
        record = build_failure_event_record(
            sfc_id="sfc_1",
            time_s=3.5,
            failure_code=EnumerateConstants.TASK_FAIL_MALICIOUS_RESULT,
        )

        self.assertEqual(record["event_type"], "failed")
        self.assertEqual(record["entity_type"], "sfc")
        self.assertEqual(record["entity_id"], "sfc_1")
        self.assertEqual(record["time_s"], 3.5)
        self.assertEqual(record["payload"]["reason"], SFCFailureReason.RELIABILITY_VIOLATION.value)


if __name__ == "__main__":
    unittest.main()
