import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.lasdm.model import LASDMQoS, LASDMServiceChain, LASDMSFCNode
from airfogsim.lasdm.task_adapter import (
    LASDM_TO_TASK_FIELD_MAPPING,
    LASDMTaskAdapter,
    propagate_parent_failure,
)


def make_chain():
    return LASDMServiceChain(
        sfc_id="sfc_adapter",
        source_node_id="UAV_0",
        sink_node_id="CLOUD_0",
        payload_semantic="video",
        payload_mb=8.0,
        qos=LASDMQoS(deadline_s=5.0, priority=3.0),
        nodes={
            "pre": LASDMSFCNode(
                node_id="pre",
                service_type="preprocess",
                required_capabilities=("video_preprocess",),
                input_semantic="video",
                output_semantic="frames",
                cpu_mb=2.0,
                metadata={"output_ratio": 0.5},
            ),
            "det": LASDMSFCNode(
                node_id="det",
                service_type="detect",
                required_capabilities=("object_detection",),
                input_semantic="frames",
                output_semantic="objects",
                cpu_mb=3.0,
                metadata={"output_payload_mb": 1.0},
            ),
            "store": LASDMSFCNode(
                node_id="store",
                service_type="store",
                required_capabilities=("storage",),
                input_semantic="objects",
                output_semantic="receipt",
                cpu_mb=1.0,
            ),
        },
        edges=[("pre", "det"), ("det", "store")],
    )


class LASDMTaskAdapterTests(unittest.TestCase):
    def test_root_task_maps_lasdm_fields_to_airfogsim_task(self):
        adapter = LASDMTaskAdapter()
        task = adapter.build_task(make_chain(), "pre", current_time=10.0)

        self.assertEqual(task.getTaskId(), "sfc_adapter::pre")
        self.assertEqual(task.getTaskNodeId(), "UAV_0")
        self.assertEqual(task.getTaskCPU(), 2.0)
        self.assertEqual(task.getTaskSize(), 8.0)
        self.assertEqual(task.getTaskDeadline(), 5.0)
        self.assertEqual(task.getTaskPriority(), 3.0)
        self.assertEqual(task.getReturnedSize(), 0.0)
        self.assertIsNone(task.getAssignedTo())
        self.assertEqual(task._lasdm["field_mapping"], LASDM_TO_TASK_FIELD_MAPPING)
        self.assertEqual(task._lasdm["lifecycle"]["role"], "root")
        self.assertTrue(task._lasdm["lifecycle"]["is_root"])
        self.assertFalse(task._lasdm["lifecycle"]["is_sink"])
        self.assertEqual(task._lasdm["data_flow"]["successors"], ["det"])
        self.assertEqual(task._lasdm["data_flow"]["input_semantic"], "video")
        self.assertEqual(task._lasdm["data_flow"]["output_semantic"], "frames")
        self.assertEqual(task._lasdm["data_flow"]["output_payload_mb"], 4.0)

    def test_child_and_sink_tasks_keep_lineage_and_return_metadata(self):
        adapter = LASDMTaskAdapter()
        chain = make_chain()
        chain.submit_time = 10.0
        child = adapter.build_task(
            chain,
            "det",
            current_time=11.0,
            input_payload_mb=4.0,
            origin_node_id="RSU_0",
            parent_task_ids=["sfc_adapter::pre"],
        )
        sink = adapter.build_task(
            chain,
            "store",
            current_time=12.0,
            input_payload_mb=1.0,
            origin_node_id="CLOUD_0",
            parent_task_ids=["sfc_adapter::det"],
        )

        self.assertEqual(child.getTaskNodeId(), "RSU_0")
        self.assertAlmostEqual(child.getTaskDeadline(), 4.0)
        self.assertAlmostEqual(sink.getTaskDeadline(), 3.0)
        self.assertEqual(child._lasdm["lifecycle"]["role"], "child")
        self.assertEqual(child._lasdm["data_flow"]["predecessors"], ["pre"])
        self.assertEqual(child._lasdm["data_flow"]["parent_task_ids"], ["sfc_adapter::pre"])
        self.assertEqual(child._lasdm["data_flow"]["output_payload_mb"], 1.0)
        self.assertEqual(sink._lasdm["lifecycle"]["role"], "sink")
        self.assertEqual(sink.getReturnedSize(), 1.0)
        self.assertEqual(sink.getToReturnNodeId(), "CLOUD_0")
        self.assertTrue(sink._lasdm["lifecycle"]["is_child"])
        self.assertTrue(sink._lasdm["lifecycle"]["is_sink"])

    def test_real_airfogsim_task_fields_include_payload_cpu_deadline_and_return(self):
        chain = LASDMServiceChain(
            sfc_id="sfc_single",
            source_node_id="VEH_0",
            sink_node_id="CLOUD_1",
            payload_semantic="frames",
            payload_mb=6.0,
            qos=LASDMQoS(deadline_s=9.0, priority=2.0),
            nodes={
                "infer": LASDMSFCNode(
                    node_id="infer",
                    service_type="inference",
                    required_capabilities=("object_detection",),
                    input_semantic="frames",
                    output_semantic="objects",
                    cpu_mb=1.0,
                    metadata={"cpu_per_mb": 2.5, "output_ratio": 0.25},
                    qos_override=LASDMQoS(deadline_s=4.0, priority=5.0),
                ),
            },
            edges=[],
        )

        task = LASDMTaskAdapter().build_task(chain, "infer", current_time=7.0)

        self.assertEqual(task.getTaskNodeId(), "VEH_0")
        self.assertEqual(task.getTaskSize(), 6.0)
        self.assertEqual(task.getTaskCPU(), 15.0)
        self.assertEqual(task.getTaskDeadline(), 4.0)
        self.assertEqual(task.getTaskPriority(), 5.0)
        self.assertEqual(task.getReturnedSize(), 1.5)
        self.assertEqual(task.getToReturnNodeId(), "CLOUD_1")
        self.assertEqual(task._lasdm["lifecycle"]["role"], "root_sink")
        self.assertEqual(task._lasdm["task_fields"]["task_size"], 6.0)
        self.assertEqual(task._lasdm["task_fields"]["task_cpu"], 15.0)
        self.assertEqual(task._lasdm["data_flow"]["required_returned_size"], 1.5)
        self.assertTrue(task._lasdm_is_root)
        self.assertFalse(task._lasdm_is_child)
        self.assertTrue(task._lasdm_is_sink)

    def test_parent_failure_propagates_to_descendants(self):
        adapter = LASDMTaskAdapter()
        chain = make_chain()
        pre = adapter.build_task(chain, "pre", current_time=0.0)
        det = adapter.build_task(
            chain,
            "det",
            current_time=1.0,
            input_payload_mb=4.0,
            parent_task_ids=[pre.getTaskId()],
        )
        store = adapter.build_task(
            chain,
            "store",
            current_time=2.0,
            input_payload_mb=1.0,
            parent_task_ids=[det.getTaskId()],
        )

        affected = propagate_parent_failure([pre, det, store], pre)

        self.assertEqual([task.getTaskId() for task in affected], [det.getTaskId(), store.getTaskId()])
        self.assertEqual(det.getTaskFailureReason(), "Task fails due to parent task failed.")
        self.assertEqual(store.getTaskFailureReason(), "Task fails due to parent task failed.")
        self.assertEqual(det._lasdm["lifecycle"]["state"], "blocked_parent_failed")
        self.assertTrue(store._lasdm_parent_failed)

    def test_parent_failure_can_start_from_sfc_node_id(self):
        adapter = LASDMTaskAdapter()
        chain = make_chain()
        det = adapter.build_task(chain, "det", current_time=1.0, input_payload_mb=4.0)
        store = adapter.build_task(
            chain,
            "store",
            current_time=2.0,
            input_payload_mb=1.0,
            parent_task_ids=[det.getTaskId()],
        )

        affected = propagate_parent_failure([det, store], "pre")

        self.assertEqual([task.getTaskId() for task in affected], [det.getTaskId(), store.getTaskId()])
        self.assertEqual(det._lasdm_failed_parent_sfc_node_ids, ["pre"])
        self.assertTrue(store._lasdm_parent_failed)


if __name__ == "__main__":
    unittest.main()
