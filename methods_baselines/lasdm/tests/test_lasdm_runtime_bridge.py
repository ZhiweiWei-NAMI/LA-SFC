import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.lasdm.instance_directory import ServiceInstance, ServiceInstanceDirectory
from airfogsim.lasdm.manager import LASDMManager
from airfogsim.lasdm.model import GraphStatus, LASDMQoS, LASDMServiceChain, LASDMSFCNode, SFCFailureReason
from airfogsim.lasdm.runtime_bridge import LASDMRuntimeBridge


class FakeTaskManager:
    def __init__(self):
        self._task_id = 0
        self.waiting = []
        self.done = []
        self.failed = []

    def registerGeneratedTask(self, task):
        task.setGenerated()
        self.waiting.append(task)
        return task

    def getDoneTasks(self):
        return list(self.done)

    def getRecentlyFailedTasks(self):
        return list(self.failed)

    def getOutOfDDLTasks(self):
        return []


class FakeEnv:
    def __init__(self):
        self.simulation_time = 0.0
        self.task_manager = FakeTaskManager()


def make_directory():
    directory = ServiceInstanceDirectory()
    directory.register(
        ServiceInstance(
            instance_id="inst_pre",
            service_id="preprocess",
            node_id="UAV_0",
            node_type="uav",
            region_id="RSU_0",
            capabilities=("video_preprocess",),
            input_semantic="video",
            output_semantic="keyframes",
            capacity={"cpu": 4.0, "memory": 256.0},
            max_concurrency=2,
            reliability_score=0.99,
            accuracy_score=0.99,
        )
    )
    directory.register(
        ServiceInstance(
            instance_id="inst_detect",
            service_id="detect",
            node_id="RSU_0",
            node_type="rsu",
            region_id="RSU_0",
            capabilities=("object_detection",),
            input_semantic="keyframes",
            output_semantic="detections",
            capacity={"cpu": 8.0, "memory": 1024.0},
            max_concurrency=2,
            reliability_score=0.99,
            accuracy_score=0.99,
        )
    )
    return directory


def make_chain():
    return LASDMServiceChain(
        sfc_id="sfc_bridge",
        source_node_id="UAV_0",
        sink_node_id="UAV_0",
        payload_semantic="video",
        payload_mb=5.0,
        qos=LASDMQoS(deadline_s=10.0, priority=3.0, reliability_min=0.9, accuracy_min=0.9),
        nodes={
            "pre": LASDMSFCNode(
                node_id="pre",
                service_type="preprocess",
                required_capabilities=("video_preprocess",),
                input_semantic="video",
                output_semantic="keyframes",
                cpu_mb=1.0,
                memory_mb=64.0,
            ),
            "det": LASDMSFCNode(
                node_id="det",
                service_type="detect",
                required_capabilities=("object_detection",),
                input_semantic="keyframes",
                output_semantic="detections",
                cpu_mb=2.0,
                memory_mb=128.0,
            ),
        },
        edges=[("pre", "det")],
        context={"preferred_region_id": "RSU_0"},
    )


class LASDMRuntimeBridgeTests(unittest.TestCase):
    def test_submit_create_and_apply_ready_function_task(self):
        env = FakeEnv()
        bridge = LASDMRuntimeBridge(LASDMManager(directory=make_directory()))
        chain = bridge.submit_service_chain(make_chain(), current_time=0.0)

        tasks = bridge.create_ready_function_tasks(env)
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        decision = bridge.manager.decisions[chain.sfc_id]
        applied = bridge.apply_orchestration_decision(decision, current_time=env.simulation_time)

        self.assertEqual(task.getTaskId(), "sfc_bridge::pre::1")
        self.assertEqual(task.getTaskNodeId(), "UAV_0")
        self.assertEqual(getattr(task, "_lasdm_sfc_id"), "sfc_bridge")
        self.assertEqual(getattr(task, "_lasdm_node_id"), "pre")
        self.assertEqual(applied[task.getTaskId()], "UAV_0")
        self.assertEqual(task.getAssignedTo(), "UAV_0")
        self.assertEqual(len(env.task_manager.waiting), 1)

    def test_sync_completed_parent_spawns_child_and_completes_sfc(self):
        env = FakeEnv()
        bridge = LASDMRuntimeBridge(LASDMManager(directory=make_directory()))
        bridge.submit_service_chain(make_chain(), current_time=0.0)

        pre_task = bridge.create_ready_function_tasks(env)[0]
        bridge.apply_orchestration_decision(bridge.manager.decisions["sfc_bridge"], current_time=0.0)
        env.simulation_time = 1.0
        env.task_manager.done.append(pre_task)
        synced = bridge.sync_from_airfogsim_tasks(env)
        child_tasks = bridge.create_ready_function_tasks(env)

        self.assertEqual(synced["completed_tasks"], 1)
        self.assertEqual(len(child_tasks), 1)
        det_task = child_tasks[0]
        self.assertEqual(getattr(det_task, "_lasdm_node_id"), "det")
        self.assertEqual(det_task.getTaskNodeId(), "UAV_0")
        self.assertEqual(det_task.getToReturnNodeId(), "UAV_0")

        bridge.apply_orchestration_decision(bridge.manager.decisions["sfc_bridge"], tasks=child_tasks, current_time=1.0)
        env.simulation_time = 2.0
        env.task_manager.done.append(det_task)
        bridge.sync_from_airfogsim_tasks(env)

        chain = bridge.manager.chains["sfc_bridge"]
        self.assertEqual(chain.status, GraphStatus.SUCCEEDED)
        metrics = bridge.collect_step_metrics(current_time=2.0)
        self.assertEqual(metrics["succeeded"], 1)
        self.assertEqual(metrics["runtime_bridge"]["completed_function_tasks"], 2)

    def test_sync_failed_task_marks_sfc_failed(self):
        env = FakeEnv()
        bridge = LASDMRuntimeBridge(LASDMManager(directory=make_directory()))
        bridge.submit_service_chain(make_chain(), current_time=0.0)
        task = bridge.create_ready_function_tasks(env)[0]

        env.simulation_time = 1.0
        env.task_manager.failed.append(task)
        synced = bridge.sync_from_airfogsim_tasks(env)

        chain = bridge.manager.chains["sfc_bridge"]
        self.assertEqual(synced["failed_tasks"], 1)
        self.assertEqual(chain.status, GraphStatus.FAILED)
        self.assertEqual(chain.failure_reason, SFCFailureReason.TASK_FAILED)


if __name__ == "__main__":
    unittest.main()
