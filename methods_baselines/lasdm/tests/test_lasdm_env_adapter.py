import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.entities.task import Task  # noqa: E402
from airfogsim.lasdm.env_adapter import LASDMEnvAdapter  # noqa: E402
from airfogsim.lasdm.instance_directory import ServiceInstance, ServiceInstanceDirectory, ServiceInstanceStatus  # noqa: E402
from airfogsim.lasdm.manager import LASDMManager  # noqa: E402
from airfogsim.lasdm.model import LASDMQoS, LASDMServiceChain, LASDMSFCNode  # noqa: E402
from airfogsim.lasdm.orchestrator import LASDMOrchestrator  # noqa: E402


class FakeNode:
    def __init__(self, node_id, node_type, cpu=10.0):
        self.node_id = node_id
        self.node_type = node_type
        self.profile = {"cpu": cpu, "memory": 1024.0, "storage": 2048.0}

    def getFogProfile(self):
        return dict(self.profile)

    def getPosition(self):
        return [0.0, 0.0, 0.0]

    def to_dict(self):
        return {
            "id": self.node_id,
            "node_type": self.node_type,
            "position": self.getPosition(),
            "fog_profile": self.getFogProfile(),
        }


class FakeTaskManager:
    def __init__(self):
        self._task_id = 0
        self.waiting = {}
        self.offloading = {}
        self.computing = {}
        self.done = []
        self.failed = []

    def registerGeneratedTask(self, task):
        self.waiting.setdefault(task.getTaskNodeId(), []).append(task)
        task.setGenerated()
        return task

    def offloadTask(self, task_node_id, task_id, target_node_id, current_time, route=None):
        for task in list(self.waiting.get(task_node_id, [])):
            if task.getTaskId() != task_id:
                continue
            task.offloadTo(target_node_id, route or [target_node_id], current_time)
            self.waiting[task_node_id].remove(task)
            if target_node_id == task_node_id:
                task.startToCompute(current_time)
                self.computing.setdefault(target_node_id, []).append(task)
            else:
                self.offloading.setdefault(task_node_id, []).append(task)
            return True
        return False

    def getWaitingToOffloadTasks(self):
        return self.waiting

    def getOffloadingTasks(self):
        return self.offloading

    def getComputingTasks(self):
        return self.computing

    def getDoneTasks(self):
        return self.done

    def getOutOfDDLTasks(self):
        return self.failed


class FakeTaskScheduler:
    def registerGeneratedTask(self, env, task):
        return env.task_manager.registerGeneratedTask(task)

    def setTaskOffloading(self, env, task_node_id, task_id, target_node_id, route=None):
        return env.task_manager.offloadTask(task_node_id, task_id, target_node_id, env.simulation_time, route)

    def getAllOffloadingTaskInfos(self, env):
        return [
            task.to_dict()
            for tasks in env.task_manager.getOffloadingTasks().values()
            for task in tasks
        ]

    def getWaitingToReturnTaskInfos(self, env):
        return {}

    def setTaskReturnRoute(self, env, task_id, route):
        env.task_return_routes[task_id] = route


class FakeCommunicationScheduler:
    def getNumberOfRB(self, env):
        return 4

    def setCommunicationWithRB(self, env, task_id, rb_nos):
        env.activated_offloading_tasks_with_RB_Nos[task_id] = rb_nos


class FakeComputationScheduler:
    def setComputingCallBack(self, env, callback):
        env.alloc_cpu_callback = callback


class FakeEnv:
    def __init__(self):
        self.simulation_time = 3.0
        self.UAVs = {"UAV_0": FakeNode("UAV_0", "U")}
        self.RSUs = {"RSU_0": FakeNode("RSU_0", "I", cpu=20.0)}
        self.vehicles = {}
        self.cloudServers = {"cloudServer_0": FakeNode("cloudServer_0", "C", cpu=100.0)}
        self.task_manager = FakeTaskManager()
        self.task_scheduler = FakeTaskScheduler()
        self.communication_scheduler = FakeCommunicationScheduler()
        self.computation_scheduler = FakeComputationScheduler()
        self.activated_offloading_tasks_with_RB_Nos = {}
        self.task_return_routes = {}
        self.simulation_interval = 1.0

    def step(self):
        for node_id, tasks in list(self.task_manager.computing.items()):
            for task in list(tasks):
                task.compute(task.getTaskCPU(), 1.0, self.simulation_time + 1.0)
                tasks.remove(task)
                self.task_manager.done.append(task)
        self.simulation_time += self.simulation_interval
        return self.isDone()

    def isDone(self):
        return self.simulation_time >= 20.0

    def close(self):
        return None

    def _getNodeById(self, node_id):
        return self.UAVs.get(node_id) or self.RSUs.get(node_id) or self.cloudServers.get(node_id)

    def _getNodeTypeById(self, node_id):
        if node_id in self.UAVs:
            return "U"
        if node_id in self.RSUs:
            return "I"
        if node_id in self.cloudServers:
            return "C"
        return None

    def getUAVIds(self):
        return list(self.UAVs)

    def getRSUIds(self):
        return list(self.RSUs)

    def getVehicleIds(self):
        return []

    def getCloudServerIds(self):
        return list(self.cloudServers)

    def isNodeAuthenticated(self, node_id):
        return node_id != "cloudServer_0"

    def getNodeTrustScore(self, node_id):
        return 0.4 if node_id == "cloudServer_0" else 0.95


def make_task():
    return Task(
        task_id="task_1",
        task_node_id="UAV_0",
        task_cpu=2.0,
        task_size=5.0,
        task_deadline=10.0,
        task_priority=1.0,
        task_arrival_time=0.0,
    )


class LASDMEnvAdapterTests(unittest.TestCase):
    def test_register_and_offload_use_airfogsim_scheduler_queues(self):
        env = FakeEnv()
        adapter = LASDMEnvAdapter()
        task = adapter.register_task(env, make_task())

        ok = adapter.schedule_task_offloading(env, task, "RSU_0", ["RSU_0"], current_time=env.simulation_time)

        self.assertTrue(ok)
        self.assertEqual(env.task_manager.waiting["UAV_0"], [])
        self.assertEqual(env.task_manager.offloading["UAV_0"][0].getTaskId(), "task_1")
        self.assertEqual(task.getAssignedTo(), "RSU_0")

    def test_scheduler_helpers_install_rb_and_cpu_callback(self):
        env = FakeEnv()
        adapter = LASDMEnvAdapter()
        task = adapter.register_task(env, make_task())
        adapter.schedule_task_offloading(env, task, "RSU_0", ["RSU_0"], current_time=env.simulation_time)

        rb = adapter.schedule_communication(env)
        adapter.schedule_computation(env)

        self.assertIn("task_1", rb)
        self.assertEqual(env.activated_offloading_tasks_with_RB_Nos["task_1"], rb["task_1"])
        self.assertTrue(callable(env.alloc_cpu_callback))

    def test_sync_service_instances_updates_load_capacity_and_availability(self):
        directory = ServiceInstanceDirectory()
        directory.register(
            ServiceInstance(
                instance_id="detect_0",
                service_id="detect",
                node_id="RSU_0",
                node_type="rsu",
                region_id="RSU_0",
                capabilities=("detect",),
                capacity={"cpu": 1.0},
            )
        )
        directory.register(
            ServiceInstance(
                instance_id="cloud_0",
                service_id="verify",
                node_id="cloudServer_0",
                node_type="cloud_server",
                region_id="cloud",
                capabilities=("verify",),
            )
        )
        env = FakeEnv()
        env.task_manager.computing["RSU_0"] = [make_task()]
        adapter = LASDMEnvAdapter(directory=directory)

        snapshot = adapter.sync_from_env(env)

        detect = directory.get("detect_0")
        cloud = directory.get("cloud_0")
        self.assertEqual(detect.current_load, 1)
        self.assertEqual(detect.capacity["cpu"], 20.0)
        self.assertEqual(cloud.status, ServiceInstanceStatus.DEGRADED)
        self.assertEqual(snapshot["instances"]["cloud_0"]["trust_score"], 0.4)

    def test_run_benchmark_uses_schedule_step_and_env_step_loop(self):
        directory = ServiceInstanceDirectory()
        directory.register(
            ServiceInstance(
                instance_id="pre_0",
                service_id="video_preprocess",
                node_id="UAV_0",
                node_type="uav",
                region_id="RSU_0",
                capabilities=("video_preprocess",),
                input_semantic="video",
                output_semantic="keyframes",
                capacity={"cpu": 8.0, "memory": 1024.0},
            )
        )
        manager = LASDMManager(
            directory=directory,
            orchestrator=LASDMOrchestrator(directory),
        )
        chain = LASDMServiceChain(
            sfc_id="runtime_sfc",
            source_node_id="UAV_0",
            sink_node_id="UAV_0",
            payload_semantic="video",
            payload_mb=1.0,
            qos=LASDMQoS(deadline_s=5.0),
            nodes={
                "pre": LASDMSFCNode(
                    node_id="pre",
                    service_type="video_preprocess",
                    required_capabilities=("video_preprocess",),
                    input_semantic="video",
                    output_semantic="keyframes",
                    cpu_mb=1.0,
                    metadata={"required_returned_size": 0.0},
                )
            },
            edges=[],
        )
        env = FakeEnv()
        adapter = LASDMEnvAdapter(
            directory=directory,
            manager=manager,
            chains=[chain],
            runtime_config={"max_steps": 3},
            env_factory=lambda: env,
        )

        result = adapter.run_benchmark()

        self.assertTrue(result["completed"], msg=result)
        self.assertEqual(result["raw_metrics"]["graph_submitted_count"], 1)
        self.assertEqual(result["raw_metrics"]["graph_complete_count"], 1)
        self.assertEqual(result["raw_metrics"]["task_done_num"], 1)
        self.assertEqual(result["raw_metrics"]["task_success_ratio"], 1.0)
        self.assertEqual(result["raw_metrics"]["decision_samples"], 1)
        self.assertIn("runtime_overhead", result)
        self.assertIn("resource_usage_timeseries", result)
        self.assertEqual(len(result["resource_usage_timeseries"]), 1)
        self.assertIn("cpu_utilization_by_node", result["resource_usage_timeseries"][0])
        self.assertEqual(manager.chains["runtime_sfc"].status.value, "succeeded")


if __name__ == "__main__":
    unittest.main()
