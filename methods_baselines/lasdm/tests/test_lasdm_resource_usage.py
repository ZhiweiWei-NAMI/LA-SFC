import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
PLOTTING_ROOT = os.path.join(WORKSPACE_ROOT, "experiment_artifacts", "plotting")
for path in (AIRFOGSIM_ROOT, PLOTTING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.resource_usage import collect_resource_usage_snapshot  # noqa: E402


class FakeNode:
    def __init__(self, cpu):
        self.cpu = cpu

    def getFogProfile(self):
        return {"cpu": self.cpu}


class FakeTask:
    def __init__(self, task_id, cpu, computed=0.0):
        self.task_id = task_id
        self.cpu = cpu
        self.computed = computed

    def getTaskId(self):
        return self.task_id

    def getTaskCPU(self):
        return self.cpu

    def getComputedSize(self):
        return self.computed


class FakeTaskManager:
    def __init__(self):
        self.computing = {"RSU_0": [FakeTask("task_1", 10.0, 2.0)]}

    def getComputingTasks(self):
        return self.computing

    def getWaitingToOffloadTasks(self):
        return {"UAV_0": [FakeTask("task_2", 1.0)]}

    def getOffloadingTasks(self):
        return {"UAV_0": [FakeTask("task_3", 1.0)]}

    def getDoneTasks(self):
        return [FakeTask("task_4", 1.0)]

    def getOutOfDDLTasks(self):
        return []


class FakeChannelManager:
    n_RB = 8


class FakeWiredManager:
    last_step_transmitted_bytes = 1024.0

    def __init__(self):
        self._queues = {("RSU_0", "cloudServer_0"): 2048.0}
        self._links = {("RSU_0", "cloudServer_0"): {"capacity_mbps": 100.0}}
        self._flows = {"task_5": {"src": "RSU_0", "dst": "cloudServer_0"}}


class FakeStorageManager:
    def getCacheState(self, node_id):
        if node_id == "RSU_0":
            return {"capacity": 4096, "used": 1024, "items": 2, "hit_ratio": 0.5}
        return {"capacity": 0, "used": 0, "items": 0, "hit_ratio": 0.0}


class FakeEnergyManager:
    _UAVs_energy_info = {"UAV_0": {"energy": 93.0}}


class FakeEnv:
    simulation_time = 7.0
    simulation_interval = 1.0
    channel = {"data_size": 12.0}
    V2I_channel = {"data_size": 8.0}
    V2U_channel = {"data_size": 3.0}
    U2I_channel = {"data_size": 1.0}

    def __init__(self):
        self.nodes = {
            "UAV_0": FakeNode(4.0),
            "RSU_0": FakeNode(20.0),
            "cloudServer_0": FakeNode(100.0),
        }
        self.task_manager = FakeTaskManager()
        self.channel_manager = FakeChannelManager()
        self.wired_manager = FakeWiredManager()
        self.storage_manager = FakeStorageManager()
        self.energy_manager = FakeEnergyManager()
        self.activated_offloading_tasks_with_RB_Nos = {"task_3": [0, 1, 2]}
        self.alloc_cpu_callback = lambda computing, **kwargs: {"task_1": 5.0}

    def getUAVIds(self):
        return ["UAV_0"]

    def getRSUIds(self):
        return ["RSU_0"]

    def getVehicleIds(self):
        return []

    def getCloudServerIds(self):
        return ["cloudServer_0"]

    def _getNodeById(self, node_id):
        return self.nodes.get(node_id)


class LASDMResourceUsageTests(unittest.TestCase):
    def test_collect_resource_usage_snapshot_reads_airfogsim_managers(self):
        snapshot = collect_resource_usage_snapshot(
            FakeEnv(),
            decision_time_s=0.012,
            previous_energy_remaining=100.0,
        )

        self.assertEqual(snapshot["time_s"], 7.0)
        self.assertEqual(snapshot["decision_time_s"], 0.012)
        self.assertEqual(snapshot["cpu_allocated_by_node"]["RSU_0"], 5.0)
        self.assertEqual(snapshot["cpu_capacity_by_node"]["RSU_0"], 20.0)
        self.assertEqual(snapshot["rb_total"], 8)
        self.assertEqual(snapshot["rb_used_unique"], 3)
        self.assertEqual(snapshot["backhaul_usage_bytes"], 1024.0)
        self.assertEqual(snapshot["energy_consumed_step"], 7.0)
        self.assertEqual(snapshot["storage_used_bytes_total"], 1024.0)
        self.assertEqual(snapshot["waiting_to_offload_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
