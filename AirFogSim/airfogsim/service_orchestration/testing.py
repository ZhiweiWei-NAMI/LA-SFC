from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional

from airfogsim.airfogsim_scheduler import AirFogSimScheduler
from airfogsim.entities.cloud_server import CloudServer
from airfogsim.entities.rsu import RSU
from airfogsim.entities.uav import UAV
from airfogsim.entities.vehicle import Vehicle
from airfogsim.manager.task_manager import TaskManager
from airfogsim.manager.wired_manager import WiredNetworkManager


class FakeChannelManager:
    def __init__(self, n_rb: int = 8, rb_bandwidth: float = 10.0):
        self.n_RB = n_rb
        self.RB_bandwidth = rb_bandwidth

    def getCSI(self, transmitter_idx, receiver_idx, transmitter_type, receiver_type):
        return 1.0

    def getSignalPowerByType(self, transmitter_type, receiver_type, is_dBm=True):
        return 30.0

    def getNoisePower(self, is_dBm=True):
        return 1.0

    def getRateByChannelType(self, transmitter_idx, receiver_idx, channel_type, allocated_rbs=None):
        allocated_rbs = allocated_rbs or [0]
        return [10.0 for _ in allocated_rbs]


class FakeTrafficManager:
    def __init__(self, env):
        self.env = env

    def getRSUInfos(self):
        return {node_id: {"id": node_id, "position": node.getPosition()} for node_id, node in self.env.RSUs.items()}

    def getNodePositionById(self, node_id):
        node = self.env._getNodeById(node_id)
        return node.getPosition() if node is not None else None

    def getUAVTrafficInfos(self):
        return {node_id: {"id": node_id, "position": node.getPosition()} for node_id, node in self.env.UAVs.items()}

    def getConfig(self, name):
        if name == "UAV_speed_range":
            return [10, 20]
        return None


class SmokeEnv:
    def __init__(self, multi_region: bool = False, include_cloud: bool = True):
        vehicle_fog = {"cpu": 2, "memory": 256, "storage": 256}
        uav_fog = {"cpu": 4, "memory": 2048, "storage": 2048}
        rsu_fog = {"cpu": 20, "memory": 8192, "storage": 8192}
        cloud_fog = {"cpu": 100, "memory": 65536, "storage": 65536}

        self.vehicles = {
            "Vehicle_0": Vehicle("Vehicle_0", (0, 0, 0), task_profile={"lambda": 0.0}, fog_profile=vehicle_fog),
        }
        self.UAVs = {
            "UAV_0": UAV("UAV_0", (10, 0, 100), task_profile={"lambda": 0.0}, fog_profile=uav_fog),
        }
        self.RSUs = {
            "RSU_0": RSU("RSU_0", (20, 0, 0), task_profile={}, fog_profile=rsu_fog),
        }
        self.cloudServers = {}

        if multi_region:
            self.UAVs["UAV_1"] = UAV("UAV_1", (220, 0, 100), task_profile={"lambda": 0.0}, fog_profile=uav_fog)
            self.RSUs["RSU_1"] = RSU(
                "RSU_1",
                (240, 0, 0),
                task_profile={},
                fog_profile={"cpu": 4, "memory": 256, "storage": 256},
            )
        if include_cloud:
            self.cloudServers["cloudServer_0"] = CloudServer(
                "cloudServer_0",
                (0, 0, 0),
                task_profile={},
                fog_profile=cloud_fog,
            )

        self.vehicle_ids_as_index = list(self.vehicles.keys())
        self.uav_ids_as_index = list(self.UAVs.keys())
        self.rsu_ids_as_index = list(self.RSUs.keys())
        self.cloud_server_ids_as_index = list(self.cloudServers.keys())
        self.task_node_ids = list(self.UAVs.keys()) + list(self.vehicles.keys())
        self.task_manager = TaskManager(
            {
                "task_generation_model": "None",
                "tti_threshold": 1.0,
                "hard_ddl": 10.0,
                "task_min_cpu": 0.01,
                "task_max_cpu": 10.0,
                "task_min_size": 0.01,
                "task_max_size": 50.0,
                "task_min_required_returned_size": 0.0,
                "task_max_required_returned_size": 50.0,
                "task_min_deadline": 0.1,
                "task_max_deadline": 10.0,
                "task_min_priority": 0.1,
                "task_max_priority": 1.0,
                "cpu_model": "Uniform",
                "cpu_kwargs": {"low": 0.1, "high": 0.2},
                "size_model": "Uniform",
                "size_kwargs": {"low": 0.1, "high": 0.2},
                "deadline_model": "Uniform",
                "deadline_kwargs": {"low": 1.0, "high": 2.0},
                "priority_model": "Uniform",
                "priority_kwargs": {"low": 0.5, "high": 1.0},
                "required_returned_size_model": "Uniform",
                "required_returned_size_kwargs": {"low": 0.0, "high": 0.0},
            }
        )
        self.task_scheduler = AirFogSimScheduler.getTaskScheduler()
        self.channel_manager = FakeChannelManager()
        self.traffic_manager = FakeTrafficManager(self)
        edges = [{"u": "RSU_0", "v": "cloudServer_0", "capacity_mbps": 1000, "prop_ms": 1.0}]
        if multi_region:
            edges.append({"u": "RSU_1", "v": "RSU_0", "capacity_mbps": 500, "prop_ms": 1.0})
        self.wired_manager = WiredNetworkManager({"edges": edges if include_cloud else []})
        self.simulation_time = 0.0
        self.traffic_interval = 0.1
        self.simulation_interval = 0.1
        self.task_return_routes = {}
        self.activated_offloading_tasks_with_RB_Nos = {}
        self.uav_mobility_patterns = {}

        for node in [*self.vehicles.values(), *self.UAVs.values(), *self.RSUs.values(), *self.cloudServers.values()]:
            node.trust_score = 1.0
        self.vehicles["Vehicle_0"].trust_score = 0.4

    def _getNodeById(self, node_id):
        return (
            self.vehicles.get(node_id)
            or self.UAVs.get(node_id)
            or self.RSUs.get(node_id)
            or self.cloudServers.get(node_id)
        )

    def _getNodeTypeById(self, node_id):
        if node_id in self.vehicles:
            return "V"
        if node_id in self.UAVs:
            return "U"
        if node_id in self.RSUs:
            return "I"
        if node_id in self.cloudServers:
            return "C"
        return None

    def _getNodeIdxById(self, node_id):
        if node_id in self.vehicles:
            return self.vehicle_ids_as_index.index(node_id)
        if node_id in self.UAVs:
            return self.uav_ids_as_index.index(node_id)
        if node_id in self.RSUs:
            return self.rsu_ids_as_index.index(node_id)
        if node_id in self.cloudServers:
            return self.cloud_server_ids_as_index.index(node_id)
        return -1

    def getNodeTrustScore(self, node_id):
        node = self._getNodeById(node_id)
        return getattr(node, "trust_score", 1.0)

    def getUAVIds(self):
        return list(self.UAVs.keys())

    def getRSUIds(self):
        return list(self.RSUs.keys())

    def getCloudServerIds(self):
        return list(self.cloudServers.keys())


def build_smoke_env(multi_region: bool = False, include_cloud: bool = True) -> SmokeEnv:
    return SmokeEnv(multi_region=multi_region, include_cloud=include_cloud)


def force_finish_all_active_tasks(env, assignment_by_microservice: Optional[Dict[str, str]] = None) -> int:
    assignment_by_microservice = assignment_by_microservice or {}
    queues = [
        env.task_manager._waiting_to_offload_tasks,
        env.task_manager._offloading_tasks,
        env.task_manager._computing_tasks,
        env.task_manager._waiting_to_return_tasks,
        env.task_manager._returning_tasks,
    ]
    tasks = []
    seen = set()
    for queue in queues:
        for node_id, node_tasks in queue.items():
            for task in list(node_tasks):
                if task.getTaskId() in seen:
                    continue
                seen.add(task.getTaskId())
                tasks.append((queue, node_id, task))

    for queue, node_id, task in tasks:
        queue[node_id].remove(task)
        ms_id = getattr(task, "_microservice_id", None)
        assigned_node = assignment_by_microservice.get(ms_id, task.getAssignedTo() or task.getTaskNodeId())
        task.setAssignedTo(assigned_node)
        if task.getReturnedSize() <= 0:
            task.setAttribute("_to_return_node_id", assigned_node)
        env.task_manager._done_tasks.setdefault(task.getTaskNodeId(), []).append(task)
    return len(tasks)


def advance_algorithm_until_complete(algorithm, env, max_rounds: int = 10) -> bool:
    for _ in range(max_rounds):
        algorithm.scheduleStep(env)
        force_finish_all_active_tasks(env)
        env.simulation_time += env.simulation_interval
        algorithm.runtime.on_airfogsim_step_finished(env)
        if algorithm.runtime.running_graphs and all(
            algorithm.runtime.is_graph_finished(graph_id)
            for graph_id in algorithm.runtime.running_graphs
        ):
            return True
    return False
