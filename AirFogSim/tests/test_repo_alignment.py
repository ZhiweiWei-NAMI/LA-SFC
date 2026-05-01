import unittest

from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.airfogsim_scheduler import AirFogSimScheduler
from airfogsim.entities.cloud_server import CloudServer
from airfogsim.entities.task import Task
from airfogsim.entities.rsu import RSU
from airfogsim.manager.task_manager import TaskManager


class RepoAlignmentTests(unittest.TestCase):
    def test_register_generated_task(self):
        task_manager = TaskManager(
            {
                "task_generation_model": "None",
                "tti_threshold": 1.0,
                "hard_ddl": 10.0,
            }
        )
        task = Task("Task_ext", "UAV_0", 1.0, 1.0, 5.0, 1.0, 0.0)
        registered = task_manager.registerGeneratedTask(task)
        self.assertIs(registered, task)
        self.assertIn(task, task_manager.getWaitingToOffloadTasks()["UAV_0"])
        self.assertIn(task, task_manager._generated_task_history["UAV_0"])

    def test_scheduler_register_generated_task(self):
        env = type("Env", (), {})()
        env.task_manager = TaskManager({"task_generation_model": "None", "tti_threshold": 1.0, "hard_ddl": 10.0})
        task = Task("Task_ext", "UAV_0", 1.0, 1.0, 5.0, 1.0, 0.0)
        registered = AirFogSimScheduler.getTaskScheduler().registerGeneratedTask(env, task)
        self.assertIs(registered, task)
        self.assertIn(task, env.task_manager.getWaitingToOffloadTasks()["UAV_0"])

    def test_cloud_config_explicit_cloud_profile(self):
        env = object.__new__(AirFogSimEnv)
        env.config = {
            "fog_profile": {"rsu": {"cpu": 10}, "cloud": {"cpu": 99}},
            "task_profile": {"rsu": {"lambda": 0.0}, "cloud": {"lambda": 0.0}},
        }
        env.RSUs = {}
        env.cloudServers = {}
        env.traffic_manager = type(
            "TrafficManagerStub",
            (),
            {
                "getRSUInfos": lambda self: {"RSU_0": {"position": (0, 0, 0)}},
                "getCloudServerInfos": lambda self: {"cloudServer_0": {"position": (0, 0, 0)}},
            },
        )()
        env._initRSUsAndCloudServers()
        self.assertIsInstance(env.RSUs["RSU_0"], RSU)
        self.assertIsInstance(env.cloudServers["cloudServer_0"], CloudServer)
        self.assertEqual(env.cloudServers["cloudServer_0"].getFogProfile()["cpu"], 99)


if __name__ == "__main__":
    unittest.main()
