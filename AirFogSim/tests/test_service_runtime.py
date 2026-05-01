import os
import unittest

from airfogsim.service_orchestration.service_catalog import ServiceCatalog
from airfogsim.service_orchestration.service_runtime import AirFogServiceRuntime
from airfogsim.service_orchestration.service_spec import PayloadProfile, QoSProfile, ServiceCategory, ServiceGraphSpec, ServiceIntent
from airfogsim.service_orchestration.testing import build_smoke_env, force_finish_all_active_tasks


ROOT = os.path.dirname(os.path.dirname(__file__))


class ServiceRuntimeTests(unittest.TestCase):
    def test_runtime_spawns_successors_incrementally(self):
        env = build_smoke_env()
        catalog = ServiceCatalog.from_yaml(os.path.join(ROOT, "examples", "service_catalog.yaml"))
        runtime = AirFogServiceRuntime()
        intent = ServiceIntent(
            intent_id="runtime",
            source_node_id="UAV_0",
            category=ServiceCategory.MISSION_APPLICATION,
            required_capabilities=["video_preprocess", "lightweight_inference", "object_verification"],
            payload=PayloadProfile("video", 5.0),
            qos=QoSProfile(3.0),
            sink_node_id="UAV_0",
        )
        graph = ServiceGraphSpec(
            graph_id="graph_runtime",
            intent=intent,
            nodes={
                "keyframe_extraction": catalog.get("keyframe_extraction"),
                "lightweight_object_detection": catalog.get("lightweight_object_detection"),
                "edge_object_verification": catalog.get("edge_object_verification"),
            },
            edges=[
                ("keyframe_extraction", "lightweight_object_detection"),
                ("lightweight_object_detection", "edge_object_verification"),
            ],
        )

        runtime.submit(env, graph)
        waiting = env.task_manager.getWaitingToOffloadTasks()["UAV_0"]
        self.assertEqual(len(waiting), 1)
        self.assertEqual(getattr(waiting[0], "_microservice_id"), "keyframe_extraction")

        force_finish_all_active_tasks(env, {"keyframe_extraction": "RSU_0"})
        runtime.on_airfogsim_step_finished(env)
        second = env.task_manager.getWaitingToOffloadTasks()["RSU_0"][0]
        self.assertEqual(getattr(second, "_microservice_id"), "lightweight_object_detection")
        self.assertEqual(second.getTaskNodeId(), "RSU_0")
        self.assertEqual(second.getReturnedSize(), 0.0)

        force_finish_all_active_tasks(env, {"lightweight_object_detection": "RSU_0"})
        runtime.on_airfogsim_step_finished(env)
        third = env.task_manager.getWaitingToOffloadTasks()["RSU_0"][0]
        self.assertEqual(getattr(third, "_microservice_id"), "edge_object_verification")
        self.assertGreater(third.getReturnedSize(), 0.0)

        force_finish_all_active_tasks(env, {"edge_object_verification": "cloudServer_0"})
        runtime.on_airfogsim_step_finished(env)
        self.assertTrue(runtime.is_graph_finished(graph.graph_id))


if __name__ == "__main__":
    unittest.main()
