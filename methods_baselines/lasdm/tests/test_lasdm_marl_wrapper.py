import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.lasdm import (  # noqa: E402
    LASDMManager,
    LASDMQoS,
    LASDMSFCNode,
    LASDMServiceChain,
    ServiceInstance,
    ServiceInstanceDirectory,
)
from airfogsim.lasdm.marl import LASDMMARLInterface, LASDMMARLWrapper  # noqa: E402


def make_directory():
    directory = ServiceInstanceDirectory()
    directory.register(
        ServiceInstance(
            instance_id="uav_pre",
            service_id="preprocess",
            node_id="UAV_0",
            node_type="uav",
            region_id="RSU_0",
            capabilities=("video_preprocess",),
            input_semantic="video",
            output_semantic="keyframes",
            capacity={"cpu": 2.0, "memory": 256.0},
            max_concurrency=2,
        )
    )
    directory.register(
        ServiceInstance(
            instance_id="rsu_detect",
            service_id="detect",
            node_id="RSU_0",
            node_type="rsu",
            region_id="RSU_0",
            capabilities=("object_detection",),
            input_semantic="keyframes",
            output_semantic="detection_result",
            capacity={"cpu": 4.0, "memory": 512.0},
            max_concurrency=2,
        )
    )
    return directory


def make_chain(sfc_id="sfc_test"):
    return LASDMServiceChain(
        sfc_id=sfc_id,
        source_node_id="UAV_0",
        sink_node_id="UAV_0",
        payload_semantic="video",
        payload_mb=4.0,
        qos=LASDMQoS(deadline_s=5.0, reliability_min=0.0, accuracy_min=0.0),
        nodes={
            "pre": LASDMSFCNode(
                node_id="pre",
                service_type="preprocess",
                required_capabilities=("video_preprocess",),
                input_semantic="video",
                output_semantic="keyframes",
                cpu_mb=1.0,
            ),
            "det": LASDMSFCNode(
                node_id="det",
                service_type="detect",
                required_capabilities=("object_detection",),
                input_semantic="keyframes",
                output_semantic="detection_result",
                cpu_mb=1.0,
            ),
        },
        edges=[("pre", "det")],
        context={"preferred_region_id": "RSU_0"},
    )


class CompletingEnv:
    def __init__(self, manager):
        self.manager = manager
        self.simulation_time = 0.0
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        self.simulation_time += 1.0
        for sfc_id, chain in self.manager.chains.items():
            if not chain.is_terminal() and sfc_id in self.manager.decisions:
                self.manager.complete(sfc_id, self.simulation_time)


class LASDMMARLWrapperTests(unittest.TestCase):
    def test_reset_step_decode_action_reward_and_done(self):
        manager = LASDMManager(directory=make_directory())
        manager.submit(make_chain(), current_time=0.0)
        env = CompletingEnv(manager)
        wrapper = LASDMMARLInterface(env=env, manager=manager)

        observation = wrapper.reset(current_time=0.0)
        self.assertEqual(observation["time_s"], 0.0)
        self.assertEqual(len(observation["graphs"]), 1)

        observation, reward, done, info = wrapper.step({"pre": "uav_pre", "det": "rsu_detect"})

        self.assertTrue(done)
        self.assertGreater(reward, 0.0)
        self.assertEqual(env.step_calls, 1)
        self.assertEqual(observation["metrics"]["succeeded"], 1)
        self.assertEqual(info["decisions"][0]["assignments"]["pre"], "uav_pre")
        self.assertEqual(info["decisions"][0]["assignments"]["det"], "rsu_detect")

    def test_step_rejects_invalid_action_and_scores_failure(self):
        manager = LASDMManager(directory=make_directory())
        manager.submit(make_chain(), current_time=0.0)
        wrapper = LASDMMARLWrapper(manager=manager)
        wrapper.reset()

        _observation, reward, done, info = wrapper.step(
            {"sfc_id": "sfc_test", "assignments": {"pre": "missing_instance"}}
        )

        self.assertTrue(done)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["summary"]["failed"], 1)
        self.assertEqual(info["decisions"][0]["rejected_reason"], "no_candidate")
        self.assertEqual(info["decisions"][0]["diagnostics"]["missing_instance_id"], "missing_instance")

    def test_no_action_uses_manager_planner(self):
        manager = LASDMManager(directory=ServiceInstanceDirectory())
        manager.submit(make_chain(), current_time=0.0)
        wrapper = LASDMMARLWrapper(manager=manager)
        wrapper.reset()

        _observation, reward, done, info = wrapper.step(current_time=0.0)

        self.assertTrue(done)
        self.assertLess(reward, 0.0)
        self.assertEqual(info["summary"]["failed"], 1)
        self.assertEqual(info["decisions"][0]["rejected_reason"], "no_candidate")


if __name__ == "__main__":
    unittest.main()
