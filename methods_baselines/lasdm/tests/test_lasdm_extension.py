import csv
import os
import sys
import tempfile
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
PLOTTING_ROOT = os.path.join(WORKSPACE_ROOT, "experiment_artifacts", "plotting")
for path in (AIRFOGSIM_ROOT, PLOTTING_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm import (
    LASDMMARLInterface,
    LASDMManager,
    LASDMQoS,
    LASDMSFCNode,
    LASDMServiceChain,
    SFCFailureReason,
    ServiceInstance,
    ServiceInstanceDirectory,
    ServiceSelectionQuery,
    build_manager_from_yaml,
)


CONFIG_PATH = os.path.join(METHOD_ROOT, "configs", "lasdm_airfogsim.yaml")


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
            max_concurrency=1,
            reliability_score=0.95,
            accuracy_score=0.95,
            trust_score=0.95,
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
            capacity={"cpu": 4.0, "memory": 1024.0},
            max_concurrency=2,
            reliability_score=0.96,
            accuracy_score=0.93,
            trust_score=0.98,
        )
    )
    return directory


def make_chain():
    return LASDMServiceChain(
        sfc_id="sfc_test",
        source_node_id="UAV_0",
        sink_node_id="UAV_0",
        payload_semantic="video",
        payload_mb=4.0,
        qos=LASDMQoS(deadline_s=2.0, reliability_min=0.9, accuracy_min=0.9),
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
                output_semantic="detection_result",
                cpu_mb=1.0,
                memory_mb=128.0,
            ),
        },
        edges=[("pre", "det")],
        context={"preferred_region_id": "RSU_0"},
    )


class LASDMExtensionTests(unittest.TestCase):
    def test_directory_filters_and_reserves_instances(self):
        directory = make_directory()
        query = ServiceSelectionQuery(
            service_id="preprocess",
            required_capabilities=("video_preprocess",),
            input_semantic="video",
            min_reliability=0.9,
            min_accuracy=0.9,
            min_trust=0.9,
            resource_request={"cpu": 1.0, "memory": 64.0},
        )
        selected = directory.select_best(query)
        self.assertIsNotNone(selected)
        directory.reserve(selected.instance_id, query.resource_request)
        self.assertEqual(directory.candidates(query), [])
        directory.release(selected.instance_id, query.resource_request)
        self.assertEqual(len(directory.candidates(query)), 1)

    def test_manager_plans_and_records_success(self):
        manager = LASDMManager(directory=make_directory())
        manager.submit(make_chain(), current_time=0.0)
        decisions = manager.step(current_time=0.0)
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0].accepted)
        self.assertEqual(decisions[0].node_mapping["pre"], "UAV_0")
        self.assertEqual(decisions[0].node_mapping["det"], "RSU_0")
        manager.complete("sfc_test", current_time=1.5)
        self.assertEqual(manager.summary()["succeeded"], 1)
        self.assertEqual(manager.summary()["qos_hit_ratio"], 1.0)

    def test_manager_records_no_candidate_failure(self):
        manager = LASDMManager(directory=ServiceInstanceDirectory())
        manager.submit(make_chain(), current_time=0.0)
        decisions = manager.step(current_time=0.0)
        self.assertFalse(decisions[0].accepted)
        self.assertEqual(decisions[0].rejected_reason, SFCFailureReason.NO_CANDIDATE)
        self.assertEqual(manager.summary()["failed"], 1)

    def test_yaml_config_loads_and_plans(self):
        manager = build_manager_from_yaml(CONFIG_PATH)
        decisions = manager.step(current_time=0.0)
        self.assertEqual(len(decisions), 4)
        self.assertTrue(all(decision.accepted for decision in decisions))
        self.assertTrue(all("verify" in decision.assignments for decision in decisions))

    def test_marl_observation_and_reward(self):
        manager = LASDMManager(directory=make_directory())
        manager.submit(make_chain(), current_time=0.0)
        marl = LASDMMARLInterface()
        obs = marl.build_observation(None, manager)
        self.assertEqual(len(obs["instances"]), 2)
        self.assertEqual(len(obs["graphs"]), 1)
        reward = marl.reward({"succeeded": 0}, {"succeeded": 1, "failed": 0, "timed_out": 0, "avg_latency_s": 1.0})
        self.assertGreater(reward, 0)

    def test_analysis_script_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "summary.csv")
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["baseline", "seed", "graph_completion_ratio"])
                writer.writeheader()
                writer.writerow({"baseline": "a", "seed": 0, "graph_completion_ratio": 0.5})
                writer.writerow({"baseline": "a", "seed": 1, "graph_completion_ratio": 1.0})
            with open(path, "r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            values = [float(row["graph_completion_ratio"]) for row in rows if row["baseline"] == "a"]
            stats = [{"baseline": "a", "metric": "graph_completion_ratio", "mean": sum(values) / len(values)}]
            self.assertEqual(stats[0]["baseline"], "a")
            self.assertAlmostEqual(stats[0]["mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
