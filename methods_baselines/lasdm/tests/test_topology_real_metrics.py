import os
import sys
import unittest


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
if AIRFOGSIM_ROOT not in sys.path:
    sys.path.insert(0, AIRFOGSIM_ROOT)

from airfogsim.lasdm.env_adapter import LASDMEnvAdapter  # noqa: E402
from airfogsim.lasdm.graph_observation import GraphObservationBuilder  # noqa: E402
from airfogsim.lasdm.marl_env import SemanticTopologyMARLEnv  # noqa: E402
from airfogsim.lasdm.topology_builder import TopologyBuilder  # noqa: E402


def _nodes():
    return {
        "vehicle_0": {
            "node_type": "vehicle",
            "region_id": "region_a",
            "position": [0.0, 0.0, 0.0],
            "fog_profile": {"cpu": 4.0},
        },
        "RSU_0": {
            "node_type": "rsu",
            "region_id": "region_b",
            "position": [300.0, 0.0, 0.0],
            "fog_profile": {"cpu": 16.0},
        },
        "cloudServer_0": {
            "node_type": "cloud_server",
            "region_id": "cloud",
            "position": [0.0, 0.0, 0.0],
            "fog_profile": {"cpu": 64.0},
        },
    }


def _snapshot(measured_links):
    return {
        "nodes": _nodes(),
        "tasks": {"computing_by_node": {}},
        "links": {"measured_links": list(measured_links)},
    }


class TopologyRealMetricTests(unittest.TestCase):
    def test_wireless_measured_link_builds_edge_without_unseen_pairs(self):
        topology = TopologyBuilder().from_snapshot(
            _snapshot(
                [
                    {
                        "src": "vehicle_0",
                        "dst": "RSU_0",
                        "link_type": "v2i",
                        "is_wireless": True,
                        "rate_mbps_sum": 30.0,
                        "transmitted_mbit": 60.0,
                        "simulation_interval_s": 1.0,
                        "allocated_rb_count": 2,
                        "task_count": 1,
                    }
                ]
            ),
            current_time=7.0,
        )

        self.assertEqual(len(topology.edges), 1)
        edge = topology.edges[0]
        self.assertEqual(edge.src, "vehicle_0")
        self.assertEqual(edge.dst, "RSU_0")
        self.assertEqual(edge.link_type, "v2i")
        self.assertEqual(edge.rate_mbps, 30.0)
        self.assertEqual(edge.latency_s, 2.0)
        self.assertEqual(edge.reliability, 1.0)
        self.assertTrue(edge.is_wireless)

    def test_wired_measured_link_uses_capacity_and_queue_delay(self):
        topology = TopologyBuilder().from_snapshot(
            _snapshot(
                [
                    {
                        "src": "RSU_0",
                        "dst": "cloudServer_0",
                        "is_wireless": False,
                        "capacity_mbps": 100.0,
                        "prop_ms": 1.0,
                        "queue_bytes_before": 12_500_000.0,
                        "queue_bytes_after": 0.0,
                        "transmitted_bytes": 1000.0,
                        "active_flow_count": 1,
                        "simulation_interval_s": 1.0,
                    }
                ]
            ),
            current_time=8.0,
        )

        self.assertEqual(len(topology.edges), 1)
        edge = topology.edges[0]
        self.assertEqual(edge.link_type, "i2c")
        self.assertFalse(edge.is_wireless)
        self.assertEqual(edge.rate_mbps, 100.0)
        self.assertAlmostEqual(edge.latency_s, 1.001)
        self.assertEqual(edge.reliability, 1.0)

    def test_empty_measured_links_do_not_create_synthetic_edges(self):
        topology = TopologyBuilder().from_snapshot(_snapshot([]), current_time=9.0)

        self.assertEqual(len(topology.edges), 0)
        self.assertEqual(topology.metadata["edge_count"], 0)

    def test_wireless_link_without_transmission_is_not_an_edge(self):
        topology = TopologyBuilder().from_snapshot(
            _snapshot(
                [
                    {
                        "src": "vehicle_0",
                        "dst": "RSU_0",
                        "link_type": "v2i",
                        "is_wireless": True,
                        "rate_mbps_sum": 0.0,
                        "transmitted_bytes": 0.0,
                        "task_count": 1,
                    }
                ]
            ),
            current_time=9.5,
        )

        self.assertEqual(len(topology.edges), 0)

    def test_adapter_reads_measured_link_snapshot_and_empty_default(self):
        class FakeEnv:
            simulation_time = 3.0
            channel = {"time": 1.0, "data_size": 2.0}
            V2U_channel = {}
            V2I_channel = {}
            U2I_channel = {}

        env = FakeEnv()
        adapter = LASDMEnvAdapter()

        empty = adapter.collect_link_snapshot(env)
        self.assertEqual(empty["measured_links"], [])

        env.last_link_metrics_snapshot = {
            "time_s": 4.0,
            "measured_links": [{"src": "vehicle_0", "dst": "RSU_0", "link_type": "v2i"}],
            "wireless": [{"src": "vehicle_0", "dst": "RSU_0"}],
            "wired": [],
        }
        measured = adapter.collect_link_snapshot(env)

        self.assertEqual(measured["time_s"], 4.0)
        self.assertEqual(measured["measured_links"][0]["src"], "vehicle_0")
        self.assertEqual(measured["aggregates"]["channel"]["data_size"], 2.0)

    def test_observation_neighbors_and_route_use_only_measured_edges(self):
        builder = TopologyBuilder()
        empty_topology = builder.from_snapshot(_snapshot([]), current_time=10.0)
        observation = GraphObservationBuilder().build("region_a", empty_topology, [])

        self.assertEqual(float(observation["edge_mask"].sum()), 0.0)
        self.assertEqual(builder.neighbor_map_by_region(empty_topology)["region_a"], [])

        env = object.__new__(SemanticTopologyMARLEnv)
        env.last_topology = empty_topology
        env._route_reachability_cache = {}
        env._route_tree_cache = {}
        env._route_adjacency_topology_id = None
        env._route_adjacency_cache = {}

        unavailable = SemanticTopologyMARLEnv._candidate_route_reachability(env, "vehicle_0", "RSU_0", 1.0)
        self.assertEqual(unavailable, (False, 0, 1.0, 0.0, 0.0, 0, 0.0))

        measured_topology = builder.from_snapshot(
            _snapshot(
                [
                    {
                        "src": "vehicle_0",
                        "dst": "RSU_0",
                        "link_type": "v2i",
                        "is_wireless": True,
                        "rate_mbps_sum": 10.0,
                        "transmitted_mbit": 5.0,
                        "task_count": 1,
                    }
                ]
            ),
            current_time=11.0,
        )
        measured_observation = GraphObservationBuilder().build("region_a", measured_topology, [])

        self.assertEqual(float(measured_observation["edge_mask"].sum()), 1.0)
        self.assertEqual(builder.neighbor_map_by_region(measured_topology)["region_a"], ["region_b"])


if __name__ == "__main__":
    unittest.main()
