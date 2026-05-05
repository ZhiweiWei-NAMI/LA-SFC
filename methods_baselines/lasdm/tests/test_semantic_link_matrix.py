import os
import sys
import tempfile
import unittest
from collections import Counter


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.marl_policy import policy_from_name  # noqa: E402
from airfogsim.lasdm.semantic_link_matrix import SemanticLinkMatrix  # noqa: E402
from airfogsim.lasdm.semantic_link_predictor import SemanticLinkScorer  # noqa: E402
from evaluate_semantic_topology_marl import IPPO_BASELINE_CONFIG_UPDATES, _baseline_settings  # noqa: E402
from train_semantic_topology_marl import (  # noqa: E402
    _materialize_offline_config,
    _materialized_node_id,
    _materialized_service_instances,
    inject_semantic_topology_decoys,
)


SEMANTIC_ROOT = os.path.join(METHOD_ROOT, "configs", "semantic")


def _matrix() -> SemanticLinkMatrix:
    return SemanticLinkMatrix(SEMANTIC_ROOT)


class SemanticLinkMatrixTests(unittest.TestCase):
    def test_yaml_profile_pool_passes_v21_audit_gates(self):
        matrix = _matrix()
        audit = matrix.audit()

        self.assertEqual(audit["service_type_count"], 16)
        self.assertGreaterEqual(audit["implementation_count"], 80)
        self.assertTrue(all(count >= 5 for count in audit["implementations_per_service_type"].values()))
        self.assertGreaterEqual(audit["profile_word_count_min"], 100)
        self.assertLessEqual(audit["profile_word_count_max"], 300)
        self.assertTrue(all(audit["quality_gates"].values()))

    def test_truth_scores_are_logically_ordered_for_relevant_service(self):
        matrix = _matrix()
        by_variant = {
            impl.variant_type: matrix.truth_for_candidate(
                "forest_fire_monitoring",
                "fire_candidate_detection",
                {"implementation_id": impl.implementation_id},
                link_input_semantic=impl.input_semantic_label,
            )["semantic_link_truth_score"]
            for impl in matrix.list_implementations_for("fire_candidate_detection")
        }

        self.assertGreater(by_variant["exact"], by_variant["equivalent"])
        self.assertGreater(by_variant["equivalent"], by_variant["compatible"])
        self.assertGreater(by_variant["compatible"], by_variant["weak"])
        self.assertGreater(by_variant["weak"], by_variant["mismatch"])

    def test_materialized_instances_and_decoys_use_truth_profiles_without_bias_fields(self):
        matrix = _matrix()
        scenario = {
            "name": "unit_truth_decoy",
            "service_nodes": {"rsu": 1, "cloud_server": 1},
            "candidate_decoys": {
                "enabled": True,
                "hard_negative_mismatch": 1,
                "borderline_weak": 1,
                "remote_exact": 1,
                "remote_compatible": 1,
                "stale_clone_exact": 1,
            },
        }
        instances = _materialized_service_instances({}, scenario, {"rsu", "cloud_server"}, 3, matrix)
        per_service = Counter(str(item["service_id"]) for item in instances)
        self.assertEqual(set(per_service), set(matrix.list_service_types()))
        self.assertGreaterEqual(min(per_service.values()), 2)

        with_decoys = inject_semantic_topology_decoys(instances, scenario, 3, {"rsu", "cloud_server"}, matrix)
        decoy_groups = Counter(
            str(item.get("metadata", {}).get("semantic_group", ""))
            for item in with_decoys
            if item.get("metadata", {}).get("is_decoy")
        )
        self.assertEqual(decoy_groups["hard_negative_mismatch"], 16)
        self.assertEqual(decoy_groups["borderline_weak"], 16)
        self.assertEqual(decoy_groups["remote_exact"], 16)
        self.assertEqual(decoy_groups["remote_compatible"], 16)
        self.assertEqual(decoy_groups["stale_clone_exact"], 16)
        self.assertFalse(any("semantic_score_bias" in str(item) for item in with_decoys))

    def test_offline_materialization_exports_v21_artifacts_to_configured_root(self):
        matrix = _matrix()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "semantic_profiles": {
                    "config_root": SEMANTIC_ROOT,
                    "output_root": temp_dir,
                },
                "semantic_exchange": {"encoder_backend": "hash"},
                "topology": {"service_role_sweeps": {"rsu_only": ["rsu"]}},
                "service_chains": [
                    {
                        "sfc_id": "sfc_unit",
                        "source_node_id": "UAV_0",
                        "sink_node_id": "UAV_0",
                        "payload_semantic": "video",
                        "payload_mb": 4.0,
                        "deadline_s": 8.0,
                        "nodes": [
                            {"node_id": "n0", "service_type": "legacy_preprocess"},
                            {"node_id": "n1", "service_type": "legacy_detect"},
                        ],
                        "edges": [["n0", "n1"]],
                        "context": {},
                    }
                ],
            }
            scenario = {
                "name": "unit_materialize",
                "service_nodes": {"rsu": 1},
                "task_nodes": {"uav": 1},
                "request_count": 1,
            }

            materialized = _materialize_offline_config(config, scenario, "rsu_only", seed=11)

            self.assertIsInstance(materialized["_semantic_matrix"], SemanticLinkMatrix)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "semantic_dataset_audit.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "semantic_link_truth.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "semantic_embedding_cache.pt")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "semantic_embedding_report.json")))
            chain = materialized["service_chains"][0]
            self.assertIn(chain["context"]["request_type"], matrix.request_types)
            self.assertNotEqual(chain["nodes"][0]["service_type"], "legacy_preprocess")
            self.assertFalse(any("semantic_score_bias" in str(item) for item in materialized["service_instances"]))

    def test_link_predictor_exposes_candidate_tensor_api(self):
        self.assertTrue(callable(getattr(SemanticLinkScorer, "score_candidate_tensor", None)))

    def test_cloud_server_materialized_node_id_matches_physical_config(self):
        self.assertEqual(_materialized_node_id("cloud_server", 0), "cloudServer_0")

class SemanticBaselinePolicyTests(unittest.TestCase):
    def test_removed_semantic_greedy_names_fail_fast(self):
        with self.assertRaises(ValueError):
            policy_from_name("semantic_greedy_no_exchange")
        with self.assertRaises(ValueError):
            policy_from_name("semantic_greedy_with_exchange")
        with self.assertRaises(ValueError):
            _baseline_settings("semantic_greedy_no_exchange")
        with self.assertRaises(ValueError):
            _baseline_settings("semantic_greedy_with_exchange")

    def test_new_semantic_greedy_baselines_have_distinct_runtime_behavior(self):
        observations = {
            "agent_rsu": {
                "candidate_sets": [
                    {
                        "sfc_id": "sfc0",
                        "sfc_node_id": "node0",
                        "raw_candidates": [
                            {
                                "instance_id": "semantic_high_runtime_bad",
                                "node_id": "RSU_1",
                                "node_type": "rsu",
                                "semantic_score": 0.99,
                                "metadata": {
                                    "route_available": 0.0,
                                    "deadline_slack_s": -4.0,
                                    "function_budget_s": 1.0,
                                    "expected_runtime_penalty_s": 5.0,
                                    "resource_available_slots": 1,
                                },
                            },
                            {
                                "instance_id": "semantic_mid_runtime_good",
                                "node_id": "RSU_0",
                                "node_type": "rsu",
                                "semantic_score": 0.74,
                                "metadata": {
                                    "route_available": 1.0,
                                    "deadline_slack_s": 2.0,
                                    "function_budget_s": 1.0,
                                    "expected_runtime_penalty_s": 0.1,
                                    "resource_available_slots": 1,
                                },
                            },
                        ],
                    }
                ]
            }
        }

        pure_action = policy_from_name("pure_semantic_greedy_no_exchange").act(observations)
        local_action = policy_from_name("local_semantic_runtime_greedy").act(observations)

        self.assertEqual(
            pure_action["agent_rsu"]["sfc0"]["node0"]["instance_id"],
            "semantic_high_runtime_bad",
        )
        self.assertEqual(
            local_action["agent_rsu"]["sfc0"]["node0"]["instance_id"],
            "semantic_mid_runtime_good",
        )

    def test_ablation_config_names_are_explicit_v21_entries(self):
        self.assertIn("pure_semantic_greedy_no_exchange", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertIn("local_semantic_runtime_greedy", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertIn("nsga2_semantic_qos", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertIn("mappo_ctde", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertIn("iql_offline", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertIn("marl_no_semantic", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertIn("marl_semantic_no_topology", IPPO_BASELINE_CONFIG_UPDATES)
        self.assertNotIn("semantic_greedy_no_exchange", IPPO_BASELINE_CONFIG_UPDATES)


if __name__ == "__main__":
    unittest.main()
