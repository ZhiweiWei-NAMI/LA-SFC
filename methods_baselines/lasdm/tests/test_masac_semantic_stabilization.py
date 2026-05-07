import os
import sys
import unittest

import numpy as np


TEST_DIR = os.path.dirname(__file__)
METHOD_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(METHOD_ROOT, "../.."))
AIRFOGSIM_ROOT = os.path.join(WORKSPACE_ROOT, "AirFogSim")
for path in (AIRFOGSIM_ROOT, METHOD_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from airfogsim.lasdm.distributed_catalog import DistributedServiceCatalog  # noqa: E402
from airfogsim.lasdm.graph_observation import BASE_CANDIDATE_FEATURE_DIM, flatten_observation  # noqa: E402
from airfogsim.lasdm.instance_directory import ServiceInstance, ServiceInstanceDirectory  # noqa: E402
from airfogsim.lasdm.marl_policy import MASACPolicy  # noqa: E402
from airfogsim.lasdm.marl_reward import SFCRewardConfig  # noqa: E402
from airfogsim.lasdm.marl_trainer import (  # noqa: E402
    ReplayBuffer,
    RunningRewardNormalizer,
    build_action_contexts_from_observations,
    build_per_action_transitions,
    masac_update_policy,
)
from airfogsim.lasdm.semantic_encoder import SemanticEncoder  # noqa: E402
from airfogsim.lasdm.semantic_exchange import SemanticExchange, SemanticExchangeConfig  # noqa: E402


RAW_CANDIDATE_DIM = BASE_CANDIDATE_FEATURE_DIM + 384


def _directory(instance_id: str, region_id: str) -> ServiceInstanceDirectory:
    directory = ServiceInstanceDirectory()
    directory.register(
        ServiceInstance(
            instance_id=instance_id,
            service_id="detect",
            node_id=f"{region_id}_node",
            capabilities=("object_detection",),
            node_type="rsu",
            region_id=region_id,
            input_semantic="video",
            output_semantic="detections",
            capacity={"cpu": 8.0, "memory": 1024.0},
            max_concurrency=4,
        )
    )
    return directory


def _candidate_features(semantic_score: float, semantic_embedding: np.ndarray) -> np.ndarray:
    features = np.zeros((2, RAW_CANDIDATE_DIM), dtype=np.float32)
    features[0, 0] = np.float32(semantic_score)
    features[0, 15] = 1.0
    features[0, 18] = 0.5
    features[0, BASE_CANDIDATE_FEATURE_DIM:] = semantic_embedding.astype(np.float32)
    features[1, 15] = 1.0
    return features


def _observation(agent_id: str, neighbor_ids: list[str], semantic_embedding: np.ndarray) -> dict:
    node_features = np.zeros((2, 16), dtype=np.float32)
    node_features[0, 12] = 1.0
    node_mask = np.ones((2,), dtype=np.float32)
    candidate_features = _candidate_features(0.8, semantic_embedding)
    return {
        "agent_id": agent_id,
        "time_s": 0.0,
        "node_features": node_features,
        "node_mask": node_mask,
        "edge_index": np.zeros((2, 1), dtype=np.int64),
        "edge_features": np.zeros((1, 17), dtype=np.float32),
        "edge_mask": np.zeros((1,), dtype=np.float32),
        "candidate_sets": [
            {
                "sfc_id": "sfc0",
                "sfc_node_id": "node0",
                "sfc_node_index": 0,
                "source_node_id": f"{agent_id}_source",
                "candidate_ids": [f"{agent_id}_candidate"],
                "candidate_features": candidate_features,
                "candidate_mask": np.asarray([1.0, 0.0], dtype=np.float32),
                "raw_candidates": [
                    {
                        "instance_id": f"{agent_id}_candidate",
                        "service_id": "detect",
                        "node_id": f"{agent_id}_node",
                        "region_id": agent_id,
                        "node_type": "rsu",
                        "source": "local",
                        "owner_agent_id": agent_id,
                        "semantic_score": 0.8,
                        "staleness_s": 0.0,
                        "is_remote": False,
                        "semantic_embedding": semantic_embedding.astype(np.float32).tolist(),
                        "metadata": {
                            "route_available": 1.0,
                            "route_hops": 1.0,
                            "deadline_slack_s": 10.0,
                            "function_budget_s": 20.0,
                            "resource_available_ratio": 1.0,
                            "max_concurrency": 4,
                            "current_load": 0,
                        },
                    }
                ],
            }
        ],
        "temporal_features": {},
        "neighbor_agent_ids": neighbor_ids,
    }


class MASACSemanticStabilizationTests(unittest.TestCase):
    def test_semantic_exchange_advertises_full_384d_embedding(self):
        encoder = SemanticEncoder(model_name="unit-hash-384", backend="hash", hash_dim=384)
        exchange = SemanticExchange(SemanticExchangeConfig(embedding_dim=384, top_k_per_agent=4))
        advertisements = exchange.build_advertisements("agent_b", _directory("detect_b", "agent_b"), encoder, now_s=1.0)

        self.assertEqual(len(advertisements), 1)
        self.assertEqual(len(advertisements[0].semantic_embedding), 384)
        self.assertIn("semantic_embedding", advertisements[0].to_dict())

        catalog = DistributedServiceCatalog("agent_a", _directory("detect_a", "agent_a"), encoder=encoder)
        accepted = catalog.ingest_remote(advertisements, now_s=1.0)
        candidates = catalog.query("video object detection", service_id="detect", now_s=1.0, top_k=4)
        remote = [candidate for candidate in candidates if candidate.is_remote]

        self.assertEqual(accepted, 1)
        self.assertTrue(remote)
        self.assertEqual(len(remote[0].semantic_embedding), 384)

    def test_masac_projection_attention_and_update_use_full_semantic_features(self):
        semantic_a = np.linspace(-1.0, 1.0, 384, dtype=np.float32)
        semantic_b = np.linspace(1.0, -1.0, 384, dtype=np.float32)
        observations = {
            "agent_a": _observation("agent_a", ["agent_b"], semantic_a),
            "agent_b": _observation("agent_b", ["agent_a"], semantic_b),
        }
        obs_dim = max(len(flatten_observation(item)) for item in observations.values())
        policy = MASACPolicy(
            observation_dim=obs_dim,
            max_candidates=2,
            candidate_feature_dim=RAW_CANDIDATE_DIM,
            hidden_dim=32,
            mlp_depth=1,
            gnn_layers=1,
            q_mlp_depth=1,
            lr=1e-3,
            q_lr=1e-3,
            device="cpu",
            semantic_projection_dim=8,
            cross_agent_attention_enabled=True,
            cross_agent_attention_heads=4,
            centralized_critic=True,
            critic_observation_dim=obs_dim * 2,
            max_critic_agents=2,
        )

        projected = policy._candidate_feature_tensor(
            observations["agent_a"]["candidate_sets"][0],
            1,
            policy.torch.zeros((1,), dtype=policy.torch.float32, device=policy.device),
        )
        context_bundle = policy._region_context_bundle(observations)
        step = policy.act_with_logprobs(observations, deterministic=True)
        replay = ReplayBuffer(capacity=4, seed=3)
        transitions, transition_metrics = build_per_action_transitions(
            observations,
            step.actions,
            step.decision_contexts,
            observations,
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )
        replay.add(transitions[0])
        metrics = masac_update_policy(policy, replay, batch_size=1, updates=1)

        self.assertEqual(projected.shape[-1], 38)
        self.assertEqual(set(context_bundle), {"agent_a", "agent_b"})
        self.assertIn("agent_a", step.actions)
        self.assertEqual(transition_metrics["replay_transitions_added"], 2.0)
        self.assertIn("critic_loss", metrics)

    def test_reward_normalizer_math_and_clipping(self):
        normalizer = RunningRewardNormalizer(clip=1.0)
        for reward in (10.0, 14.0, 18.0):
            normalizer.update(reward)

        self.assertAlmostEqual(normalizer.mean, 14.0)
        self.assertAlmostEqual(normalizer.std, 4.0)
        self.assertAlmostEqual(normalizer.transform(14.0), 0.0)
        self.assertAlmostEqual(normalizer.transform(100.0), 1.0)
        self.assertAlmostEqual(normalizer.transform(-100.0), -1.0)

    def test_replay_update_schedule_keeps_advancing_after_capacity(self):
        replay = ReplayBuffer(capacity=4, seed=3)
        update_steps = []
        observations = {"agent_a": _observation("agent_a", [], np.zeros((384,), dtype=np.float32))}
        action = {"agent_a": {"sfc0": {"node0": {"instance_id": "agent_a_candidate", "compute_level": 1.0, "bandwidth_level": 1.0}}}}
        for index in range(10):
            transitions, _metrics = build_per_action_transitions(
                observations,
                action,
                build_action_contexts_from_observations(observations, action),
                observations,
                False,
                index,
                "unit",
                SFCRewardConfig(),
            )
            replay.add(
                transitions[0]
            )
            replay.mark_env_step()
            if replay.should_update(warmup_steps=2, update_interval=3):
                update_steps.append(index + 1)

        self.assertEqual(len(replay), 4)
        self.assertEqual(replay.total_added, 10)
        self.assertEqual(update_steps, [3, 6, 9])


if __name__ == "__main__":
    unittest.main()
