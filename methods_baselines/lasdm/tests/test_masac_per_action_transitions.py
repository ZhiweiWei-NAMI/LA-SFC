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

from airfogsim.lasdm.graph_observation import BASE_CANDIDATE_FEATURE_DIM, flatten_observation  # noqa: E402
from airfogsim.lasdm.marl_policy import MASACPolicy  # noqa: E402
from airfogsim.lasdm.marl_reward import SFCRewardConfig, compute_candidate_action_reward  # noqa: E402
from airfogsim.lasdm.marl_trainer import (  # noqa: E402
    ReplayBuffer,
    RunningRewardNormalizer,
    add_per_action_transitions,
    apply_stage_credits,
    apply_terminal_credits,
    build_action_contexts_from_observations,
    build_per_action_transitions,
)


RAW_CANDIDATE_DIM = BASE_CANDIDATE_FEATURE_DIM + 384


def _candidate_set(sfc_id: str, node_id: str, index: int, instance_id: str) -> dict:
    features = np.zeros((1, RAW_CANDIDATE_DIM), dtype=np.float32)
    features[0, 0] = 0.8
    features[0, 15] = 1.0
    features[0, BASE_CANDIDATE_FEATURE_DIM:] = np.linspace(-1.0, 1.0, 384, dtype=np.float32)
    return {
        "sfc_id": sfc_id,
        "sfc_node_id": node_id,
        "sfc_node_index": index,
        "source_node_id": "src",
        "candidate_ids": [instance_id],
        "candidate_features": features,
        "candidate_mask": np.asarray([1.0], dtype=np.float32),
        "raw_candidates": [
            {
                "instance_id": instance_id,
                "node_id": f"{instance_id}_node",
                "service_id": "svc",
                "region_id": "agent_a",
                "semantic_score": 0.8,
                "metadata": {
                    "route_available": 1.0,
                    "route_hops": 1.0,
                    "deadline_slack_s": 5.0,
                    "function_budget_s": 10.0,
                    "resource_available_ratio": 0.5,
                    "remaining_deadline_ratio": 0.5,
                    "semantic_cumulative_quality_if_selected": 0.7,
                },
            }
        ],
    }


def _observation() -> dict:
    return {
        "agent_id": "agent_a",
        "time_s": 0.0,
        "node_features": np.zeros((1, 16), dtype=np.float32),
        "node_mask": np.zeros((1,), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_features": np.zeros((0, 17), dtype=np.float32),
        "edge_mask": np.zeros((0,), dtype=np.float32),
        "candidate_sets": [
            _candidate_set("sfc0", "node0", 0, "inst0"),
            _candidate_set("sfc0", "node1", 1, "inst1"),
            _candidate_set("sfc1", "node0", 0, "inst2"),
        ],
        "temporal_features": {},
        "neighbor_agent_ids": [],
    }


def _actions() -> dict:
    return {
        "agent_a": {
            "sfc0": {
                "node0": {"instance_id": "inst0", "compute_level": 1.0, "bandwidth_level": 1.0},
                "node1": {"instance_id": "inst1", "compute_level": 0.5, "bandwidth_level": 0.5},
            },
            "sfc1": {
                "node0": {"instance_id": "inst2", "compute_level": 1.0, "bandwidth_level": 0.5},
            },
        }
    }


class MASACPerActionTransitionTests(unittest.TestCase):
    def test_three_placement_actions_generate_three_transitions(self):
        observations = {"agent_a": _observation()}
        actions = _actions()
        transitions, metrics = build_per_action_transitions(
            observations,
            actions,
            build_action_contexts_from_observations(observations, actions),
            observations,
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )

        self.assertEqual(len(transitions), 3)
        self.assertEqual(metrics["replay_transitions_added"], 3.0)
        self.assertEqual([item.selected_instance_id for item in transitions], ["inst0", "inst1", "inst2"])
        self.assertEqual(transitions[1].action_filter(), {"agent_id": "agent_a", "sfc_id": "sfc0", "sfc_node_id": "node1"})
        self.assertEqual(transitions[0].next_action_filter, {"agent_id": "agent_a", "sfc_id": "sfc0", "sfc_node_id": "node1"})

    def test_second_stage_context_uses_previous_selected_host(self):
        observation = _observation()
        node1_candidate = observation["candidate_sets"][1]["raw_candidates"][0]
        node1_candidate["metadata"] = {
            **node1_candidate["metadata"],
            "route_available_by_source": {"src": 0.0, "inst0_node": 1.0},
            "route_hops_by_source": {"src": 9.0, "inst0_node": 1.0},
        }
        observations = {"agent_a": observation}
        actions = _actions()
        contexts = build_action_contexts_from_observations(observations, actions)
        transitions, _metrics = build_per_action_transitions(
            observations,
            actions,
            contexts,
            observations,
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )

        self.assertEqual(transitions[1].source_node_id, "inst0_node")
        self.assertEqual(transitions[1].selected_candidate["metadata"]["route_available"], 1.0)
        self.assertEqual(transitions[1].selected_candidate["metadata"]["route_hops"], 1.0)

    def test_zero_action_step_adds_no_transition(self):
        transitions, metrics = build_per_action_transitions(
            {"agent_a": _observation()},
            {},
            {},
            {"agent_a": _observation()},
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )

        self.assertEqual(transitions, [])
        self.assertEqual(metrics["no_action_steps_skipped"], 1.0)

    def test_candidate_dense_reward_is_single_candidate_local(self):
        candidate = _observation()["candidate_sets"][0]["raw_candidates"][0]
        reward = compute_candidate_action_reward(candidate, SFCRewardConfig(dense_clip=100.0))
        worse = dict(candidate)
        worse["metadata"] = {**candidate["metadata"], "route_available": 0.0}

        self.assertGreater(reward, compute_candidate_action_reward(worse, SFCRewardConfig(dense_clip=100.0)))

    def test_terminal_credit_updates_last_chain_action_and_normalizer(self):
        observations = {"agent_a": _observation()}
        actions = _actions()
        transitions, _metrics = build_per_action_transitions(
            observations,
            actions,
            build_action_contexts_from_observations(observations, actions),
            observations,
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )
        replay = ReplayBuffer(capacity=8, seed=1)
        normalizer = RunningRewardNormalizer(clip=5.0)
        last_by_chain = {}
        add_per_action_transitions(replay, transitions, last_by_chain, reward_normalizer=normalizer)
        old_count = normalizer.count
        old_dense = transitions[1].dense_reward
        metrics = apply_terminal_credits(
            [{"sfc_id": "sfc0", "status": "succeeded", "terminal_reward": 10.0}],
            last_by_chain,
            replay,
            reward_normalizer=normalizer,
        )

        self.assertEqual(metrics["orphan_terminal_credit_count"], 0.0)
        self.assertEqual(normalizer.count, old_count)
        self.assertAlmostEqual(transitions[1].reward, old_dense + 10.0)
        self.assertAlmostEqual(transitions[0].terminal_credit, 0.0)

    def test_stage_credit_updates_matching_function_transition(self):
        observations = {"agent_a": _observation()}
        actions = {"agent_a": {"sfc0": {"node0": {"instance_id": "inst0", "compute_level": 1.0, "bandwidth_level": 1.0}}}}
        transitions, _metrics = build_per_action_transitions(
            observations,
            actions,
            build_action_contexts_from_observations(observations, actions),
            observations,
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )
        replay = ReplayBuffer(capacity=8, seed=1)
        last_by_chain = {}
        last_by_decision = {}
        add_per_action_transitions(replay, transitions, last_by_chain, last_by_decision)
        dense = transitions[0].dense_reward
        metrics = apply_stage_credits(
            [{"sfc_id": "sfc0", "sfc_node_id": "node0", "status": "succeeded", "stage_count": 3}],
            last_by_decision,
            replay,
            SFCRewardConfig(),
        )

        self.assertEqual(metrics["orphan_stage_credit_count"], 0.0)
        self.assertGreater(transitions[0].stage_credit, 0.0)
        self.assertAlmostEqual(transitions[0].reward, dense + transitions[0].stage_credit)

    def test_masac_q_and_actor_are_filtered_to_one_slot(self):
        observations = {"agent_a": _observation()}
        actions = {"agent_a": {"sfc0": {"node0": {"instance_id": "inst0", "compute_level": 1.0, "bandwidth_level": 1.0}}}}
        transitions, _metrics = build_per_action_transitions(
            observations,
            actions,
            build_action_contexts_from_observations(observations, actions),
            observations,
            False,
            0,
            "unit",
            SFCRewardConfig(),
        )
        obs_dim = max(len(flatten_observation(item)) for item in observations.values())
        policy = MASACPolicy(
            observation_dim=obs_dim,
            max_candidates=1,
            candidate_feature_dim=RAW_CANDIDATE_DIM,
            hidden_dim=16,
            mlp_depth=1,
            gnn_layers=1,
            q_mlp_depth=1,
            lr=1e-3,
            q_lr=1e-3,
            device="cpu",
            centralized_critic=True,
            critic_observation_dim=obs_dim,
            max_critic_agents=1,
        )

        with self.assertRaises(ValueError):
            policy.masac_selected_q_values(observations, transitions[0].actions, action_filter={}, action_context={})
        selected = policy.masac_selected_q_values(
            observations,
            transitions[0].actions,
            action_filter=transitions[0].action_filter(),
            action_context=transitions[0].action_context(),
        )
        actor_loss, entropy, target_entropy, count = policy.masac_actor_loss(
            observations,
            action_filter=transitions[0].action_filter(),
            action_context=transitions[0].action_context(),
        )

        self.assertEqual(selected["action_count"], 1)
        self.assertEqual(count, 1)
        self.assertEqual(actor_loss.numel(), 1)
        self.assertEqual(entropy.numel(), 1)
        self.assertEqual(target_entropy.numel(), 1)


if __name__ == "__main__":
    unittest.main()
