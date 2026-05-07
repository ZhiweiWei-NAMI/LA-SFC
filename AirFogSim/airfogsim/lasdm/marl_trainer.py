from __future__ import annotations

import csv
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .marl_env import SemanticTopologyMARLEnv
from .marl_policy import BaseMARLPolicy, MASACPolicy
from .marl_reward import SFCReward, SFCRewardConfig, compute_candidate_action_reward


@dataclass
class TrainingMetrics:
    episode: int
    step: int
    mean_reward: float
    total_reward: float
    succeeded: int
    failed: int
    timed_out: int
    active_graphs: int
    task_done_num: int = 0
    task_fail_num: int = 0
    task_success_ratio: float = 0.0
    scenario: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SACTransition:
    observations: Mapping[str, Mapping[str, Any]]
    actions: Mapping[str, Any]
    reward: float
    next_observations: Mapping[str, Mapping[str, Any]]
    done: bool
    agent_id: str
    sfc_id: str
    sfc_node_id: str
    action_payload: Mapping[str, Any]
    selected_instance_id: str
    dense_reward: float
    terminal_credit: float
    decision_key: str
    next_action_filter: Mapping[str, str] = field(default_factory=dict)
    episode: int = 0
    scenario: str = ""
    source: str = "policy"
    transition_id: int = 0
    normalized_reward_tracked: bool = False

    def action_filter(self) -> Dict[str, str]:
        return {
            "agent_id": str(self.agent_id),
            "sfc_id": str(self.sfc_id),
            "sfc_node_id": str(self.sfc_node_id),
        }

    def add_terminal_credit(self, value: float) -> Tuple[float, float]:
        old_reward = float(self.reward)
        self.terminal_credit = float(self.terminal_credit) + float(value)
        self.reward = float(self.dense_reward) + float(self.terminal_credit)
        return old_reward, float(self.reward)


class RunningRewardNormalizer:
    """Online Welford normalizer for MASAC update targets."""

    def __init__(self, clip: float = 5.0):
        self.clip = abs(float(clip))
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, reward: float) -> None:
        value = float(reward)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        return max(1e-12, self.m2 / float(self.count - 1))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def transform(self, reward: float) -> float:
        normalized = (float(reward) - self.mean) / max(1e-6, self.std)
        if self.clip > 0.0:
            normalized = max(-self.clip, min(self.clip, normalized))
        return float(normalized)

    def snapshot(self) -> Dict[str, float]:
        return {
            "reward_normalizer_count": float(self.count),
            "reward_normalizer_mean": float(self.mean),
            "reward_normalizer_std": float(self.std),
            "reward_normalizer_clip": float(self.clip),
        }


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, seed: int = 0):
        self.capacity = max(1, int(capacity))
        self._items: deque[SACTransition] = deque(maxlen=self.capacity)
        self._rng = random.Random(int(seed))
        self.total_added = 0
        self.env_steps_seen = 0

    def add(self, transition: SACTransition) -> None:
        if not transition.agent_id or not transition.sfc_id or not transition.sfc_node_id or not transition.selected_instance_id:
            raise ValueError("SACTransition must represent exactly one placement action")
        if not transition.decision_key:
            raise ValueError("SACTransition.decision_key is required")
        self._items.append(transition)
        self.total_added += 1
        transition.transition_id = int(self.total_added)

    def mark_env_step(self) -> None:
        self.env_steps_seen += 1

    def should_update(self, warmup_steps: int, update_interval: int) -> bool:
        warmup = max(1, int(warmup_steps))
        interval = max(1, int(update_interval))
        return len(self._items) >= warmup and self.env_steps_seen > 0 and self.env_steps_seen % interval == 0

    def sample(self, batch_size: int, strategy: str = "uniform") -> List[SACTransition]:
        size = min(max(1, int(batch_size)), len(self._items))
        items = list(self._items)
        if str(strategy or "uniform") != "scenario_balanced":
            return self._rng.sample(items, size)
        groups: Dict[str, List[SACTransition]] = {}
        for item in items:
            key = str(getattr(item, "scenario", "") or "default")
            groups.setdefault(key, []).append(item)
        if len(groups) <= 1:
            return self._rng.sample(items, size)
        sampled: List[SACTransition] = []
        keys = sorted(groups)
        quota = max(1, size // len(keys))
        for key in keys:
            group = groups[key]
            sampled.extend(self._rng.sample(group, min(quota, len(group))))
        while len(sampled) < size:
            sampled.append(self._rng.choice(items))
        if len(sampled) > size:
            sampled = self._rng.sample(sampled, size)
        self._rng.shuffle(sampled)
        return sampled

    def __len__(self) -> int:
        return len(self._items)

    def contains(self, transition: SACTransition) -> bool:
        return any(item is transition for item in self._items)


def count_placement_actions(actions: Mapping[str, Any]) -> int:
    return sum(1 for _ in iter_placement_actions(actions))


def iter_placement_actions(actions: Mapping[str, Any]):
    for agent_id, agent_payload in (actions or {}).items():
        if not isinstance(agent_payload, Mapping):
            continue
        for sfc_id, sfc_payload in agent_payload.items():
            if not isinstance(sfc_payload, Mapping):
                continue
            for sfc_node_id, action_payload in sfc_payload.items():
                if not isinstance(action_payload, Mapping):
                    continue
                instance_id = str(action_payload.get("instance_id", action_payload.get("service_instance_id", "")) or "")
                if not instance_id:
                    continue
                yield str(agent_id), str(sfc_id), str(sfc_node_id), dict(action_payload), instance_id


def build_per_action_transitions(
    observations: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, Any],
    next_observations: Mapping[str, Mapping[str, Any]],
    done: bool,
    episode: int,
    scenario: str,
    reward_config: SFCRewardConfig,
    source: str = "policy",
) -> Tuple[List[SACTransition], Dict[str, float]]:
    transitions: List[SACTransition] = []
    lookup = _candidate_lookup_by_decision(observations)
    env_action_count = count_placement_actions(actions)
    for agent_id, sfc_id, sfc_node_id, action_payload, instance_id in iter_placement_actions(actions):
        key = (agent_id, sfc_id, sfc_node_id, instance_id)
        if key not in lookup:
            raise ValueError(f"selected action does not match an observed candidate: {key}")
        candidate, candidate_set = lookup[key]
        dense_reward = compute_candidate_action_reward(candidate, reward_config)
        action = {agent_id: {sfc_id: {sfc_node_id: dict(action_payload)}}}
        decision_key = f"{agent_id}:{sfc_id}:{sfc_node_id}"
        transitions.append(
            SACTransition(
                observations=observations,
                actions=action,
                reward=float(dense_reward),
                next_observations=next_observations,
                done=bool(done),
                agent_id=agent_id,
                sfc_id=sfc_id,
                sfc_node_id=sfc_node_id,
                action_payload=dict(action_payload),
                selected_instance_id=instance_id,
                dense_reward=float(dense_reward),
                terminal_credit=0.0,
                decision_key=decision_key,
                next_action_filter=_next_decision_filter(next_observations, agent_id, sfc_id, candidate_set),
                episode=int(episode),
                scenario=str(scenario or ""),
                source=str(source or "policy"),
            )
        )
    dense_rewards = [float(item.dense_reward) for item in transitions]
    diagnostics = {
        "env_action_count": float(env_action_count),
        "replay_transitions_added": float(len(transitions)),
        "no_action_steps_skipped": float(1.0 if env_action_count <= 0 else 0.0),
        "dense_reward_mean": _mean_float(dense_rewards),
        "dense_reward_min": float(min(dense_rewards) if dense_rewards else 0.0),
        "dense_reward_max": float(max(dense_rewards) if dense_rewards else 0.0),
    }
    return transitions, diagnostics


def add_per_action_transitions(
    replay: ReplayBuffer,
    transitions: Sequence[SACTransition],
    last_transition_by_chain: Dict[str, SACTransition],
    reward_normalizer: Optional[RunningRewardNormalizer] = None,
) -> None:
    for transition in transitions:
        replay.add(transition)
        last_transition_by_chain[str(transition.sfc_id)] = transition
        if reward_normalizer is not None:
            reward_normalizer.update(float(transition.reward))
            transition.normalized_reward_tracked = True


def apply_terminal_credits(
    terminal_events: Sequence[Mapping[str, Any]],
    last_transition_by_chain: Mapping[str, SACTransition],
    replay: ReplayBuffer,
    reward_normalizer: Optional[RunningRewardNormalizer] = None,
) -> Dict[str, float]:
    credits: List[float] = []
    orphan_count = 0
    for event in terminal_events or []:
        sfc_id = str(event.get("sfc_id", "") or "")
        terminal_reward = float(event.get("terminal_reward", 0.0) or 0.0)
        if not sfc_id or terminal_reward == 0.0:
            continue
        transition = last_transition_by_chain.get(sfc_id)
        if transition is None or not replay.contains(transition):
            orphan_count += 1
            continue
        transition.add_terminal_credit(terminal_reward)
        credits.append(float(terminal_reward))
    return {
        "terminal_credit_mean": _mean_float(credits),
        "terminal_credit_min": float(min(credits) if credits else 0.0),
        "terminal_credit_max": float(max(credits) if credits else 0.0),
        "orphan_terminal_credit_count": float(orphan_count),
    }


def reward_config_from_env(env: SemanticTopologyMARLEnv) -> SFCRewardConfig:
    reward_fn = getattr(env, "reward_fn", None)
    if not isinstance(reward_fn, SFCReward):
        raise TypeError("per-action MASAC reward requires env.reward_fn to be SFCReward")
    return reward_fn.config


def _candidate_lookup_by_decision(
    observations: Mapping[str, Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str], Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    lookup: Dict[Tuple[str, str, str, str], Tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for agent_id, observation in (observations or {}).items():
        for candidate_set in observation.get("candidate_sets", []) or []:
            sfc_id = str(candidate_set.get("sfc_id", "") or "")
            sfc_node_id = str(candidate_set.get("sfc_node_id", "") or "")
            for candidate in candidate_set.get("raw_candidates", []) or []:
                instance_id = str(candidate.get("instance_id", "") or "")
                if instance_id:
                    lookup[(str(agent_id), sfc_id, sfc_node_id, instance_id)] = (candidate, candidate_set)
    return lookup


def _next_decision_filter(
    next_observations: Mapping[str, Mapping[str, Any]],
    agent_id: str,
    sfc_id: str,
    current_candidate_set: Mapping[str, Any],
) -> Dict[str, str]:
    current_index = int(current_candidate_set.get("sfc_node_index", 0) or 0)
    next_sets = []
    observation = dict((next_observations or {}).get(str(agent_id), {}) or {})
    for candidate_set in observation.get("candidate_sets", []) or []:
        if str(candidate_set.get("sfc_id", "") or "") != str(sfc_id):
            continue
        next_index = int(candidate_set.get("sfc_node_index", 0) or 0)
        if next_index > current_index and candidate_set.get("raw_candidates", []):
            next_sets.append((next_index, str(candidate_set.get("sfc_node_id", "") or "")))
    if not next_sets:
        return {}
    _, sfc_node_id = min(next_sets, key=lambda item: item[0])
    return {"agent_id": str(agent_id), "sfc_id": str(sfc_id), "sfc_node_id": sfc_node_id}


def _mean_float(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class HeuristicEvaluator:
    def __init__(
        self,
        env: SemanticTopologyMARLEnv,
        policy: BaseMARLPolicy,
        env_factory: Optional[Callable[[int], SemanticTopologyMARLEnv]] = None,
        close_env: Optional[Callable[[SemanticTopologyMARLEnv], None]] = None,
    ):
        self.env = env
        self.policy = policy
        self.env_factory = env_factory
        self.close_env = close_env

    def run(self, episodes: int = 1, max_steps: int = 100, output_dir: Optional[str] = None) -> List[TrainingMetrics]:
        rows: List[TrainingMetrics] = []
        target = Path(output_dir) if output_dir is not None else None
        for episode in range(int(episodes)):
            env = self._episode_env(episode)
            try:
                scenario_name = str(getattr(getattr(env, "config", None), "scenario_name", "") or "")
                observations = env.reset()
                total = 0.0
                for step in range(int(max_steps)):
                    actions = self.policy.act(observations, deterministic=True)
                    observations, rewards, done, info = env.step(actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    summary = info.get("summary", {})
                    rows.append(
                        TrainingMetrics(
                            episode=episode,
                            step=step,
                            mean_reward=mean_reward,
                            total_reward=total,
                            succeeded=int(summary.get("succeeded", 0) or 0),
                            failed=int(summary.get("failed", 0) or 0),
                            timed_out=int(summary.get("timed_out", 0) or 0),
                            active_graphs=int(summary.get("active_graphs", 0) or 0),
                            scenario=scenario_name,
                        )
                    )
                    if done:
                        break
                if target is not None and episode + 1 == int(episodes):
                    target.mkdir(parents=True, exist_ok=True)
                    env.write_traces(str(target))
            finally:
                self._close_episode_env(env)
        return rows

    def _episode_env(self, episode: int) -> SemanticTopologyMARLEnv:
        if self.env_factory is None:
            return self.env
        return self.env_factory(int(episode))

    def _close_episode_env(self, env: SemanticTopologyMARLEnv) -> None:
        if self.env_factory is not None and self.close_env is not None:
            self.close_env(env)


class MASACTrainer:
    """Off-policy discrete SAC trainer for the MASAC CTDE policy."""

    def __init__(
        self,
        env: SemanticTopologyMARLEnv,
        policy: MASACPolicy,
        env_factory: Optional[Callable[[int], SemanticTopologyMARLEnv]] = None,
        close_env: Optional[Callable[[SemanticTopologyMARLEnv], None]] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        replay_capacity: int = 10000,
        replay_warmup_steps: int = 64,
        update_interval: int = 1,
        actor_update_interval: int = 2,
        updates_per_env_step: int = 1,
        max_grad_norm: float = 1.0,
        reward_scale: float = 1.0,
        reward_normalization: bool = True,
        reward_clip: float = 5.0,
        replay_sample_strategy: str = "uniform",
        seed: int = 0,
    ):
        self.env = env
        self.env_factory = env_factory
        self.close_env = close_env
        self.policy = policy
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.replay_warmup_steps = int(replay_warmup_steps)
        self.update_interval = max(1, int(update_interval))
        self.actor_update_interval = max(1, int(actor_update_interval))
        self.updates_per_env_step = int(updates_per_env_step)
        self.max_grad_norm = float(max_grad_norm)
        self.reward_scale = float(reward_scale)
        self.reward_normalizer = RunningRewardNormalizer(clip=reward_clip) if bool(reward_normalization) else None
        self.replay_sample_strategy = str(replay_sample_strategy or "uniform")
        self.replay = ReplayBuffer(replay_capacity, seed=seed)

    def train(self, episodes: int = 10, max_steps: int = 100, output_dir: Optional[str] = None) -> List[TrainingMetrics]:
        rows: List[TrainingMetrics] = []
        diagnostics: List[Dict[str, Any]] = []
        update_index = 0
        target = Path(output_dir) if output_dir is not None else None
        for episode in range(int(episodes)):
            env = self._episode_env(episode)
            try:
                scenario_name = str(getattr(getattr(env, "config", None), "scenario_name", "") or "")
                observations = env.reset()
                total = 0.0
                last_transition_by_chain: Dict[str, SACTransition] = {}
                step_diagnostic_totals = {
                    "env_action_count": 0.0,
                    "replay_transitions_added": 0.0,
                    "no_action_steps_skipped": 0.0,
                    "orphan_terminal_credit_count": 0.0,
                }
                for step in range(int(max_steps)):
                    current_observations = observations
                    policy_step = self.policy.act_with_logprobs(current_observations, deterministic=False, track_grad=False)
                    observations, rewards, done, info = env.step(policy_step.actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    transitions, replay_step_metrics = build_per_action_transitions(
                        current_observations,
                        policy_step.actions,
                        observations,
                        bool(done or step + 1 >= int(max_steps)),
                        int(episode),
                        scenario_name,
                        reward_config_from_env(env),
                    )
                    add_per_action_transitions(
                        self.replay,
                        transitions,
                        last_transition_by_chain,
                        reward_normalizer=self.reward_normalizer,
                    )
                    terminal_metrics = apply_terminal_credits(
                        info.get("terminal_events", []) or [],
                        last_transition_by_chain,
                        self.replay,
                        reward_normalizer=self.reward_normalizer,
                    )
                    for metrics_source in (replay_step_metrics, terminal_metrics):
                        for key in step_diagnostic_totals:
                            step_diagnostic_totals[key] += float(metrics_source.get(key, 0.0) or 0.0)
                    self.replay.mark_env_step()
                    if self.replay.should_update(self.replay_warmup_steps, self.update_interval):
                        next_update_index = update_index + 1
                        metrics = masac_update_policy(
                            self.policy,
                            self.replay,
                            batch_size=self.batch_size,
                            updates=self.updates_per_env_step,
                            gamma=self.gamma,
                            tau=self.tau,
                            max_grad_norm=self.max_grad_norm,
                            reward_scale=self.reward_scale,
                            reward_transform=self.reward_normalizer.transform if self.reward_normalizer is not None else None,
                            sample_strategy=self.replay_sample_strategy,
                            update_actor=(next_update_index % self.actor_update_interval == 0),
                        )
                        if metrics:
                            if self.reward_normalizer is not None:
                                metrics = {**metrics, **self.reward_normalizer.snapshot()}
                            metrics = {**metrics, **step_diagnostic_totals}
                            step_diagnostic_totals = {
                                "env_action_count": 0.0,
                                "replay_transitions_added": 0.0,
                                "no_action_steps_skipped": 0.0,
                                "orphan_terminal_credit_count": 0.0,
                            }
                            update_index += 1
                            diagnostics.append(
                                {
                                    "episode": int(episode),
                                    "step": int(step),
                                    "scenario": scenario_name,
                                    "update_index": update_index,
                                    "replay_size": len(self.replay),
                                    "replay_total_added": self.replay.total_added,
                                    **metrics,
                                }
                            )
                    summary = info.get("summary", {})
                    rows.append(
                        TrainingMetrics(
                            episode=episode,
                            step=step,
                            mean_reward=mean_reward,
                            total_reward=total,
                            succeeded=int(summary.get("succeeded", 0) or 0),
                            failed=int(summary.get("failed", 0) or 0),
                            timed_out=int(summary.get("timed_out", 0) or 0),
                            active_graphs=int(summary.get("active_graphs", 0) or 0),
                            scenario=scenario_name,
                        )
                    )
                    if done:
                        break
                if target is not None and episode + 1 == int(episodes):
                    target.mkdir(parents=True, exist_ok=True)
                    env.write_traces(str(target))
            finally:
                self._close_episode_env(env)
            if target is not None:
                target.mkdir(parents=True, exist_ok=True)
                write_reward_curve(target / "reward_curve.csv", rows)
                write_sac_diagnostics(target / "sac_diagnostics.csv", diagnostics)
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            write_reward_curve(target / "reward_curve.csv", rows)
            write_sac_diagnostics(target / "sac_diagnostics.csv", diagnostics)
            try:
                self.policy.torch.save(self.policy.sac_state_dict(), target / "masac_policy.pt")
            except Exception:
                pass
        return rows

    def _episode_env(self, episode: int) -> SemanticTopologyMARLEnv:
        if self.env_factory is None:
            return self.env
        return self.env_factory(int(episode))

    def _close_episode_env(self, env: SemanticTopologyMARLEnv) -> None:
        if self.env_factory is not None and self.close_env is not None:
            self.close_env(env)


def write_reward_curve(path: str | Path, rows: List[TrainingMetrics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(TrainingMetrics(0, 0, 0.0, 0.0, 0, 0, 0, 0).to_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def write_sac_diagnostics(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))


def masac_update_policy(
    policy: MASACPolicy,
    replay: ReplayBuffer,
    batch_size: int = 128,
    updates: int = 1,
    gamma: float = 0.99,
    tau: float = 0.005,
    max_grad_norm: float = 1.0,
    reward_scale: float = 1.0,
    reward_transform: Optional[Callable[[float], float]] = None,
    sample_strategy: str = "uniform",
    update_actor: bool = True,
) -> Dict[str, float]:
    if len(replay) <= 0:
        return {}
    torch = policy.torch
    metrics: Dict[str, float] = {}
    for _ in range(max(1, int(updates))):
        update_start = time.perf_counter()
        batch = replay.sample(batch_size, strategy=sample_strategy)
        sample_end = time.perf_counter()
        q1_items = []
        q2_items = []
        target_items = []
        update_rewards = []
        dense_rewards = []
        terminal_credits = []
        reward_clip_positive = 0
        reward_clip_negative = 0
        selected_actions = 0
        for transition in batch:
            action_filter = transition.action_filter()
            selected = policy.masac_selected_q_values(
                transition.observations,
                transition.actions,
                action_filter=action_filter,
                detach_encoder=False,
            )
            with torch.no_grad():
                if transition.next_action_filter:
                    next_value, _next_entropy, _next_count = policy.masac_soft_state_value(
                        transition.next_observations,
                        target=True,
                        detach_encoder=True,
                        action_filter=transition.next_action_filter,
                    )
                else:
                    next_value = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
                done = 1.0 if transition.done else 0.0
                if reward_transform is not None:
                    dense_component = float(reward_transform(float(transition.dense_reward)))
                    update_reward = dense_component + float(transition.terminal_credit)
                else:
                    update_reward = float(transition.reward)
                normalizer_obj = getattr(reward_transform, "__self__", None) if reward_transform is not None else None
                clip_value = float(getattr(normalizer_obj, "clip", 0.0) or 0.0)
                if clip_value > 0.0 and update_reward >= clip_value:
                    reward_clip_positive += 1
                if clip_value > 0.0 and update_reward <= -clip_value:
                    reward_clip_negative += 1
                target = float(reward_scale) * update_reward + float(gamma) * (1.0 - done) * next_value
            q1_items.append(selected["q1"])
            q2_items.append(selected["q2"])
            target_items.append(target.reshape(()))
            update_rewards.append(update_reward)
            dense_rewards.append(float(transition.dense_reward))
            terminal_credits.append(float(transition.terminal_credit))
            selected_actions += int(selected.get("action_count", 0) or 0)
        valid_q_samples = len(q1_items)
        if valid_q_samples != len(batch):
            raise RuntimeError(f"per-action MASAC critic expected {len(batch)} valid Q samples, got {valid_q_samples}")
        critic_eval_end = time.perf_counter()
        q1_values = torch.stack([item.reshape(()) for item in q1_items])
        q2_values = torch.stack([item.reshape(()) for item in q2_items])
        targets = torch.stack([item.reshape(()) for item in target_items]).detach()
        critic_loss = torch.nn.functional.mse_loss(q1_values, targets) + torch.nn.functional.mse_loss(q2_values, targets)
        policy.q_optimizer.zero_grad()
        critic_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(policy.q_train_parameters(), float(max_grad_norm))
        policy.q_optimizer.step()
        critic_backward_end = time.perf_counter()

        actor_losses = []
        entropy_values = []
        target_entropy_values = []
        actor_candidate_sets = 0
        if update_actor:
            for transition in batch:
                actor_loss, entropy, target_entropy, count = policy.masac_actor_loss(
                    transition.observations,
                    action_filter=transition.action_filter(),
                )
                if int(count) <= 0:
                    raise RuntimeError(f"per-action MASAC actor found no decision slot for {transition.decision_key}")
                actor_losses.append(actor_loss.reshape(()))
                entropy_values.append(entropy.reshape(()))
                target_entropy_values.append(target_entropy.reshape(()))
                actor_candidate_sets += int(count)
        actor_eval_end = time.perf_counter()
        if update_actor and actor_losses:
            actor_loss = torch.stack(actor_losses).mean()
            entropy_mean = torch.stack(entropy_values).mean()
            target_entropy_mean = torch.stack(target_entropy_values).mean()
            policy.optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(policy.actor_head_parameters(), float(max_grad_norm))
            policy.optimizer.step()
            alpha_loss, alpha_grad_norm = policy.update_alpha(
                entropy_mean,
                target_entropy_mean,
                max_grad_norm=float(max_grad_norm),
            )
        else:
            actor_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            entropy_mean = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            target_entropy_mean = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            actor_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            alpha_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            alpha_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
        actor_backward_end = time.perf_counter()
        policy.soft_update_targets(float(tau))
        target_update_end = time.perf_counter()
        with torch.no_grad():
            td_error = (q1_values - targets).abs().mean()
            q_min_mean = torch.minimum(q1_values, q2_values).mean()
        metrics = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "target_entropy": float(target_entropy_mean.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": float(policy.alpha_tensor.detach().cpu().item()),
            "alpha_grad_norm": float(alpha_grad_norm.detach().cpu().item()),
            "auto_alpha": float(1.0 if getattr(policy, "auto_alpha", False) else 0.0),
            "td_error": float(td_error.detach().cpu().item()),
            "q_min_mean": float(q_min_mean.detach().cpu().item()),
            "q_target_mean": float(targets.mean().detach().cpu().item()),
            "update_reward_mean": float(sum(update_rewards) / max(1, len(update_rewards))),
            "update_reward_min": float(min(update_rewards) if update_rewards else 0.0),
            "update_reward_max": float(max(update_rewards) if update_rewards else 0.0),
            "dense_reward_mean": _mean_float(dense_rewards),
            "dense_reward_min": float(min(dense_rewards) if dense_rewards else 0.0),
            "dense_reward_max": float(max(dense_rewards) if dense_rewards else 0.0),
            "terminal_credit_mean": _mean_float(terminal_credits),
            "terminal_credit_min": float(min(terminal_credits) if terminal_credits else 0.0),
            "terminal_credit_max": float(max(terminal_credits) if terminal_credits else 0.0),
            "reward_clip_positive_ratio": float(reward_clip_positive / max(1, len(update_rewards))),
            "reward_clip_negative_ratio": float(reward_clip_negative / max(1, len(update_rewards))),
            "q_grad_norm": float(q_grad_norm.detach().cpu().item()),
            "actor_grad_norm": float(actor_grad_norm.detach().cpu().item()),
            "selected_actions": float(selected_actions),
            "valid_q_samples": float(valid_q_samples),
            "valid_q_sample_ratio": float(valid_q_samples / max(1, len(batch))),
            "actor_candidate_sets": float(actor_candidate_sets),
            "actor_updated": float(1.0 if update_actor and actor_losses else 0.0),
            "sample_s": float(sample_end - update_start),
            "critic_eval_s": float(critic_eval_end - sample_end),
            "critic_backward_s": float(critic_backward_end - critic_eval_end),
            "actor_eval_s": float(actor_eval_end - critic_backward_end),
            "actor_backward_s": float(actor_backward_end - actor_eval_end),
            "target_update_s": float(target_update_end - actor_backward_end),
            "update_total_s": float(target_update_end - update_start),
            "updates": float(metrics.get("updates", 0.0) + 1.0),
        }
    return metrics
