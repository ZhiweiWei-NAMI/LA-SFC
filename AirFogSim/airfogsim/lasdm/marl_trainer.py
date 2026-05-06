from __future__ import annotations

import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .marl_env import SemanticTopologyMARLEnv
from .marl_policy import BaseMARLPolicy, MASACPolicy


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
    episode: int = 0
    scenario: str = ""
    source: str = "policy"


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, seed: int = 0):
        self.capacity = max(1, int(capacity))
        self._items: deque[SACTransition] = deque(maxlen=self.capacity)
        self._rng = random.Random(int(seed))

    def add(self, transition: SACTransition) -> None:
        self._items.append(transition)

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
                for step in range(int(max_steps)):
                    current_observations = observations
                    policy_step = self.policy.act_with_logprobs(current_observations, deterministic=False, track_grad=False)
                    observations, rewards, done, info = env.step(policy_step.actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    self.replay.add(
                        SACTransition(
                            observations=current_observations,
                            actions=policy_step.actions,
                            reward=float(mean_reward),
                            next_observations=observations,
                            done=bool(done or step + 1 >= int(max_steps)),
                            episode=int(episode),
                            scenario=scenario_name,
                        )
                    )
                    if len(self.replay) >= max(1, self.replay_warmup_steps) and len(self.replay) % self.update_interval == 0:
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
                            sample_strategy=self.replay_sample_strategy,
                            update_actor=(next_update_index % self.actor_update_interval == 0),
                        )
                        if metrics:
                            update_index += 1
                            diagnostics.append(
                                {
                                    "episode": int(episode),
                                    "step": int(step),
                                    "scenario": scenario_name,
                                    "update_index": update_index,
                                    "replay_size": len(self.replay),
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
        selected_actions = 0
        for transition in batch:
            selected = policy.masac_selected_q_values(transition.observations, transition.actions, detach_encoder=False)
            if selected is None:
                continue
            with torch.no_grad():
                next_value, _next_entropy, _next_count = policy.masac_soft_state_value(
                    transition.next_observations,
                    target=True,
                    detach_encoder=True,
                )
                done = 1.0 if transition.done else 0.0
                target = float(reward_scale) * float(transition.reward) + float(gamma) * (1.0 - done) * next_value
            q1_items.append(selected["q1"])
            q2_items.append(selected["q2"])
            target_items.append(target.reshape(()))
            selected_actions += int(selected.get("action_count", 0) or 0)
        if not q1_items:
            continue
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
                actor_loss, entropy, target_entropy, count = policy.masac_actor_loss(transition.observations)
                if int(count) <= 0:
                    continue
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
            alpha_loss = policy.update_alpha(entropy_mean, target_entropy_mean)
        else:
            actor_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            entropy_mean = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            target_entropy_mean = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            actor_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            alpha_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
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
            "auto_alpha": float(1.0 if getattr(policy, "auto_alpha", False) else 0.0),
            "td_error": float(td_error.detach().cpu().item()),
            "q_min_mean": float(q_min_mean.detach().cpu().item()),
            "q_target_mean": float(targets.mean().detach().cpu().item()),
            "q_grad_norm": float(q_grad_norm.detach().cpu().item()),
            "actor_grad_norm": float(actor_grad_norm.detach().cpu().item()),
            "selected_actions": float(selected_actions),
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
