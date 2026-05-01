from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .marl_env import SemanticTopologyMARLEnv
from .marl_policy import BaseMARLPolicy, IPPOPolicy


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

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PPORolloutStep:
    observations: Mapping[str, Mapping[str, Any]]
    actions: Mapping[str, Any]
    reward: float
    old_log_prob: float
    old_value: float
    done: bool


class HeuristicEvaluator:
    def __init__(self, env: SemanticTopologyMARLEnv, policy: BaseMARLPolicy):
        self.env = env
        self.policy = policy

    def run(self, episodes: int = 1, max_steps: int = 100) -> List[TrainingMetrics]:
        rows: List[TrainingMetrics] = []
        for episode in range(int(episodes)):
            observations = self.env.reset()
            total = 0.0
            for step in range(int(max_steps)):
                actions = self.policy.act(observations, deterministic=True)
                observations, rewards, done, info = self.env.step(actions)
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
                    )
                )
                if done:
                    break
        return rows


class IPPOTrainer:
    """Compact PPO trainer for the IPPOPolicy core implementation.

    It keeps the experiment loop small while using clipped PPO, GAE, entropy
    regularization, mini-batches, and multi-epoch rollout updates.
    """

    def __init__(
        self,
        env: SemanticTopologyMARLEnv,
        policy: IPPOPolicy,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.env = env
        self.policy = policy
        self.gamma = float(gamma)
        self.clip_eps = float(clip_eps)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)

    def train(self, episodes: int = 10, max_steps: int = 100, output_dir: Optional[str] = None) -> List[TrainingMetrics]:
        rows: List[TrainingMetrics] = []
        for episode in range(int(episodes)):
            observations = self.env.reset()
            total = 0.0
            episode_steps: List[PPORolloutStep] = []
            for step in range(int(max_steps)):
                current_observations = observations
                policy_step = self.policy.act_with_logprobs(current_observations, deterministic=False, track_grad=False)
                observations, rewards, done, info = self.env.step(policy_step.actions)
                mean_reward = sum(rewards.values()) / max(1, len(rewards))
                total += mean_reward
                step_log_prob = _sum_tensor(policy_step.log_prob_tensors)
                step_value = _mean_tensor(policy_step.value_tensors)
                if step_log_prob is not None and step_value is not None:
                    episode_steps.append(
                        PPORolloutStep(
                            observations=current_observations,
                            actions=policy_step.actions,
                            reward=float(mean_reward),
                            old_log_prob=_tensor_float(step_log_prob),
                            old_value=_tensor_float(step_value),
                            done=bool(done),
                        )
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
                    )
                )
                if done:
                    break
            self.update_policy(episode_steps)
        if output_dir is not None:
            self.write_outputs(output_dir, rows)
        return rows

    def update_policy(self, episode_steps: Sequence[PPORolloutStep]) -> None:
        ppo_update_policy(
            self.policy,
            episode_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_eps=self.clip_eps,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
        )

    def write_outputs(self, output_dir: str, rows: List[TrainingMetrics]) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        write_reward_curve(target / "reward_curve.csv", rows)
        try:
            self.policy.torch.save(self.policy.model.state_dict(), target / "ippo_policy.pt")
        except Exception:
            pass
        self.env.write_traces(str(target))

    def _flatten(self, observation: Mapping[str, Any]) -> np.ndarray:
        from .graph_observation import flatten_observation

        return flatten_observation(observation)


def write_reward_curve(path: str | Path, rows: List[TrainingMetrics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(TrainingMetrics(0, 0, 0.0, 0.0, 0, 0, 0, 0).to_dict().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def ppo_update_policy(
    policy: IPPOPolicy,
    rollout: Sequence[PPORolloutStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    max_grad_norm: float = 0.5,
    prior_l2_coef: float = 0.0,
) -> Dict[str, float]:
    if not rollout:
        return {}
    torch = policy.torch
    device = getattr(policy, "device", torch.device("cpu"))
    rewards = torch.tensor([step.reward for step in rollout], dtype=torch.float32, device=device)
    old_values = torch.tensor([step.old_value for step in rollout], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor([step.old_log_prob for step in rollout], dtype=torch.float32, device=device)
    dones = torch.tensor([1.0 if step.done else 0.0 for step in rollout], dtype=torch.float32, device=device)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    for index in reversed(range(len(rollout))):
        next_value = old_values[index + 1] if index + 1 < len(rollout) else torch.tensor(0.0, dtype=torch.float32, device=device)
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + float(gamma) * next_value * nonterminal - old_values[index]
        last_gae = delta + float(gamma) * float(gae_lambda) * nonterminal * last_gae
        advantages[index] = last_gae
    returns = advantages + old_values
    if len(rollout) > 1:
        std = advantages.std(unbiased=False)
        if float(std.item()) > 1e-6:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    batch_size = max(1, min(int(minibatch_size), len(rollout)))
    update_epochs = max(1, int(update_epochs))
    metrics: Dict[str, float] = {}
    for _epoch in range(update_epochs):
        permutation = torch.randperm(len(rollout), device=device)
        for start in range(0, len(rollout), batch_size):
            batch_indices = permutation[start : start + batch_size]
            evaluations = []
            valid_indices = []
            for raw_index in batch_indices.tolist():
                evaluation = policy.evaluate_actions(rollout[raw_index].observations, rollout[raw_index].actions)
                if evaluation is not None:
                    evaluations.append(evaluation)
                    valid_indices.append(raw_index)
            if not evaluations:
                continue
            index_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
            new_log_probs = torch.stack([item.log_prob_tensor.reshape(()) for item in evaluations])
            new_values = torch.stack([item.value_tensor.reshape(()) for item in evaluations])
            entropy = torch.stack([item.entropy_tensor.reshape(()) for item in evaluations])

            batch_old_log_probs = old_log_probs[index_tensor]
            batch_advantages = advantages[index_tensor]
            batch_returns = returns[index_tensor]
            batch_old_values = old_values[index_tensor]
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            unclipped_actor = ratio * batch_advantages
            clipped_actor = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * batch_advantages
            actor_loss = -torch.min(unclipped_actor, clipped_actor).mean()

            value_pred_clipped = batch_old_values + (new_values - batch_old_values).clamp(-float(clip_eps), float(clip_eps))
            value_loss_unclipped = (new_values - batch_returns).pow(2)
            value_loss_clipped = (value_pred_clipped - batch_returns).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            entropy_bonus = entropy.mean()
            loss = actor_loss + float(value_coef) * critic_loss - float(entropy_coef) * entropy_bonus
            prior_l2 = None
            if float(prior_l2_coef) > 0.0 and hasattr(policy, "prior_regularization_loss"):
                prior_l2 = policy.prior_regularization_loss()
                if prior_l2 is not None:
                    loss = loss + float(prior_l2_coef) * prior_l2

            policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), float(max_grad_norm))
            policy.optimizer.step()
            metrics = {
                "loss": float(loss.detach().cpu().item()),
                "actor_loss": float(actor_loss.detach().cpu().item()),
                "critic_loss": float(critic_loss.detach().cpu().item()),
                "entropy": float(entropy_bonus.detach().cpu().item()),
                "prior_l2": float(prior_l2.detach().cpu().item()) if prior_l2 is not None else 0.0,
            }
    return metrics


def _sum_tensor(items: Sequence[Any]) -> Optional[Any]:
    if not items:
        return None
    return sum(item.reshape(()) for item in items)


def _mean_tensor(items: List[Any]) -> Optional[Any]:
    if not items:
        return None
    return sum(item.reshape(()) for item in items) / float(len(items))


def _tensor_float(item: Any) -> float:
    return float(item.detach().cpu().item())
