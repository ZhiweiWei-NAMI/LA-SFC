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
    episode: int = 0
    action_filter: Optional[Mapping[str, str]] = None


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
        actor_loss_coef: float = 1.0,
        value_coef: float = 1.0,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        prior_l2_coef: float = 0.0,
        target_kl: float = 0.02,
        rollout_episodes_per_update: int = 1,
        max_grad_norm: float = 0.5,
        normalize_returns: bool = True,
        return_norm_momentum: float = 0.95,
        return_norm_eps: float = 1e-6,
        entropy_coef_start: Optional[float] = None,
        entropy_coef_end: Optional[float] = None,
        entropy_decay_episodes: int = 0,
        per_function_rollout_samples: bool = False,
        share_reward_across_function_samples: bool = True,
        per_function_reward_mode: str = "shared",
    ):
        self.env = env
        self.policy = policy
        self.gamma = float(gamma)
        self.clip_eps = float(clip_eps)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.actor_loss_coef = float(actor_loss_coef)
        self.value_coef = float(value_coef)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.prior_l2_coef = float(prior_l2_coef)
        self.target_kl = float(target_kl)
        self.rollout_episodes_per_update = max(1, int(rollout_episodes_per_update))
        self.max_grad_norm = float(max_grad_norm)
        self.normalize_returns = bool(normalize_returns)
        self.return_norm_momentum = float(return_norm_momentum)
        self.return_norm_eps = float(return_norm_eps)
        self.entropy_coef_start = None if entropy_coef_start is None else float(entropy_coef_start)
        self.entropy_coef_end = None if entropy_coef_end is None else float(entropy_coef_end)
        self.entropy_decay_episodes = max(0, int(entropy_decay_episodes))
        self.per_function_rollout_samples = bool(per_function_rollout_samples)
        self.share_reward_across_function_samples = bool(share_reward_across_function_samples)
        self.per_function_reward_mode = str(per_function_reward_mode or "shared")

    def train(self, episodes: int = 10, max_steps: int = 100, output_dir: Optional[str] = None) -> List[TrainingMetrics]:
        rows: List[TrainingMetrics] = []
        diagnostics: List[Dict[str, Any]] = []
        rollout_buffer: List[PPORolloutStep] = []
        rollout_episode_count = 0
        update_index = 0
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
                episode_steps.extend(
                    rollout_steps_from_policy_step(
                        self.policy,
                        current_observations,
                        policy_step.actions,
                        mean_reward=float(mean_reward),
                        done=bool(done or step + 1 >= int(max_steps)),
                        episode=int(episode),
                        per_function_samples=self.per_function_rollout_samples,
                        share_reward_across_functions=self.share_reward_across_function_samples,
                        per_function_reward_mode=self.per_function_reward_mode,
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
            if episode_steps:
                rollout_buffer.extend(episode_steps)
                rollout_episode_count += 1
            should_update = rollout_episode_count >= self.rollout_episodes_per_update or episode == int(episodes) - 1
            if should_update and rollout_buffer:
                metrics = self.update_policy(rollout_buffer, episode=episode)
                update_index += 1
                diagnostics.append(
                    {
                        "episode": int(episode),
                        "update_index": update_index,
                        "rollout_episodes": rollout_episode_count,
                        "rollout_steps": len(rollout_buffer),
                        **metrics,
                    }
                )
                rollout_buffer = []
                rollout_episode_count = 0
        if output_dir is not None:
            self.write_outputs(output_dir, rows)
            write_ppo_diagnostics(Path(output_dir) / "ppo_diagnostics.csv", diagnostics)
        return rows

    def update_policy(self, episode_steps: Sequence[PPORolloutStep], episode: int = 0) -> Dict[str, float]:
        return ppo_update_policy(
            self.policy,
            episode_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_eps=self.clip_eps,
            entropy_coef=self._entropy_coef_for_episode(episode),
            actor_loss_coef=self.actor_loss_coef,
            value_coef=self.value_coef,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            max_grad_norm=self.max_grad_norm,
            prior_l2_coef=self.prior_l2_coef,
            target_kl=self.target_kl,
            normalize_returns=self.normalize_returns,
            return_norm_momentum=self.return_norm_momentum,
            return_norm_eps=self.return_norm_eps,
        )

    def _entropy_coef_for_episode(self, episode: int) -> float:
        if self.entropy_coef_start is None or self.entropy_coef_end is None or self.entropy_decay_episodes <= 0:
            return self.entropy_coef
        progress = max(0.0, min(1.0, float(episode) / float(self.entropy_decay_episodes)))
        return self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress

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


def write_ppo_diagnostics(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))


def rollout_steps_from_policy_step(
    policy: IPPOPolicy,
    observations: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, Any],
    mean_reward: float,
    done: bool,
    episode: int,
    per_function_samples: bool = False,
    share_reward_across_functions: bool = True,
    per_function_reward_mode: str = "shared",
    reward_aux: Optional[Mapping[str, Any]] = None,
) -> List[PPORolloutStep]:
    torch = policy.torch
    if not per_function_samples:
        with torch.no_grad():
            evaluation = policy.evaluate_actions(observations, actions)
        if evaluation is None:
            return []
        return [
            PPORolloutStep(
                observations=observations,
                actions=actions,
                reward=float(mean_reward),
                old_log_prob=_tensor_float(evaluation.log_prob_tensor),
                old_value=_tensor_float(evaluation.value_tensor),
                done=bool(done),
                episode=int(episode),
            )
        ]
    scopes = _action_scopes(actions)
    if not scopes:
        return []
    base_reward = float(mean_reward) / float(len(scopes)) if bool(share_reward_across_functions) else float(mean_reward)
    rollout_steps: List[PPORolloutStep] = []
    with torch.no_grad():
        for scope in scopes:
            evaluation = policy.evaluate_actions(observations, actions, action_filter=scope)
            if evaluation is None:
                continue
            rollout_steps.append(
                PPORolloutStep(
                    observations=observations,
                    actions=actions,
                    reward=base_reward,
                    old_log_prob=_tensor_float(evaluation.log_prob_tensor),
                    old_value=_tensor_float(evaluation.value_tensor),
                    done=bool(done),
                    episode=int(episode),
                    action_filter=dict(scope),
                )
            )
    return rollout_steps


def _action_scopes(actions: Mapping[str, Any]) -> List[Dict[str, str]]:
    scopes: List[Dict[str, str]] = []
    for agent_id, agent_payload in sorted(dict(actions or {}).items(), key=lambda item: str(item[0])):
        if not isinstance(agent_payload, Mapping):
            continue
        for sfc_id, assignments in sorted(dict(agent_payload).items(), key=lambda item: str(item[0])):
            if not isinstance(assignments, Mapping):
                continue
            for sfc_node_id in sorted(str(item) for item in assignments.keys()):
                scopes.append({"agent_id": str(agent_id), "sfc_id": str(sfc_id), "sfc_node_id": str(sfc_node_id)})
    return scopes


def ppo_update_policy(
    policy: IPPOPolicy,
    rollout: Sequence[PPORolloutStep],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    actor_loss_coef: float = 1.0,
    value_coef: float = 1.0,
    update_epochs: int = 4,
    minibatch_size: int = 64,
    max_grad_norm: float = 0.5,
    prior_l2_coef: float = 0.0,
    target_kl: float = 0.02,
    normalize_returns: bool = True,
    return_norm_momentum: float = 0.95,
    return_norm_eps: float = 1e-6,
) -> Dict[str, float]:
    if not rollout:
        return {}
    torch = policy.torch
    device = getattr(policy, "device", torch.device("cpu"))
    rewards = torch.tensor([step.reward for step in rollout], dtype=torch.float32, device=device)
    old_values = torch.tensor([step.old_value for step in rollout], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor([step.old_log_prob for step in rollout], dtype=torch.float32, device=device)
    dones = torch.tensor([1.0 if step.done else 0.0 for step in rollout], dtype=torch.float32, device=device)
    raw_advantage_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
    raw_advantage_std = torch.tensor(0.0, dtype=torch.float32, device=device)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, dtype=torch.float32, device=device)
    for index in reversed(range(len(rollout))):
        next_value = old_values[index + 1] if index + 1 < len(rollout) else torch.tensor(0.0, dtype=torch.float32, device=device)
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + float(gamma) * next_value * nonterminal - old_values[index]
        last_gae = delta + float(gamma) * float(gae_lambda) * nonterminal * last_gae
        advantages[index] = last_gae
    returns = advantages + old_values
    return_batch_mean = returns.mean()
    return_batch_std = returns.std(unbiased=False) if len(rollout) > 1 else torch.tensor(0.0, dtype=torch.float32, device=device)
    return_norm_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
    return_norm_std = torch.tensor(1.0, dtype=torch.float32, device=device)
    if bool(normalize_returns):
        return_norm_mean, return_norm_std = _update_return_normalizer(
            policy,
            returns,
            momentum=float(return_norm_momentum),
            eps=float(return_norm_eps),
        )
    if len(rollout) > 0:
        raw_advantage_mean = advantages.mean()
        raw_advantage_std = advantages.std(unbiased=False) if len(rollout) > 1 else torch.tensor(0.0, dtype=torch.float32, device=device)
    if len(rollout) > 1:
        std = advantages.std(unbiased=False)
        if float(std.item()) > 1e-6:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    batch_size = max(1, min(int(minibatch_size), len(rollout)))
    update_epochs = max(1, int(update_epochs))
    target_kl_value = max(0.0, float(target_kl or 0.0))
    metrics: Dict[str, float] = {
        "loss": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "actor_loss_coef": float(actor_loss_coef),
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "prior_l2": 0.0,
        "mean_advantage": float(raw_advantage_mean.detach().cpu().item()),
        "std_advantage": float(raw_advantage_std.detach().cpu().item()),
        "return_batch_mean": float(return_batch_mean.detach().cpu().item()),
        "return_batch_std": float(return_batch_std.detach().cpu().item()),
        "return_norm_mean": float(return_norm_mean.detach().cpu().item()),
        "return_norm_std": float(return_norm_std.detach().cpu().item()),
        "return_norm_enabled": 1.0 if bool(normalize_returns) else 0.0,
        "updates": 0.0,
        "epochs_completed": 0.0,
        "target_kl": target_kl_value,
        "stopped_early": 0.0,
    }
    stop_early = False
    for epoch in range(update_epochs):
        permutation = torch.randperm(len(rollout), device=device)
        for start in range(0, len(rollout), batch_size):
            batch_indices = permutation[start : start + batch_size]
            evaluations = []
            valid_indices = []
            for raw_index in batch_indices.tolist():
                evaluation = policy.evaluate_actions(
                    rollout[raw_index].observations,
                    rollout[raw_index].actions,
                    action_filter=rollout[raw_index].action_filter,
                )
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
            log_ratio = new_log_probs - batch_old_log_probs
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > float(clip_eps)).float().mean()
            unclipped_actor = ratio * batch_advantages
            clipped_actor = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * batch_advantages
            actor_loss = -torch.min(unclipped_actor, clipped_actor).mean()

            scaled_new_values = (new_values - return_norm_mean) / return_norm_std
            scaled_returns = (batch_returns - return_norm_mean) / return_norm_std
            scaled_old_values = (batch_old_values - return_norm_mean) / return_norm_std
            value_pred_clipped = scaled_old_values + (scaled_new_values - scaled_old_values).clamp(-float(clip_eps), float(clip_eps))
            value_loss_unclipped = (scaled_new_values - scaled_returns).pow(2)
            value_loss_clipped = (value_pred_clipped - scaled_returns).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            entropy_bonus = entropy.mean()
            loss = float(actor_loss_coef) * actor_loss + float(value_coef) * critic_loss - float(entropy_coef) * entropy_bonus
            prior_l2 = None
            if float(prior_l2_coef) > 0.0 and hasattr(policy, "prior_regularization_loss"):
                prior_l2 = policy.prior_regularization_loss()
                if prior_l2 is not None:
                    loss = loss + float(prior_l2_coef) * prior_l2

            policy.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), float(max_grad_norm))
            policy.optimizer.step()
            with torch.no_grad():
                post_evaluations = []
                post_old_log_probs = []
                for raw_index in valid_indices:
                    post_eval = policy.evaluate_actions(
                        rollout[raw_index].observations,
                        rollout[raw_index].actions,
                        action_filter=rollout[raw_index].action_filter,
                    )
                    if post_eval is not None:
                        post_evaluations.append(post_eval)
                        post_old_log_probs.append(old_log_probs[raw_index])
                if post_evaluations:
                    post_log_probs = torch.stack([item.log_prob_tensor.reshape(()) for item in post_evaluations])
                    post_old = torch.stack([item.reshape(()) for item in post_old_log_probs])
                    post_ratio = torch.exp(post_log_probs - post_old)
                    post_log_ratio = post_log_probs - post_old
                    approx_kl = ((post_ratio - 1.0) - post_log_ratio).mean()
                    clip_fraction = ((post_ratio - 1.0).abs() > float(clip_eps)).float().mean()
            metrics = {
                "loss": float(loss.detach().cpu().item()),
                "actor_loss": float(actor_loss.detach().cpu().item()),
                "critic_loss": float(critic_loss.detach().cpu().item()),
                "entropy": float(entropy_bonus.detach().cpu().item()),
                "actor_loss_coef": float(actor_loss_coef),
                "approx_kl": float(approx_kl.detach().cpu().item()),
                "clip_fraction": float(clip_fraction.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "prior_l2": float(prior_l2.detach().cpu().item()) if prior_l2 is not None else 0.0,
                "mean_advantage": float(raw_advantage_mean.detach().cpu().item()),
                "std_advantage": float(raw_advantage_std.detach().cpu().item()),
                "return_batch_mean": float(return_batch_mean.detach().cpu().item()),
                "return_batch_std": float(return_batch_std.detach().cpu().item()),
                "return_norm_mean": float(return_norm_mean.detach().cpu().item()),
                "return_norm_std": float(return_norm_std.detach().cpu().item()),
                "return_norm_enabled": 1.0 if bool(normalize_returns) else 0.0,
                "updates": float(metrics.get("updates", 0.0) + 1.0),
                "epochs_completed": float(epoch + 1),
                "target_kl": target_kl_value,
                "stopped_early": 0.0,
            }
            if target_kl_value > 0.0 and float(approx_kl.detach().cpu().item()) > 1.5 * target_kl_value:
                metrics["stopped_early"] = 1.0
                stop_early = True
                break
        if stop_early:
            break
    return metrics


def _update_return_normalizer(policy: IPPOPolicy, returns: Any, momentum: float = 0.95, eps: float = 1e-6) -> Any:
    """Track raw return scale while keeping critic outputs in raw reward units."""

    torch = policy.torch
    device = returns.device
    eps_value = max(float(eps), 1e-8)
    momentum_value = max(0.0, min(0.9999, float(momentum)))
    batch_mean = returns.detach().mean()
    batch_var = returns.detach().var(unbiased=False) if len(returns) > 1 else torch.tensor(0.0, dtype=torch.float32, device=device)
    batch_var = batch_var.clamp_min(eps_value * eps_value)
    initialized = bool(getattr(policy, "_ppo_return_norm_initialized", False))
    if not initialized:
        mean = batch_mean
        var = batch_var
        policy._ppo_return_norm_initialized = True
    else:
        old_mean = getattr(policy, "_ppo_return_norm_mean").to(device=device)
        old_var = getattr(policy, "_ppo_return_norm_var").to(device=device)
        delta = batch_mean - old_mean
        mean = momentum_value * old_mean + (1.0 - momentum_value) * batch_mean
        var = momentum_value * old_var + (1.0 - momentum_value) * batch_var + momentum_value * (1.0 - momentum_value) * delta.pow(2)
        var = var.clamp_min(eps_value * eps_value)
    policy._ppo_return_norm_mean = mean.detach()
    policy._ppo_return_norm_var = var.detach()
    return policy._ppo_return_norm_mean.to(device=device), torch.sqrt(policy._ppo_return_norm_var.to(device=device)).clamp_min(eps_value)


def _mean_tensor(items: List[Any]) -> Optional[Any]:
    if not items:
        return None
    return sum(item.reshape(()) for item in items) / float(len(items))


def _tensor_float(item: Any) -> float:
    return float(item.detach().cpu().item())
