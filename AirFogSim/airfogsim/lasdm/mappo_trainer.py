from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .mappo_policy import MAPPOPolicy
from .marl_env import SemanticTopologyMARLEnv
from .marl_trainer import TrainingMetrics, write_reward_curve


@dataclass
class MAPPORolloutItem:
    observations: Mapping[str, Mapping[str, Any]]
    actions: Mapping[str, Any]
    old_log_prob: float
    value: float
    reward: float
    done: bool
    advantage: float = 0.0
    ret: float = 0.0


class MAPPORolloutBuffer:
    def __init__(self) -> None:
        self.items: List[MAPPORolloutItem] = []

    def add(self, item: MAPPORolloutItem) -> None:
        self.items.append(item)

    def clear(self) -> None:
        self.items.clear()

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> None:
        gae = 0.0
        for index in range(len(self.items) - 1, -1, -1):
            item = self.items[index]
            next_value = float(last_value) if index == len(self.items) - 1 else float(self.items[index + 1].value)
            not_done = 0.0 if item.done else 1.0
            delta = float(item.reward) + float(gamma) * next_value * not_done - float(item.value)
            gae = delta + float(gamma) * float(lam) * not_done * gae
            item.advantage = gae
            item.ret = gae + float(item.value)

    def __len__(self) -> int:
        return len(self.items)


class MAPPOTrainer:
    """On-policy PPO updates with decentralized execution and centralized value."""

    def __init__(
        self,
        env: SemanticTopologyMARLEnv,
        policy: MAPPOPolicy,
        env_factory: Optional[Callable[[int], SemanticTopologyMARLEnv]] = None,
        close_env: Optional[Callable[[SemanticTopologyMARLEnv], None]] = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        rollout_steps: int = 128,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.env = env
        self.env_factory = env_factory
        self.close_env = close_env
        self.policy = policy
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.rollout_steps = int(rollout_steps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)

    def train(self, episodes: int = 10, max_steps: int = 100, output_dir: Optional[str] = None) -> List[TrainingMetrics]:
        rows: List[TrainingMetrics] = []
        diagnostics: List[Dict[str, Any]] = []
        buffer = MAPPORolloutBuffer()
        update_index = 0
        target = Path(output_dir) if output_dir is not None else None
        for episode in range(int(episodes)):
            env = self._episode_env(episode)
            try:
                scenario_name = str(getattr(getattr(env, "config", None), "scenario_name", "") or "")
                observations = env.reset()
                total = 0.0
                for step in range(int(max_steps)):
                    current = observations
                    policy_step = self.policy.act_with_logprobs(current, deterministic=False, track_grad=False)
                    observations, rewards, done, info = env.step(policy_step.actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    buffer.add(
                        MAPPORolloutItem(
                            observations=current,
                            actions=policy_step.actions,
                            old_log_prob=_mean_tensor_value(policy_step.log_prob_tensors),
                            value=_mean_value(policy_step.values),
                            reward=float(mean_reward),
                            done=bool(done or step + 1 >= int(max_steps)),
                        )
                    )
                    if len(buffer) >= max(1, self.rollout_steps) or done or step + 1 >= int(max_steps):
                        last_value = 0.0 if done else self._value_estimate(observations)
                        buffer.compute_gae(last_value, self.gamma, self.lam)
                        update_index += 1
                        diagnostics.append(
                            {
                                "update_index": update_index,
                                "episode": int(episode),
                                "step": int(step),
                                "scenario": scenario_name,
                                **self._ppo_update(buffer),
                            }
                        )
                        buffer.clear()
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
                _write_diagnostics(target / "mappo_diagnostics.csv", diagnostics)
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            write_reward_curve(target / "reward_curve.csv", rows)
            _write_diagnostics(target / "mappo_diagnostics.csv", diagnostics)
            self.policy.torch.save(self.policy.mappo_state_dict(), target / "mappo_policy.pt")
        return rows

    def _episode_env(self, episode: int) -> SemanticTopologyMARLEnv:
        if self.env_factory is None:
            return self.env
        return self.env_factory(int(episode))

    def _close_episode_env(self, env: SemanticTopologyMARLEnv) -> None:
        if self.env_factory is not None and self.close_env is not None:
            self.close_env(env)

    def _ppo_update(self, buffer: MAPPORolloutBuffer) -> Dict[str, float]:
        if not buffer.items:
            return {}
        torch = self.policy.torch
        advantages = torch.tensor([item.advantage for item in buffer.items], dtype=torch.float32, device=self.policy.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        returns = torch.tensor([item.ret for item in buffer.items], dtype=torch.float32, device=self.policy.device)
        old_log_probs = torch.tensor([item.old_log_prob for item in buffer.items], dtype=torch.float32, device=self.policy.device)
        metrics: Dict[str, float] = {}
        for _epoch in range(max(1, self.ppo_epochs)):
            log_probs = []
            values = []
            entropies = []
            kept_advantages = []
            kept_returns = []
            kept_old_log_probs = []
            for idx, item in enumerate(buffer.items):
                evaluation = self.policy.evaluate_actions(item.observations, item.actions)
                if evaluation is None:
                    continue
                log_probs.append(evaluation.log_prob_tensor.reshape(()))
                values.append(evaluation.value_tensor.reshape(()))
                entropies.append(evaluation.entropy_tensor.reshape(()))
                kept_advantages.append(advantages[idx].reshape(()))
                kept_returns.append(returns[idx].reshape(()))
                kept_old_log_probs.append(old_log_probs[idx].reshape(()))
            if not log_probs:
                continue
            new_log_probs = torch.stack(log_probs)
            value_tensor = torch.stack(values)
            entropy_tensor = torch.stack(entropies)
            advantage_tensor = torch.stack(kept_advantages).detach()
            return_tensor = torch.stack(kept_returns).detach()
            old_log_prob_tensor = torch.stack(kept_old_log_probs).detach()
            ratio = torch.exp(new_log_probs - old_log_prob_tensor)
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage_tensor
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value_tensor - return_tensor).pow(2).mean()
            entropy = entropy_tensor.mean()
            total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            self.policy.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(_actor_parameters(self.policy), self.max_grad_norm)
            self.policy.optimizer.step()
            metrics = {
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": float(entropy.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "rollout_size": float(len(buffer.items)),
            }
        return metrics

    def _value_estimate(self, observations: Mapping[str, Mapping[str, Any]]) -> float:
        torch = self.policy.torch
        with torch.no_grad():
            value = self.policy._centralized_value_tensor(observations)
        return float(value.detach().cpu().item())


def _mean_tensor_value(items: List[Any]) -> float:
    if not items:
        return 0.0
    return float(sum(float(item.detach().cpu().item()) for item in items) / float(len(items)))


def _mean_value(values: Mapping[str, float]) -> float:
    if "__global__" in values:
        return float(values["__global__"])
    if not values:
        return 0.0
    return float(sum(float(item) for item in values.values()) / float(len(values)))


def _actor_parameters(policy: MAPPOPolicy) -> List[Any]:
    return list(policy.model.parameters())


def _write_diagnostics(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
