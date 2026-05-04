from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .iql_policy import IQLPolicy
from .marl_env import SemanticTopologyMARLEnv
from .marl_policy import BaseMARLPolicy, policy_from_name
from .marl_trainer import HeuristicEvaluator, ReplayBuffer, SACTransition, TrainingMetrics, write_reward_curve


class IQLTrainer:
    """Offline IQL trainer over a fixed behavior-policy dataset."""

    def __init__(
        self,
        env: SemanticTopologyMARLEnv,
        policy: IQLPolicy,
        behavior_policy: Optional[BaseMARLPolicy] = None,
        env_factory: Optional[Callable[[int], SemanticTopologyMARLEnv]] = None,
        eval_env_factory: Optional[Callable[[], SemanticTopologyMARLEnv]] = None,
        close_env: Optional[Callable[[SemanticTopologyMARLEnv], None]] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        replay_capacity: int = 10000,
        offline_updates: int = 0,
        updates_per_transition: float = 1.0,
        max_grad_norm: float = 1.0,
        reward_scale: float = 1.0,
        seed: int = 0,
    ):
        self.env = env
        self.env_factory = env_factory
        self.eval_env_factory = eval_env_factory
        self.close_env = close_env
        self.policy = policy
        self.behavior_policy = behavior_policy or policy_from_name("utility_prior_with_exchange", seed=seed)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.offline_updates = int(offline_updates)
        self.updates_per_transition = float(updates_per_transition)
        self.max_grad_norm = float(max_grad_norm)
        self.reward_scale = float(reward_scale)
        self.replay = ReplayBuffer(replay_capacity, seed=seed)

    def train(self, episodes: int = 10, max_steps: int = 100, output_dir: Optional[str] = None) -> List[TrainingMetrics]:
        behavior_rows: List[TrainingMetrics] = []
        diagnostics: List[Dict[str, Any]] = []
        for episode in range(int(episodes)):
            env = self._behavior_env(episode)
            try:
                observations = env.reset()
                total = 0.0
                for step in range(int(max_steps)):
                    current = observations
                    actions = self.behavior_policy.act(current, deterministic=True)
                    observations, rewards, done, info = env.step(actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    self.replay.add(
                        SACTransition(
                            observations=current,
                            actions=actions,
                            reward=float(mean_reward),
                            next_observations=observations,
                            done=bool(done or step + 1 >= int(max_steps)),
                            episode=int(episode),
                            source="offline_behavior",
                        )
                    )
                    summary = info.get("summary", {})
                    behavior_rows.append(
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
            finally:
                self._close_episode_env(env)
            if output_dir is not None:
                target = Path(output_dir)
                target.mkdir(parents=True, exist_ok=True)
                write_reward_curve(target / "iql_behavior_reward_curve.csv", behavior_rows)
        update_count = self.offline_updates
        if update_count <= 0:
            update_count = max(1, int(round(len(self.replay) * self.updates_per_transition)))
        for update_index in range(update_count):
            metrics = iql_update_policy(
                self.policy,
                self.replay,
                batch_size=self.batch_size,
                updates=1,
                gamma=self.gamma,
                tau=self.tau,
                max_grad_norm=self.max_grad_norm,
                reward_scale=self.reward_scale,
            )
            if metrics:
                diagnostics.append({"update_index": update_index + 1, "replay_size": len(self.replay), **metrics})
        eval_env = self._eval_env()
        try:
            eval_rows = HeuristicEvaluator(eval_env, self.policy).run(episodes=1, max_steps=max_steps)
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                eval_env.write_traces(str(output_dir))
        finally:
            self._close_eval_env(eval_env)
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            write_reward_curve(target / "iql_behavior_reward_curve.csv", behavior_rows)
            write_reward_curve(target / "iql_eval_reward_curve.csv", eval_rows)
            write_reward_curve(target / "reward_curve.csv", eval_rows)
            _write_diagnostics(target / "iql_diagnostics.csv", diagnostics)
            self.policy.torch.save(self.policy.iql_state_dict(), target / "iql_policy.pt")
        return eval_rows

    def _behavior_env(self, episode: int) -> SemanticTopologyMARLEnv:
        if self.env_factory is None:
            return self.env
        return self.env_factory(int(episode))

    def _eval_env(self) -> SemanticTopologyMARLEnv:
        if self.eval_env_factory is not None:
            return self.eval_env_factory()
        if self.env_factory is not None:
            return self.env_factory(900000)
        return self.env

    def _close_episode_env(self, env: SemanticTopologyMARLEnv) -> None:
        if self.env_factory is not None and self.close_env is not None:
            self.close_env(env)

    def _close_eval_env(self, env: SemanticTopologyMARLEnv) -> None:
        if (self.env_factory is not None or self.eval_env_factory is not None) and self.close_env is not None:
            self.close_env(env)


def expectile_loss(diff: Any, expectile: float) -> Any:
    positive_weight = diff.new_full(diff.shape, float(expectile))
    negative_weight = diff.new_full(diff.shape, 1.0 - float(expectile))
    weight = positive_weight.where(diff > 0.0, negative_weight)
    return (weight * diff.pow(2)).mean()


def iql_update_policy(
    policy: IQLPolicy,
    replay: ReplayBuffer,
    batch_size: int = 128,
    updates: int = 1,
    gamma: float = 0.99,
    tau: float = 0.005,
    max_grad_norm: float = 1.0,
    reward_scale: float = 1.0,
) -> Dict[str, float]:
    if len(replay) <= 0:
        return {}
    torch = policy.torch
    metrics: Dict[str, float] = {}
    for _ in range(max(1, int(updates))):
        batch = replay.sample(batch_size)
        v_losses = []
        q1_items = []
        q2_items = []
        target_items = []
        actor_losses = []
        weights = []

        for transition in batch:
            for item in policy._masac_candidate_items(transition.observations, target=False, detach_encoder=True):
                q1, q2 = policy._masac_joint_q_values(
                    item["global_context"],
                    item["local_context"],
                    item["candidate_features"],
                    target=False,
                    detach_encoder=True,
                )
                if q1.numel() <= 0:
                    continue
                q_data = torch.minimum(q1, q2).detach().reshape(-1).max()
                v_pred = policy.v_net(policy.iql_v_input(item)).squeeze(-1)
                v_losses.append(expectile_loss(q_data - v_pred, policy.expectile))

            selected = policy.masac_selected_q_values(transition.observations, transition.actions, detach_encoder=True)
            if selected is not None:
                with torch.no_grad():
                    next_v = policy.iql_state_value(transition.next_observations, detach_encoder=True)
                    done = 1.0 if transition.done else 0.0
                    target = float(reward_scale) * float(transition.reward) + float(gamma) * (1.0 - done) * next_v
                q1_items.append(selected["q1"])
                q2_items.append(selected["q2"])
                target_items.append(target.reshape(()))

                evaluation = policy.evaluate_actions(transition.observations, transition.actions)
                if evaluation is not None:
                    with torch.no_grad():
                        v_current = policy.iql_state_value(transition.observations, detach_encoder=True)
                        q_selected = torch.minimum(selected["q1"], selected["q2"]).detach()
                        advantage = q_selected - v_current.detach()
                        weight = torch.exp(advantage / max(1e-6, float(policy.beta))).clamp(max=100.0)
                    actor_losses.append(-weight * evaluation.log_prob_tensor.reshape(()))
                    weights.append(weight.reshape(()))

        if v_losses:
            v_loss = torch.stack([item.reshape(()) for item in v_losses]).mean()
            policy.v_optimizer.zero_grad()
            v_loss.backward()
            v_grad_norm = torch.nn.utils.clip_grad_norm_(policy.v_net.parameters(), float(max_grad_norm))
            policy.v_optimizer.step()
        else:
            v_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            v_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=policy.device)

        if q1_items:
            q1_values = torch.stack([item.reshape(()) for item in q1_items])
            q2_values = torch.stack([item.reshape(()) for item in q2_items])
            targets = torch.stack([item.reshape(()) for item in target_items]).detach()
            q_loss = torch.nn.functional.mse_loss(q1_values, targets) + torch.nn.functional.mse_loss(q2_values, targets)
            policy.q_optimizer.zero_grad()
            q_loss.backward()
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(policy.q1.parameters()) + list(policy.q2.parameters()),
                float(max_grad_norm),
            )
            policy.q_optimizer.step()
        else:
            q_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            q_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            targets = torch.tensor([0.0], dtype=torch.float32, device=policy.device)

        if actor_losses:
            actor_loss = torch.stack([item.reshape(()) for item in actor_losses]).mean()
            policy.optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(_actor_parameters(policy), float(max_grad_norm))
            policy.optimizer.step()
            weight_mean = torch.stack([item.reshape(()) for item in weights]).mean()
        else:
            actor_loss = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            actor_grad_norm = torch.tensor(0.0, dtype=torch.float32, device=policy.device)
            weight_mean = torch.tensor(0.0, dtype=torch.float32, device=policy.device)

        policy.soft_update_targets(float(tau))
        metrics = {
            "v_loss": float(v_loss.detach().cpu().item()),
            "q_loss": float(q_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "adv_weight_mean": float(weight_mean.detach().cpu().item()),
            "v_grad_norm": float(v_grad_norm.detach().cpu().item()),
            "q_grad_norm": float(q_grad_norm.detach().cpu().item()),
            "actor_grad_norm": float(actor_grad_norm.detach().cpu().item()),
            "q_target_mean": float(targets.mean().detach().cpu().item()),
        }
    return metrics


def _actor_parameters(policy: IQLPolicy) -> List[Any]:
    params = list(policy.model.parameters())
    if policy.semantic_scorer is not None:
        params.extend(list(policy.semantic_scorer.parameters()))
    return params


def _write_diagnostics(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
