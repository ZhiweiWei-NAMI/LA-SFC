from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .iql_policy import IQLPolicy
from .marl_env import SemanticTopologyMARLEnv
from .marl_reward import reward_config_from_env
from .marl_trainer import (
    ReplayBuffer,
    RunningRewardNormalizer,
    SACTransition,
    TrainingMetrics,
    add_per_action_transitions,
    apply_stage_credits,
    apply_terminal_credits,
    build_per_action_transitions,
    write_reward_curve,
)


class IQLTrainer:
    """Online replay-buffer IQL trainer aligned with the MASAC transition path."""

    def __init__(
        self,
        env: SemanticTopologyMARLEnv,
        policy: IQLPolicy,
        env_factory: Optional[Callable[[int], SemanticTopologyMARLEnv]] = None,
        close_env: Optional[Callable[[SemanticTopologyMARLEnv], None]] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        replay_capacity: int = 10000,
        replay_warmup_steps: int = 64,
        update_interval: int = 1,
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
                last_transition_by_decision: Dict[str, SACTransition] = {}
                step_diagnostic_totals = {
                    "env_action_count": 0.0,
                    "replay_transitions_added": 0.0,
                    "no_action_steps_skipped": 0.0,
                    "orphan_terminal_credit_count": 0.0,
                    "orphan_stage_credit_count": 0.0,
                }
                for step in range(int(max_steps)):
                    current = observations
                    policy_step = self.policy.act_with_logprobs(current, deterministic=False, track_grad=False)
                    actions = policy_step.actions
                    observations, rewards, done, info = env.step(actions)
                    mean_reward = sum(rewards.values()) / max(1, len(rewards))
                    total += mean_reward
                    transitions, transition_metrics = build_per_action_transitions(
                        current,
                        actions,
                        policy_step.decision_contexts,
                        observations,
                        bool(done or step + 1 >= int(max_steps)),
                        int(episode),
                        scenario_name,
                        reward_config_from_env(env),
                        source="iql_online",
                    )
                    add_per_action_transitions(
                        self.replay,
                        transitions,
                        last_transition_by_chain,
                        last_transition_by_decision,
                        reward_normalizer=self.reward_normalizer,
                    )
                    stage_metrics = apply_stage_credits(
                        info.get("stage_events", []) or [],
                        last_transition_by_decision,
                        self.replay,
                        reward_config_from_env(env),
                    )
                    terminal_metrics = apply_terminal_credits(
                        info.get("terminal_events", []) or [],
                        last_transition_by_chain,
                        self.replay,
                        reward_normalizer=self.reward_normalizer,
                    )
                    for metrics_source in (transition_metrics, stage_metrics, terminal_metrics):
                        for key in step_diagnostic_totals:
                            step_diagnostic_totals[key] += float(metrics_source.get(key, 0.0) or 0.0)
                    self.replay.mark_env_step()
                    if self.replay.should_update(self.replay_warmup_steps, self.update_interval):
                        metrics = iql_update_policy(
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
                                "orphan_stage_credit_count": 0.0,
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
                                    "batch_size": self.batch_size,
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
                _write_diagnostics(target / "iql_diagnostics.csv", diagnostics)
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            write_reward_curve(target / "reward_curve.csv", rows)
            _write_diagnostics(target / "iql_diagnostics.csv", diagnostics)
            self.policy.torch.save(self.policy.iql_state_dict(), target / "iql_policy.pt")
        return rows

    def _episode_env(self, episode: int) -> SemanticTopologyMARLEnv:
        if self.env_factory is None:
            return self.env
        return self.env_factory(int(episode))

    def _close_episode_env(self, env: SemanticTopologyMARLEnv) -> None:
        if self.env_factory is not None and self.close_env is not None:
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
    reward_transform: Optional[Callable[[float], float]] = None,
    sample_strategy: str = "uniform",
) -> Dict[str, float]:
    if len(replay) <= 0:
        return {}
    torch = policy.torch
    metrics: Dict[str, float] = {}
    for _ in range(max(1, int(updates))):
        batch = replay.sample(batch_size, strategy=sample_strategy)
        v_losses = []
        q1_items = []
        q2_items = []
        target_items = []
        actor_losses = []
        weights = []
        valid_q_samples = 0
        update_rewards: List[float] = []
        dense_rewards: List[float] = []
        stage_credits: List[float] = []
        terminal_credits: List[float] = []
        reward_clip_positive = 0
        reward_clip_negative = 0

        for transition in batch:
            action_filter = transition.action_filter()
            for item in policy._masac_candidate_items(
                transition.observations,
                target=False,
                detach_encoder=True,
                action_filter=action_filter,
                action_context=transition.action_context(),
            ):
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

            selected = policy.masac_selected_q_values(
                transition.observations,
                transition.actions,
                action_filter=action_filter,
                action_context=transition.action_context(),
                detach_encoder=True,
            )
            with torch.no_grad():
                next_filter = transition.next_action_filter if transition.next_action_filter else None
                next_v = (
                    policy.iql_state_value(
                        transition.next_observations,
                        detach_encoder=True,
                        action_filter=next_filter,
                        action_context=transition.next_action_context,
                    )
                    if next_filter
                    else torch.tensor(0.0, dtype=torch.float32, device=policy.device)
                )
                done = 1.0 if transition.done else 0.0
                if reward_transform is not None:
                    dense_component = float(reward_transform(float(transition.dense_reward)))
                    update_reward = dense_component + float(transition.stage_credit) + float(transition.terminal_credit)
                else:
                    update_reward = float(transition.reward)
                normalizer_obj = getattr(reward_transform, "__self__", None) if reward_transform is not None else None
                clip_value = float(getattr(normalizer_obj, "clip", 0.0) or 0.0)
                if clip_value > 0.0 and update_reward >= clip_value:
                    reward_clip_positive += 1
                if clip_value > 0.0 and update_reward <= -clip_value:
                    reward_clip_negative += 1
                target = float(reward_scale) * update_reward + float(gamma) * (1.0 - done) * next_v
            q1_items.append(selected["q1"])
            q2_items.append(selected["q2"])
            target_items.append(target.reshape(()))
            valid_q_samples += 1
            update_rewards.append(update_reward)
            dense_rewards.append(float(transition.dense_reward))
            stage_credits.append(float(transition.stage_credit))
            terminal_credits.append(float(transition.terminal_credit))

            evaluation = policy.evaluate_actions(
                transition.observations,
                transition.actions,
                action_filter=action_filter,
                action_context=transition.action_context(),
            )
            if evaluation is not None:
                with torch.no_grad():
                    v_current = policy.iql_state_value(
                        transition.observations,
                        detach_encoder=True,
                        action_filter=action_filter,
                        action_context=transition.action_context(),
                    )
                    q_selected = torch.minimum(selected["q1"], selected["q2"]).detach()
                    advantage = q_selected - v_current.detach()
                    weight = torch.exp(advantage / max(1e-6, float(policy.beta))).clamp(max=100.0)
                actor_losses.append(-weight * evaluation.log_prob_tensor.reshape(()))
                weights.append(weight.reshape(()))
        if valid_q_samples != len(batch):
            raise RuntimeError(f"per-action IQL expected {len(batch)} valid Q samples, got {valid_q_samples}")

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
            "valid_q_samples": float(valid_q_samples),
            "valid_q_sample_ratio": float(valid_q_samples / max(1, len(batch))),
            "update_reward_mean": _mean(update_rewards),
            "dense_reward_mean": _mean(dense_rewards),
            "dense_reward_min": min(dense_rewards) if dense_rewards else 0.0,
            "dense_reward_max": max(dense_rewards) if dense_rewards else 0.0,
            "stage_credit_mean": _mean(stage_credits),
            "stage_credit_min": min(stage_credits) if stage_credits else 0.0,
            "stage_credit_max": max(stage_credits) if stage_credits else 0.0,
            "terminal_credit_mean": _mean(terminal_credits),
            "terminal_credit_min": min(terminal_credits) if terminal_credits else 0.0,
            "terminal_credit_max": max(terminal_credits) if terminal_credits else 0.0,
            "reward_clip_positive_ratio": float(reward_clip_positive / max(1, len(batch))),
            "reward_clip_negative_ratio": float(reward_clip_negative / max(1, len(batch))),
        }
    return metrics


def _mean(values: List[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _actor_parameters(policy: IQLPolicy) -> List[Any]:
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
