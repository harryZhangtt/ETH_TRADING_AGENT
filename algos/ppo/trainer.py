from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .losses import PPOLossConfig, compute_ppo_aug_loss


@dataclass(frozen=True)
class PPOTrainerConfig:
    rollout_steps: int = 512
    ppo_epochs: int = 10
    minibatch_size: int = 64

    gamma: float = 0.99
    gae_lambda: float = 0.95

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    device: str = "cpu"


def _obs_to_tensors(
    obs: Mapping[str, np.ndarray], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    if "x" not in obs or "p" not in obs:
        raise KeyError("Observation must contain keys 'x' and 'p'.")

    x = torch.as_tensor(obs["x"], dtype=torch.float32, device=device).unsqueeze(0)
    p = torch.as_tensor(obs["p"], dtype=torch.float32, device=device).unsqueeze(0)

    if x.ndim != 3:
        raise ValueError(f"obs['x'] must become [1,T,F], got {tuple(x.shape)}")
    if p.ndim != 2:
        raise ValueError(f"obs['p'] must become [1,P], got {tuple(p.shape)}")

    return x, p


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, dones must all have shape [N]")

    N = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(N)):
        if t == N - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def _safe_tensor_std(x: torch.Tensor) -> float:
    if x.numel() <= 1:
        return 0.0
    return float(x.std(unbiased=False).item())


def _safe_np_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _safe_np_last(xs: List[float]) -> float:
    return float(xs[-1]) if xs else 0.0


class PPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        env: Any,
        loss_cfg: PPOLossConfig,
        trainer_cfg: PPOTrainerConfig = PPOTrainerConfig(),
        augmenter: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.model = model
        self.env = env
        self.loss_cfg = loss_cfg
        self.cfg = trainer_cfg
        self.augmenter = augmenter

        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        if self.augmenter is not None:
            self.augmenter.to(self.device)

        self.optimizer = optimizer or Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self._obs: Optional[Dict[str, np.ndarray]] = None
        self._info: Optional[Dict[str, Any]] = None

        self._episode_return: float = 0.0
        self._episode_length: int = 0

    def reset_env(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        self._info = info
        self._episode_return = 0.0
        self._episode_length = 0
        return obs

    @torch.no_grad()
    def _value_of_obs(self, obs: Mapping[str, np.ndarray]) -> torch.Tensor:
        x, p = _obs_to_tensors(obs, self.device)
        _, value = self.model(x, p)  # [1]
        return value.squeeze(0)

    @torch.no_grad()
    def collect_rollout(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        self.model.eval()
        if self.augmenter is not None:
            self.augmenter.eval()

        if self._obs is None:
            self.reset_env()

        xs: List[torch.Tensor] = []
        ps: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        old_logprobs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []

        finished_episode_returns: List[float] = []
        finished_episode_lengths: List[int] = []

        for _ in range(self.cfg.rollout_steps):
            assert self._obs is not None

            x, p = _obs_to_tensors(self._obs, self.device)

            action, logprob, value = self.model.act(x, p)
            action_i = int(action.item())

            next_obs, reward, terminated, truncated, info = self.env.step(action_i)
            done = bool(terminated or truncated)

            xs.append(x.squeeze(0))
            ps.append(p.squeeze(0))
            actions.append(action.squeeze(0).long())
            old_logprobs.append(logprob.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(
                torch.tensor(float(reward), dtype=torch.float32, device=self.device)
            )
            dones.append(
                torch.tensor(float(done), dtype=torch.float32, device=self.device)
            )

            self._episode_return += float(reward)
            self._episode_length += 1

            if done:
                finished_episode_returns.append(self._episode_return)
                finished_episode_lengths.append(self._episode_length)

                next_obs, next_info = self.env.reset()
                self._episode_return = 0.0
                self._episode_length = 0
                self._obs = next_obs
                self._info = next_info
            else:
                self._obs = next_obs
                self._info = info

        assert self._obs is not None
        last_value = self._value_of_obs(self._obs)

        x_batch = torch.stack(xs, dim=0)  # [N, T, F]
        p_batch = torch.stack(ps, dim=0)  # [N, P]
        action_batch = torch.stack(actions, dim=0)  # [N]
        old_logprob_batch = torch.stack(old_logprobs, dim=0)  # [N]
        value_batch = torch.stack(values, dim=0)  # [N]
        reward_batch = torch.stack(rewards, dim=0)  # [N]
        done_batch = torch.stack(dones, dim=0)  # [N]

        advantages, returns = _compute_gae(
            rewards=reward_batch,
            values=value_batch,
            dones=done_batch,
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        batch = {
            "x": x_batch,
            "p": p_batch,
            "actions": action_batch,
            "old_logprob": old_logprob_batch,
            "advantages": advantages,
            "returns": returns,
        }

        action_unique, action_counts = torch.unique(action_batch, return_counts=True)
        action_hist = {
            f"action_frac_{int(a.item())}": float(c.item())
            / float(action_batch.numel())
            for a, c in zip(action_unique, action_counts)
        }

        rollout_stats = {
            "rollout_steps": float(self.cfg.rollout_steps),
            "rollout_num_dones": float(done_batch.sum().item()),
            "rollout_done_frac": float(done_batch.float().mean().item()),
            "rollout_mean_reward": float(reward_batch.mean().item()),
            "rollout_reward_std": _safe_tensor_std(reward_batch),
            "rollout_reward_sum": float(reward_batch.sum().item()),
            "rollout_mean_value": float(value_batch.mean().item()),
            "rollout_value_std": _safe_tensor_std(value_batch),
            "rollout_last_value_bootstrap": float(last_value.item()),
            "rollout_adv_mean": float(advantages.mean().item()),
            "rollout_adv_std": _safe_tensor_std(advantages),
            "rollout_adv_abs_mean": float(advantages.abs().mean().item()),
            "rollout_return_mean": float(returns.mean().item()),
            "rollout_return_std": _safe_tensor_std(returns),
            "episode_return_mean": _safe_np_mean(finished_episode_returns),
            "episode_length_mean": _safe_np_mean(
                [float(x) for x in finished_episode_lengths]
            ),
            "episode_return_last": _safe_np_last(finished_episode_returns),
            "episode_length_last": _safe_np_last(
                [float(x) for x in finished_episode_lengths]
            ),
            "episode_return_running": float(self._episode_return),
            "episode_length_running": float(self._episode_length),
        }
        rollout_stats.update(action_hist)

        return batch, rollout_stats

    def _iterate_minibatches(self, batch: Dict[str, torch.Tensor]):
        N = batch["actions"].shape[0]
        mb = min(self.cfg.minibatch_size, N)
        indices = torch.randperm(N, device=self.device)

        for start in range(0, N, mb):
            end = min(start + mb, N)
            idx = indices[start:end]
            yield {k: v[idx] for k, v in batch.items()}

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run PPO updates on one collected rollout.
        """
        self.model.train()
        if self.augmenter is not None:
            self.augmenter.train()

        stat_accumulator: Dict[str, List[float]] = {}
        grad_norms: List[float] = []
        num_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for minibatch in self._iterate_minibatches(batch):
                self.optimizer.zero_grad(set_to_none=True)

                loss, stats = compute_ppo_aug_loss(
                    model=self.model,
                    batch=minibatch,
                    cfg=self.loss_cfg,
                    augmenter=self.augmenter,
                )

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                grad_norm_f = (
                    float(grad_norm.item())
                    if isinstance(grad_norm, torch.Tensor)
                    else float(grad_norm)
                )
                grad_norms.append(grad_norm_f)
                num_updates += 1

                stats_float = {k: float(v.item()) for k, v in stats.items()}
                for k, v in stats_float.items():
                    stat_accumulator.setdefault(k, []).append(v)

        mean_stats = {k: float(np.mean(v)) for k, v in stat_accumulator.items()}

        if grad_norms:
            mean_stats["grad_norm"] = float(np.mean(grad_norms))
            mean_stats["grad_norm_max"] = float(np.max(grad_norms))
            mean_stats["grad_norm_min"] = float(np.min(grad_norms))
        else:
            mean_stats["grad_norm"] = 0.0
            mean_stats["grad_norm_max"] = 0.0
            mean_stats["grad_norm_min"] = 0.0

        mean_stats["update_steps"] = float(num_updates)
        mean_stats["ppo_epochs"] = float(self.cfg.ppo_epochs)
        mean_stats["effective_minibatch_size"] = float(
            min(self.cfg.minibatch_size, batch["actions"].shape[0])
        )

        return mean_stats

    def train_iteration(self) -> Dict[str, float]:
        batch, rollout_stats = self.collect_rollout()
        update_stats = self.update(batch)

        out: Dict[str, float] = {}
        out.update(rollout_stats)
        out.update(update_stats)
        return out
