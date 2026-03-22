from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from augment.drac import policy_consistency_loss, value_consistency_loss


@dataclass(frozen=True)
class PPOLossConfig:
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    normalize_advantages: bool = True
    advantage_eps: float = 1e-8

    value_loss_type: str = "mse"  # "mse" or "smooth_l1"

    aug_policy_coef: float = 1.0
    aug_value_coef: float = 0.25
    detach_aug_ref: bool = True


def _require_batch_key(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    raise KeyError(f"None of the required keys were found in batch: {keys}")


def _extract_obs(batch: Mapping[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    if "obs" in batch:
        obs = batch["obs"]
        if not isinstance(obs, Mapping):
            raise TypeError(
                "batch['obs'] must be a mapping containing keys 'x' and 'p'"
            )
        x = obs["x"]
        p = obs["p"]
        return x, p

    x = _require_batch_key(batch, "x", "obs_x")
    p = _require_batch_key(batch, "p", "obs_p")
    return x, p


def _extract_actions(batch: Mapping[str, Any]) -> torch.Tensor:
    actions = _require_batch_key(batch, "actions", "action")
    if actions.ndim != 1:
        raise ValueError(f"actions must have shape [B], got {tuple(actions.shape)}")
    return actions.long()


def _extract_old_logprob(batch: Mapping[str, Any]) -> torch.Tensor:
    old_logprob = _require_batch_key(
        batch, "old_logprob", "old_logp", "logprob", "logp"
    )
    if old_logprob.ndim != 1:
        raise ValueError(
            f"old_logprob must have shape [B], got {tuple(old_logprob.shape)}"
        )
    return old_logprob


def _extract_advantages(batch: Mapping[str, Any]) -> torch.Tensor:
    advantages = _require_batch_key(batch, "advantages", "adv")
    if advantages.ndim != 1:
        raise ValueError(
            f"advantages must have shape [B], got {tuple(advantages.shape)}"
        )
    return advantages


def _extract_returns(batch: Mapping[str, Any]) -> torch.Tensor:
    returns = _require_batch_key(batch, "returns", "ret")
    if returns.ndim != 1:
        raise ValueError(f"returns must have shape [B], got {tuple(returns.shape)}")
    return returns


def _normalize_advantages(advantages: torch.Tensor, eps: float) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def _compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(values, returns)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(values, returns)
    raise ValueError(f"Unsupported value_loss_type: {loss_type}")


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.std(unbiased=False)


def _compute_explained_variance(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    var_y = torch.var(y_true, unbiased=False)
    if torch.isnan(var_y) or var_y.item() < 1e-12:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)
    return 1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y


def compute_ppo_aug_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    cfg: PPOLossConfig,
    augmenter: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x, p = _extract_obs(batch)
    actions = _extract_actions(batch)
    old_logprob = _extract_old_logprob(batch)
    advantages_raw = _extract_advantages(batch)
    returns = _extract_returns(batch)

    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, T, F], got {tuple(x.shape)}")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [B, P], got {tuple(p.shape)}")
    if x.shape[0] != p.shape[0]:
        raise ValueError(f"Batch mismatch: x has B={x.shape[0]}, p has B={p.shape[0]}")

    B = x.shape[0]
    for name, tensor in {
        "actions": actions,
        "old_logprob": old_logprob,
        "advantages": advantages_raw,
        "returns": returns,
    }.items():
        if tensor.shape[0] != B:
            raise ValueError(
                f"{name} batch size mismatch: expected {B}, got {tensor.shape[0]}"
            )

    advantages = advantages_raw
    if cfg.normalize_advantages:
        advantages = _normalize_advantages(advantages, eps=cfg.advantage_eps)

    logits, values = model(x, p)

    if logits.ndim != 2:
        raise ValueError(
            f"Model logits must have shape [B, A], got {tuple(logits.shape)}"
        )
    if values.ndim != 1:
        raise ValueError(f"Model values must have shape [B], got {tuple(values.shape)}")

    if logits.shape[0] != B or values.shape[0] != B:
        raise ValueError(
            f"Model output batch mismatch: expected B={B}, "
            f"got logits B={logits.shape[0]}, values B={values.shape[0]}"
        )

    dist = Categorical(logits=logits)
    logprob = dist.log_prob(actions)  # [B]
    entropy = dist.entropy()  # [B]

    log_ratio = logprob - old_logprob
    ratio = torch.exp(log_ratio)  # [B]

    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = _compute_value_loss(values, returns, loss_type=cfg.value_loss_type)
    entropy_bonus = entropy.mean()

    device = x.device
    aug_policy_loss = torch.zeros((), device=device)
    aug_value_loss = torch.zeros((), device=device)

    use_aug = augmenter is not None and (
        cfg.aug_policy_coef > 0.0 or cfg.aug_value_coef > 0.0
    )

    if use_aug:
        x_aug, p_aug = augmenter(x, p)
        aug_logits, aug_values = model(x_aug, p_aug)

        if cfg.aug_policy_coef > 0.0:
            aug_policy_loss = policy_consistency_loss(
                ref_logits=logits,
                aug_logits=aug_logits,
                detach_ref=cfg.detach_aug_ref,
                reduction="mean",
            )

        if cfg.aug_value_coef > 0.0:
            aug_value_loss = value_consistency_loss(
                ref_values=values,
                aug_values=aug_values,
                detach_ref=cfg.detach_aug_ref,
                reduction="mean",
            )

    total_loss = (
        policy_loss
        + cfg.value_coef * value_loss
        - cfg.entropy_coef * entropy_bonus
        + cfg.aug_policy_coef * aug_policy_loss
        + cfg.aug_value_coef * aug_value_loss
    )

    with torch.no_grad():
        # PPO diagnostics
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        sample_kl = (old_logprob - logprob).mean()
        clipfrac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

        # Entropy diagnostics
        action_dim = logits.shape[-1]
        if action_dim > 1:
            max_entropy = torch.log(
                torch.tensor(float(action_dim), device=device, dtype=logits.dtype)
            )
            normalized_entropy = entropy_bonus / max_entropy
        else:
            normalized_entropy = torch.zeros((), device=device, dtype=logits.dtype)

        # Advantage diagnostics
        adv_mean_raw = advantages_raw.mean()
        adv_std_raw = _safe_std(advantages_raw)
        adv_abs_mean_raw = advantages_raw.abs().mean()

        adv_mean_used = advantages.mean()
        adv_std_used = _safe_std(advantages)
        adv_abs_mean_used = advantages.abs().mean()

        # Value diagnostics
        explained_var = _compute_explained_variance(values, returns)
        value_pred_mean = values.mean()
        value_pred_std = _safe_std(values)
        return_mean = returns.mean()
        return_std = _safe_std(returns)

        # Ratio diagnostics
        ratio_mean = ratio.mean()
        ratio_std = _safe_std(ratio)
        ratio_min = ratio.min()
        ratio_max = ratio.max()

        # Surrogate diagnostics
        surr1_mean = surr1.mean()
        surr2_mean = surr2.mean()
        clipped_surr_mean = torch.min(surr1, surr2).mean()

        # Action diagnostics
        action_prob = torch.softmax(logits, dim=-1)
        chosen_action_prob = action_prob.gather(1, actions.unsqueeze(1)).squeeze(1)
        chosen_action_prob_mean = chosen_action_prob.mean()

    stats: Dict[str, torch.Tensor] = {
        # Main losses
        "loss_total": total_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_bonus.detach(),
        "aug_policy_loss": aug_policy_loss.detach(),
        "aug_value_loss": aug_value_loss.detach(),
        # PPO diagnostics
        "approx_kl": approx_kl.detach(),
        "sample_kl": sample_kl.detach(),
        "clipfrac": clipfrac.detach(),
        # Ratio diagnostics
        "ratio_mean": ratio_mean.detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio_min.detach(),
        "ratio_max": ratio_max.detach(),
        # Surrogate diagnostics
        "surr1_mean": surr1_mean.detach(),
        "surr2_mean": surr2_mean.detach(),
        "clipped_surr_mean": clipped_surr_mean.detach(),
        # Entropy diagnostics
        "normalized_entropy": normalized_entropy.detach(),
        # Value / return diagnostics
        "value_pred_mean": value_pred_mean.detach(),
        "value_pred_std": value_pred_std.detach(),
        "return_mean": return_mean.detach(),
        "return_std": return_std.detach(),
        "explained_variance": explained_var.detach(),
        # Advantage diagnostics
        "adv_mean_raw": adv_mean_raw.detach(),
        "adv_std_raw": adv_std_raw.detach(),
        "adv_abs_mean_raw": adv_abs_mean_raw.detach(),
        "adv_mean_used": adv_mean_used.detach(),
        "adv_std_used": adv_std_used.detach(),
        "adv_abs_mean_used": adv_abs_mean_used.detach(),
        # Chosen action confidence
        "chosen_action_prob_mean": chosen_action_prob_mean.detach(),
    }

    return total_loss, stats
