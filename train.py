from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from augment.transforms import AugmentConfig, TradingAugmenter
from algos import PPOLossConfig
from algos import PPOTrainer, PPOTrainerConfig
from datasets import DataPipelineConfig, build_trading_envs_from_csv
from envs import TradingEnvConfig
from models import ModelConfig, PPOSharedTransformerActorCritic


@dataclass(frozen=True)
class TrainConfig:
    csv_path: str
    output_dir: str = "outputs/test"

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    num_iterations: int = 200
    eval_every: int = 5

    deterministic_eval: bool = True

    history_days: int = 60
    sigma_days: int = 20
    multi_day_return_days: int = 7
    rows_per_day: int = 24
    train_frac: float = 0.70
    val_frac: float = 0.15

    alpha: float = 0.001
    lambda_risk: float = 0.1
    init_value: float = 1.0
    init_w: float = 0.0
    sigma_is_std: bool = True

    rollout_steps: int = 512
    ppo_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    normalize_advantages: bool = True
    advantage_eps: float = 1e-8
    value_loss_type: str = "mse"

    p_jitter: float = 0.8
    p_feature_dropout: float = 0.5
    p_time_mask: float = 0.5
    p_scale: float = 0.5

    jitter_std: float = 0.01
    feature_dropout_prob: float = 0.10
    max_time_mask_frac: float = 0.15
    scale_range: float = 0.05
    leave_last_n: int = 1

    augment_portfolio: bool = False
    portfolio_jitter_std: float = 0.002

    aug_policy_coef: float = 1.0
    aug_value_coef: float = 0.25
    detach_aug_ref: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_to_tensors(
    obs: Mapping[str, np.ndarray], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(obs["x"], dtype=torch.float32, device=device).unsqueeze(0)
    p = torch.as_tensor(obs["p"], dtype=torch.float32, device=device).unsqueeze(0)
    return x, p


def compute_max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = 1.0 - (values / np.clip(running_max, 1e-12, None))
    return float(np.max(drawdowns))


@torch.no_grad()
def evaluate_policy(
    model: PPOSharedTransformerActorCritic,
    env: Any,
    device: torch.device,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    obs, info = env.reset(seed=seed, options={"random_start": False})

    rewards = []
    values = [float(info["V"])]
    weights = []
    turnovers = []
    market_returns = []
    sigma_series = []

    done = False
    while not done:
        x, p = obs_to_tensors(obs, device)
        logits, value = model(x, p)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

        next_obs, reward, terminated, truncated, step_info = env.step(
            int(action.item())
        )
        done = bool(terminated or truncated)

        rewards.append(float(reward))
        values.append(float(step_info["V_next"]))
        weights.append(float(step_info["w_next"]))
        turnovers.append(abs(float(step_info["delta_w"])))
        market_returns.append(float(step_info["r_mkt_log"]))
        sigma_series.append(float(step_info["sigma"]))

        obs = next_obs

    reward_arr = np.asarray(rewards, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    turnover_arr = np.asarray(turnovers, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    mkt_arr = np.asarray(market_returns, dtype=np.float64)
    sigma_arr = np.asarray(sigma_series, dtype=np.float64)

    stats = {
        "episode_length": float(len(rewards)),
        "reward_sum": float(reward_arr.sum()) if len(reward_arr) > 0 else 0.0,
        "reward_mean": float(reward_arr.mean()) if len(reward_arr) > 0 else 0.0,
        "final_value": float(value_arr[-1]) if len(value_arr) > 0 else 0.0,
        "max_drawdown": compute_max_drawdown(value_arr),
        "avg_weight": float(weight_arr.mean()) if len(weight_arr) > 0 else 0.0,
        "avg_turnover": float(turnover_arr.mean()) if len(turnover_arr) > 0 else 0.0,
        "mean_r_mkt_log": float(mkt_arr.mean()) if len(mkt_arr) > 0 else 0.0,
        "mean_sigma": float(sigma_arr.mean()) if len(sigma_arr) > 0 else 0.0,
    }

    return stats


def prefix_keys(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def save_checkpoint(
    path: Path,
    model: PPOSharedTransformerActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    data_meta: Dict[str, Any],
    feature_cols: Tuple[str, ...],
) -> None:
    payload = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_config": asdict(train_cfg),
        "model_config": asdict(model_cfg),
        "data_meta": data_meta,
        "feature_cols": list(feature_cols),
    }
    torch.save(payload, path)


def build_objects(cfg: TrainConfig) -> Dict[str, Any]:
    window_rows = cfg.history_days * cfg.rows_per_day

    data_cfg = DataPipelineConfig(
        csv_path=cfg.csv_path,
        history_days=cfg.history_days,
        sigma_days=cfg.sigma_days,
        multi_day_return_days=cfg.multi_day_return_days,
        rows_per_day=cfg.rows_per_day,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )

    env_cfg = TradingEnvConfig(
        window_len=window_rows,
        alpha=cfg.alpha,
        lambda_risk=cfg.lambda_risk,
        init_value=cfg.init_value,
        init_w=cfg.init_w,
        sigma_is_std=cfg.sigma_is_std,
    )

    bundle = build_trading_envs_from_csv(data_cfg, env_cfg)
    train_env = bundle["train_env"]
    val_env = bundle["val_env"]
    test_env = bundle["test_env"]
    feature_cols = tuple(bundle["feature_cols"])
    feature_groups = bundle["feature_groups"]
    data_meta = bundle["meta"]

    if data_meta["window_rows"] != window_rows:
        raise ValueError(
            f"Pipeline returned window_rows={data_meta['window_rows']}, but expected {window_rows}."
        )

    model_cfg = ModelConfig(
        seq_feature_dim=len(feature_cols),
        portfolio_dim=3,
        action_dim=len(train_env.action_grid),
        max_seq_len=window_rows,
    )
    model = PPOSharedTransformerActorCritic(model_cfg)

    aug_cfg = AugmentConfig(
        p_jitter=cfg.p_jitter,
        p_feature_dropout=cfg.p_feature_dropout,
        p_time_mask=cfg.p_time_mask,
        p_scale=cfg.p_scale,
        jitter_std=cfg.jitter_std,
        feature_dropout_prob=cfg.feature_dropout_prob,
        max_time_mask_frac=cfg.max_time_mask_frac,
        scale_range=cfg.scale_range,
        leave_last_n=cfg.leave_last_n,
        augment_portfolio=cfg.augment_portfolio,
        portfolio_jitter_std=cfg.portfolio_jitter_std,
    )
    augmenter = TradingAugmenter(aug_cfg, feature_groups)

    loss_cfg = PPOLossConfig(
        clip_eps=cfg.clip_eps,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        normalize_advantages=cfg.normalize_advantages,
        advantage_eps=cfg.advantage_eps,
        value_loss_type=cfg.value_loss_type,
        aug_policy_coef=cfg.aug_policy_coef,
        aug_value_coef=cfg.aug_value_coef,
        detach_aug_ref=cfg.detach_aug_ref,
    )

    trainer_cfg = PPOTrainerConfig(
        rollout_steps=cfg.rollout_steps,
        ppo_epochs=cfg.ppo_epochs,
        minibatch_size=cfg.minibatch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
    )

    trainer = PPOTrainer(
        model=model,
        env=train_env,
        loss_cfg=loss_cfg,
        trainer_cfg=trainer_cfg,
        augmenter=augmenter,
    )

    return {
        "bundle": bundle,
        "train_env": train_env,
        "val_env": val_env,
        "test_env": test_env,
        "feature_cols": feature_cols,
        "data_meta": data_meta,
        "model_cfg": model_cfg,
        "model": model,
        "augmenter": augmenter,
        "trainer": trainer,
    }


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    objs = build_objects(cfg)
    train_env = objs["train_env"]
    val_env = objs["val_env"]
    test_env = objs["test_env"]
    feature_cols = objs["feature_cols"]
    data_meta = objs["data_meta"]
    model_cfg = objs["model_cfg"]
    model = objs["model"]
    trainer = objs["trainer"]

    device = torch.device(cfg.device)
    model.to(device)

    print("=" * 80)
    print("Training setup")
    print(f"device            : {cfg.device}")
    print(f"csv_path          : {cfg.csv_path}")
    print(f"output_dir        : {cfg.output_dir}")
    print(f"feature_dim       : {len(feature_cols)}")
    print(f"portfolio_dim     : 3")
    print(f"action_dim        : {len(train_env.action_grid)}")
    print(f"history_days      : {cfg.history_days}")
    print(f"rows_per_day      : {cfg.rows_per_day}")
    print(f"window_rows       : {data_meta['window_rows']}")
    print(f"train_env_T       : {data_meta['train_env_T']}")
    print(f"val_env_T         : {data_meta['val_env_T']}")
    print(f"test_env_T        : {data_meta['test_env_T']}")
    print(f"num_iterations    : {cfg.num_iterations}")
    print("=" * 80)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "data_meta.json", "w", encoding="utf-8") as f:
        json.dump(data_meta, f, indent=2)
    with open(out_dir / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(list(feature_cols), f, indent=2)

    records = []
    best_val_final_value = -float("inf")
    best_iter = -1

    for iteration in range(1, cfg.num_iterations + 1):
        train_stats = trainer.train_iteration()
        row: Dict[str, float] = {"iteration": float(iteration)}
        row.update(prefix_keys(train_stats, "train/"))

        do_eval = (
            (iteration % cfg.eval_every == 0)
            or (iteration == 1)
            or (iteration == cfg.num_iterations)
        )

        if do_eval:
            val_stats = evaluate_policy(
                model=model,
                env=val_env,
                device=device,
                deterministic=cfg.deterministic_eval,
                seed=cfg.seed,
            )
            row.update(prefix_keys(val_stats, "val/"))

            if val_stats["final_value"] > best_val_final_value:
                best_val_final_value = val_stats["final_value"]
                best_iter = iteration

                save_checkpoint(
                    path=out_dir / "best_model.pt",
                    model=model,
                    optimizer=trainer.optimizer,
                    iteration=iteration,
                    train_cfg=cfg,
                    model_cfg=model_cfg,
                    data_meta=data_meta,
                    feature_cols=feature_cols,
                )

                best_test_stats = evaluate_policy(
                    model=model,
                    env=test_env,
                    device=device,
                    deterministic=cfg.deterministic_eval,
                    seed=cfg.seed,
                )
                row.update(prefix_keys(best_test_stats, "test_at_best/"))

        records.append(row)

        save_checkpoint(
            path=checkpoints_dir / f"iter_{iteration:04d}.pt",
            model=model,
            optimizer=trainer.optimizer,
            iteration=iteration,
            train_cfg=cfg,
            model_cfg=model_cfg,
            data_meta=data_meta,
            feature_cols=feature_cols,
        )

        pd.DataFrame(records).to_csv(out_dir / "metrics.csv", index=False)

        msg = (
            f"[iter {iteration:04d}] "
            f"train_loss={train_stats.get('loss_total', float('nan')):.4f} "
            f"train_reward={train_stats.get('rollout_mean_reward', float('nan')):.6f}"
        )
        if do_eval:
            msg += (
                f" | val_final_V={row.get('val/final_value', float('nan')):.6f}"
                f" val_mdd={row.get('val/max_drawdown', float('nan')):.4f}"
            )
        print(msg)

    best_path = out_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        final_test_stats = evaluate_policy(
            model=model,
            env=test_env,
            device=device,
            deterministic=cfg.deterministic_eval,
            seed=cfg.seed,
        )
        final_summary = {
            "best_iteration": best_iter,
            "best_val_final_value": best_val_final_value,
            **prefix_keys(final_test_stats, "best_model_test/"),
        }
    else:
        final_summary = {
            "best_iteration": -1,
            "best_val_final_value": float("nan"),
        }

    with open(out_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    pd.DataFrame(records).to_csv(out_dir / "metrics.csv", index=False)

    print("=" * 80)
    print("Training finished")
    print(f"Best iteration      : {best_iter}")
    print(f"Best val final value: {best_val_final_value:.6f}")
    print(f"Artifacts saved to  : {out_dir}")
    print("=" * 80)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train PPO + augmentation for ETH hourly trading."
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/ppo_eth")
    parser.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=5)

    parser.add_argument("--history_days", type=int, default=60)
    parser.add_argument("--sigma_days", type=int, default=20)
    parser.add_argument("--multi_day_return_days", type=int, default=7)
    parser.add_argument("--rows_per_day", type=int, default=24)

    parser.add_argument("--rollout_steps", type=int, default=512)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    return TrainConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        num_iterations=args.num_iterations,
        eval_every=args.eval_every,
        history_days=args.history_days,
        sigma_days=args.sigma_days,
        multi_day_return_days=args.multi_day_return_days,
        rows_per_day=args.rows_per_day,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
