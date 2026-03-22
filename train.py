from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from augment.transforms import AugmentConfig, TradingAugmenter
from algos import PPOLossConfig
from algos import PPOTrainer, PPOTrainerConfig
from datasets import (
    DataPipelineConfig,
    build_feature_groups,
    fit_standardizer,
    load_actionable_df,
    make_env_from_slice,
)
from envs import TradingEnvConfig
from models import ModelConfig, PPOSharedTransformerActorCritic
from utils import TensorBoardConfig, TensorBoardLogger


@dataclass(frozen=True)
class TrainConfig:
    csv_path: str = "data/gap_repair/eth_metrics_midway_conservative_fill.csv"
    output_dir: str = "outputs/ppo_eth"

    seed: int = 42
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

    # Training schedule
    num_iterations: int = 200
    eval_every: int = 5
    deterministic_eval: bool = True

    # Data pipeline
    history_days: int = 60
    sigma_days: int = 20
    multi_day_return_days: int = 7
    rows_per_day: int = 24

    # Expanding walk-forward split
    min_train_days: int = 365
    val_days: int = 90
    test_days: int = 90
    step_days: int = 90

    # Env / reward
    alpha: float = 0.001
    lambda_risk: float = 0.1
    init_value: float = 1.0
    init_w: float = 0.0
    sigma_is_std: bool = True

    # PPO trainer
    rollout_steps: int = 512
    ppo_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # PPO loss
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    normalize_advantages: bool = True
    advantage_eps: float = 1e-8
    value_loss_type: str = "mse"

    # Augmentation
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

    # Validation selection score
    val_reward_coef: float = 1.0
    val_mdd_coef: float = 0.25
    val_turnover_coef: float = 0.05

    # Eval diagnostics
    trade_eps: float = 1e-8
    high_exposure_threshold: float = 0.75

    # TensorBoard
    enable_tensorboard: bool = True
    tensorboard_flush_every: int = 1


@dataclass(frozen=True)
class Fold:
    index: int
    train_range: Tuple[int, int]
    val_range: Tuple[int, int]
    test_range: Tuple[int, int]


def generate_folds(
    n: int,
    rows_per_day: int,
    min_train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
) -> List[Fold]:
    min_train = min_train_days * rows_per_day
    val_rows = val_days * rows_per_day
    test_rows = test_days * rows_per_day
    step = step_days * rows_per_day

    folds: List[Fold] = []
    train_end = min_train
    fold_idx = 0

    while train_end + val_rows + test_rows <= n:
        val_end = train_end + val_rows
        test_end = val_end + test_rows
        folds.append(
            Fold(
                index=fold_idx,
                train_range=(0, train_end),
                val_range=(train_end, val_end),
                test_range=(val_end, test_end),
            )
        )
        train_end += step
        fold_idx += 1

    return folds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_to_tensors(
    obs: Dict[str, np.ndarray], device: torch.device
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


def _np_mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size > 0 else 0.0


def _np_std(x: np.ndarray) -> float:
    return float(x.std()) if x.size > 0 else 0.0


def _compute_sharpe_like(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    std = float(x.std())
    if std < 1e-12:
        return 0.0
    return float(x.mean() / std)


def _compute_ann_sharpe_like(x: np.ndarray, periods_per_year: int) -> float:
    base = _compute_sharpe_like(x)
    if periods_per_year <= 0:
        return 0.0
    return float(base * np.sqrt(periods_per_year))


def _compute_eval_score(stats: Dict[str, float], cfg: TrainConfig) -> float:
    return float(
        cfg.val_reward_coef * stats.get("reward_sum", 0.0)
        - cfg.val_mdd_coef * stats.get("max_drawdown", 0.0)
        - cfg.val_turnover_coef * stats.get("avg_turnover", 0.0)
    )


def _aggregate_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        return {}
    keys = sorted(set().union(*(d.keys() for d in metric_dicts)))
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in metric_dicts if k in d]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def prefix_keys(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in d.items()}


@torch.no_grad()
def evaluate_policy(
    model: PPOSharedTransformerActorCritic,
    env: Any,
    device: torch.device,
    cfg: TrainConfig,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    obs, info = env.reset(seed=seed, options={"random_start": False})

    rewards: List[float] = []
    values: List[float] = [float(info["V"])]
    weights: List[float] = []
    turnovers: List[float] = []
    market_returns: List[float] = []
    sigma_series: List[float] = []
    critic_values: List[float] = []

    done = False
    while not done:
        x, p = obs_to_tensors(obs, device)
        logits, critic_value = model(x, p)

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
        critic_values.append(float(critic_value.item()))

        obs = next_obs

    reward_arr = np.asarray(rewards, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    turnover_arr = np.asarray(turnovers, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    mkt_arr = np.asarray(market_returns, dtype=np.float64)
    sigma_arr = np.asarray(sigma_series, dtype=np.float64)
    critic_arr = np.asarray(critic_values, dtype=np.float64)

    initial_value = float(value_arr[0]) if value_arr.size > 0 else 0.0
    final_value = float(value_arr[-1]) if value_arr.size > 0 else 0.0
    value_return = (
        float(final_value / max(initial_value, 1e-12) - 1.0)
        if value_arr.size > 0
        else 0.0
    )

    value_logret_arr = (
        np.diff(np.log(np.clip(value_arr, 1e-12, None)))
        if value_arr.size > 1
        else np.asarray([], dtype=np.float64)
    )

    num_trades = (
        float(np.sum(turnover_arr > cfg.trade_eps)) if turnover_arr.size > 0 else 0.0
    )
    fraction_in_market = (
        float(np.mean(weight_arr > cfg.trade_eps)) if weight_arr.size > 0 else 0.0
    )
    fraction_high_exposure = (
        float(np.mean(weight_arr >= cfg.high_exposure_threshold))
        if weight_arr.size > 0
        else 0.0
    )
    fraction_flat = (
        float(np.mean(weight_arr <= cfg.trade_eps)) if weight_arr.size > 0 else 0.0
    )

    buy_hold_log_return = float(mkt_arr.sum()) if mkt_arr.size > 0 else 0.0
    buy_hold_final_value = float(initial_value * np.exp(buy_hold_log_return))
    buy_hold_value_return = (
        float(buy_hold_final_value / max(initial_value, 1e-12) - 1.0)
        if initial_value > 0.0
        else 0.0
    )

    stats = {
        "episode_length": float(len(rewards)),
        "reward_sum": float(reward_arr.sum()) if reward_arr.size > 0 else 0.0,
        "reward_mean": _np_mean(reward_arr),
        "reward_std": _np_std(reward_arr),
        "reward_sharpe_like": _compute_sharpe_like(reward_arr),
        "final_value": final_value,
        "value_return": value_return,
        "value_logret_mean": _np_mean(value_logret_arr),
        "value_logret_std": _np_std(value_logret_arr),
        "value_sharpe_like": _compute_sharpe_like(value_logret_arr),
        "value_ann_sharpe_like": _compute_ann_sharpe_like(
            value_logret_arr, periods_per_year=cfg.rows_per_day * 365
        ),
        "max_drawdown": compute_max_drawdown(value_arr),
        "calmar_like": (
            float(value_return / max(compute_max_drawdown(value_arr), 1e-12))
            if value_arr.size > 0
            else 0.0
        ),
        "avg_weight": _np_mean(weight_arr),
        "weight_std": _np_std(weight_arr),
        "max_weight": float(weight_arr.max()) if weight_arr.size > 0 else 0.0,
        "min_weight": float(weight_arr.min()) if weight_arr.size > 0 else 0.0,
        "avg_turnover": _np_mean(turnover_arr),
        "turnover_std": _np_std(turnover_arr),
        "max_turnover": float(turnover_arr.max()) if turnover_arr.size > 0 else 0.0,
        "avg_abs_delta_w": _np_mean(turnover_arr),
        "num_trades": num_trades,
        "fraction_in_market": fraction_in_market,
        "fraction_high_exposure": fraction_high_exposure,
        "fraction_flat": fraction_flat,
        "mean_r_mkt_log": _np_mean(mkt_arr),
        "market_log_return_sum": float(mkt_arr.sum()) if mkt_arr.size > 0 else 0.0,
        "buy_hold_final_value": buy_hold_final_value,
        "buy_hold_value_return": buy_hold_value_return,
        "excess_value_return_vs_bh": float(value_return - buy_hold_value_return),
        "mean_sigma": _np_mean(sigma_arr),
        "sigma_std": _np_std(sigma_arr),
        "critic_value_mean": _np_mean(critic_arr),
        "critic_value_std": _np_std(critic_arr),
    }
    stats["score"] = _compute_eval_score(stats, cfg)
    return stats


def save_checkpoint(
    path: Path,
    model: PPOSharedTransformerActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    fold: Fold,
    feature_cols: Tuple[str, ...],
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_config": asdict(train_cfg),
        "model_config": asdict(model_cfg),
        "fold": asdict(fold),
        "feature_cols": list(feature_cols),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(payload, path)


def _train_one_fold(
    fold: Fold,
    df: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    cfg: TrainConfig,
    out_dir: Path,
    device: torch.device,
    tb_logger: Optional[TensorBoardLogger] = None,
) -> Dict[str, Any]:
    window_rows = cfg.history_days * cfg.rows_per_day

    env_cfg = TradingEnvConfig(
        window_len=window_rows,
        alpha=cfg.alpha,
        lambda_risk=cfg.lambda_risk,
        init_value=cfg.init_value,
        init_w=cfg.init_w,
        sigma_is_std=cfg.sigma_is_std,
    )

    scaler = fit_standardizer(
        df, fold.train_range[0], fold.train_range[1], feature_cols
    )

    train_env = make_env_from_slice(
        df=df,
        core_start=fold.train_range[0],
        core_end=fold.train_range[1],
        feature_cols=feature_cols,
        scaler=scaler,
        env_cfg=env_cfg,
        window_rows=window_rows,
        timestamp_col="timestamp",
    )
    val_env = make_env_from_slice(
        df=df,
        core_start=fold.val_range[0],
        core_end=fold.val_range[1],
        feature_cols=feature_cols,
        scaler=scaler,
        env_cfg=env_cfg,
        window_rows=window_rows,
        timestamp_col="timestamp",
    )
    test_env = make_env_from_slice(
        df=df,
        core_start=fold.test_range[0],
        core_end=fold.test_range[1],
        feature_cols=feature_cols,
        scaler=scaler,
        env_cfg=env_cfg,
        window_rows=window_rows,
        timestamp_col="timestamp",
    )

    feature_groups = build_feature_groups(feature_cols)

    model_cfg = ModelConfig(
        seq_feature_dim=len(feature_cols),
        portfolio_dim=3,
        action_dim=len(train_env.action_grid),
        max_seq_len=window_rows,
    )
    model = PPOSharedTransformerActorCritic(model_cfg).to(device)

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

    fold_dir = out_dir / f"fold_{fold.index:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = fold_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    records: List[Dict[str, float]] = []
    best_val_score = -float("inf")
    best_val_stats: Dict[str, float] = {}
    best_test_stats: Dict[str, float] = {}
    best_iter = -1

    pbar = tqdm(
        range(1, cfg.num_iterations + 1),
        desc=f"Fold {fold.index:02d}",
        leave=True,
        dynamic_ncols=True,
    )

    for iteration in pbar:
        train_stats = trainer.train_iteration()
        row: Dict[str, float] = {
            "iteration": float(iteration),
            "is_best": 0.0,
        }
        row.update(prefix_keys(train_stats, "train/"))

        do_eval = (
            (iteration % cfg.eval_every == 0)
            or (iteration == 1)
            or (iteration == cfg.num_iterations)
        )

        val_stats: Dict[str, float] = {}
        test_at_best_stats_for_tb: Optional[Dict[str, float]] = None

        if do_eval:
            val_stats = evaluate_policy(
                model=model,
                env=val_env,
                device=device,
                cfg=cfg,
                deterministic=cfg.deterministic_eval,
                seed=cfg.seed,
            )
            row.update(prefix_keys(val_stats, "val/"))

            current_val_score = val_stats["score"]
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                best_iter = iteration
                best_val_stats = dict(val_stats)
                row["is_best"] = 1.0

                save_checkpoint(
                    path=fold_dir / "best_model.pt",
                    model=model,
                    optimizer=trainer.optimizer,
                    iteration=iteration,
                    train_cfg=cfg,
                    model_cfg=model_cfg,
                    fold=fold,
                    feature_cols=feature_cols,
                    extra_state={
                        "best_iter": best_iter,
                        "best_val_score": best_val_score,
                        "best_val_stats": best_val_stats,
                    },
                )

                best_test_stats = evaluate_policy(
                    model=model,
                    env=test_env,
                    device=device,
                    cfg=cfg,
                    deterministic=cfg.deterministic_eval,
                    seed=cfg.seed,
                )
                row.update(prefix_keys(best_test_stats, "test_at_best/"))
                test_at_best_stats_for_tb = best_test_stats

        records.append(row)

        save_checkpoint(
            path=ckpt_dir / f"iter_{iteration:04d}.pt",
            model=model,
            optimizer=trainer.optimizer,
            iteration=iteration,
            train_cfg=cfg,
            model_cfg=model_cfg,
            fold=fold,
            feature_cols=feature_cols,
        )

        pd.DataFrame(records).to_csv(fold_dir / "metrics.csv", index=False)

        if tb_logger is not None:
            tb_logger.log_train_iteration(
                fold_index=fold.index,
                iteration=iteration,
                train_stats=train_stats,
            )

            tb_logger.log_best_marker(
                fold_index=fold.index,
                iteration=iteration,
                is_best=bool(row["is_best"] > 0.5),
            )

            if do_eval:
                tb_logger.log_eval_iteration(
                    split="val",
                    fold_index=fold.index,
                    iteration=iteration,
                    eval_stats=val_stats,
                )

            if test_at_best_stats_for_tb is not None:
                tb_logger.log_eval_iteration(
                    split="test_at_best",
                    fold_index=fold.index,
                    iteration=iteration,
                    eval_stats=test_at_best_stats_for_tb,
                )

        postfix = {
            "best_val_score": f"{best_val_score:.4f}" if best_iter >= 0 else "nan",
            "rr": f"{train_stats.get('rollout_mean_reward', float('nan')):.4f}",
        }
        if do_eval:
            postfix["val_score"] = f"{val_stats.get('score', float('nan')):.4f}"
            postfix["val_V"] = f"{val_stats.get('final_value', float('nan')):.4f}"
            postfix["val_mdd"] = f"{val_stats.get('max_drawdown', float('nan')):.4f}"
        pbar.set_postfix(postfix)

    pbar.close()

    best_path = fold_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        final_test_stats = evaluate_policy(
            model=model,
            env=test_env,
            device=device,
            cfg=cfg,
            deterministic=cfg.deterministic_eval,
            seed=cfg.seed,
        )
    else:
        final_test_stats = {}

    if tb_logger is not None:
        tb_logger.log_fold_summary(
            fold_index=fold.index,
            best_iter=best_iter,
            best_val_stats=best_val_stats,
            final_test_stats=final_test_stats,
        )

    print(
        f"Fold {fold.index:02d} done | "
        f"best_iter={best_iter} "
        f"best_val_score={best_val_score:.6f} "
        f"best_val_V={best_val_stats.get('final_value', float('nan')):.6f} "
        f"test_V={final_test_stats.get('final_value', float('nan')):.6f} "
        f"test_mdd={final_test_stats.get('max_drawdown', float('nan')):.4f}"
    )

    return {
        "fold_index": fold.index,
        "best_iter": best_iter,
        "best_val_score": best_val_score,
        "best_val_stats": best_val_stats,
        "test_at_best_stats": best_test_stats,
        "final_test_stats": final_test_stats,
        "records": records,
    }


def run_training(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        out_dir=out_dir,
        cfg=TensorBoardConfig(
            enabled=cfg.enable_tensorboard,
            flush_every=cfg.tensorboard_flush_every,
        ),
    )

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    data_cfg = DataPipelineConfig(
        csv_path=cfg.csv_path,
        history_days=cfg.history_days,
        sigma_days=cfg.sigma_days,
        multi_day_return_days=cfg.multi_day_return_days,
        rows_per_day=cfg.rows_per_day,
    )

    df, feature_cols = load_actionable_df(data_cfg)
    n = len(df)

    folds = generate_folds(
        n=n,
        rows_per_day=cfg.rows_per_day,
        min_train_days=cfg.min_train_days,
        val_days=cfg.val_days,
        test_days=cfg.test_days,
        step_days=cfg.step_days,
    )

    if not folds:
        raise ValueError(
            f"Not enough data for any walk-forward fold. "
            f"Actionable rows={n}; need at least "
            f"{(cfg.min_train_days + cfg.val_days + cfg.test_days) * cfg.rows_per_day}."
        )

    device = torch.device(cfg.device)
    window_rows = cfg.history_days * cfg.rows_per_day

    print("=" * 80)
    print("Walk-forward training setup")
    print(f"device              : {cfg.device}")
    print(f"csv_path            : {cfg.csv_path}")
    print(f"output_dir          : {cfg.output_dir}")
    print(f"actionable rows     : {n}")
    print(f"feature_dim         : {len(feature_cols)}")
    print(f"window_rows         : {window_rows}")
    print(f"num_folds           : {len(folds)}")
    print(f"num_iterations      : {cfg.num_iterations} per fold")
    print(f"tensorboard         : {cfg.enable_tensorboard}")
    print(
        "val_score           : "
        f"{cfg.val_reward_coef:.3f} * reward_sum "
        f"- {cfg.val_mdd_coef:.3f} * max_drawdown "
        f"- {cfg.val_turnover_coef:.3f} * avg_turnover"
    )
    for fold in folds:
        print(
            f"  fold {fold.index:02d}: "
            f"train=[{fold.train_range[0]},{fold.train_range[1]}) "
            f"val=[{fold.val_range[0]},{fold.val_range[1]}) "
            f"test=[{fold.test_range[0]},{fold.test_range[1]})"
        )
    print("=" * 80)

    fold_summaries = []
    try:
        for fold in folds:
            print(f"\n{'─' * 40} Fold {fold.index} {'─' * 40}")
            result = _train_one_fold(
                fold=fold,
                df=df,
                feature_cols=feature_cols,
                cfg=cfg,
                out_dir=out_dir,
                device=device,
                tb_logger=tb_logger,
            )
            fold_summaries.append(result)
    finally:
        pass

    all_best_val = [r["best_val_stats"] for r in fold_summaries if r["best_val_stats"]]
    all_test = [r["final_test_stats"] for r in fold_summaries if r["final_test_stats"]]

    agg_best_val = _aggregate_metric_dicts(all_best_val)
    agg_test = _aggregate_metric_dicts(all_test)

    final_summary = {
        "num_folds": len(folds),
        "folds": [
            {
                "fold_index": r["fold_index"],
                "best_iter": r["best_iter"],
                "best_val_score": r["best_val_score"],
                **prefix_keys(r["best_val_stats"], "best_val/"),
                **prefix_keys(r["final_test_stats"], "test/"),
            }
            for r in fold_summaries
        ],
        **prefix_keys(agg_best_val, "avg_best_val/"),
        **prefix_keys(agg_test, "avg_test/"),
    }

    with open(out_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    if tb_logger is not None:
        tb_logger.log_run_summary(
            avg_best_val_stats=agg_best_val,
            avg_test_stats=agg_test,
            step=len(folds),
        )
        tb_logger.close()

    print("=" * 80)
    print("Walk-forward training finished")
    print(f"Folds completed       : {len(folds)}")
    if agg_best_val:
        print(f"Avg best val score    : {agg_best_val.get('score', float('nan')):.6f}")
        print(
            f"Avg best val V        : {agg_best_val.get('final_value', float('nan')):.6f}"
        )
        print(
            f"Avg best val MDD      : {agg_best_val.get('max_drawdown', float('nan')):.4f}"
        )
    if agg_test:
        print(
            f"Avg test final_value  : {agg_test.get('final_value', float('nan')):.6f}"
        )
        print(
            f"Avg test max_drawdown : {agg_test.get('max_drawdown', float('nan')):.4f}"
        )
        print(f"Avg test score        : {agg_test.get('score', float('nan')):.6f}")
    print(f"Artifacts saved to    : {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    cfg = TrainConfig()
    run_training(cfg)
