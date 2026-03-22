from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import math

from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class TensorBoardConfig:
    enabled: bool = True
    flush_every: int = 1


TRAIN_KEYS = (
    "rollout_mean_reward",
    "value_loss",
    "entropy",
    "approx_kl",
    "explained_variance",
)

VAL_KEYS = (
    "score",
    "final_value",
    "max_drawdown",
    "avg_turnover",
)


def create_tensorboard_writer(
    out_dir: Path | str,
    cfg: TensorBoardConfig = TensorBoardConfig(),
) -> Optional["SummaryWriter"]:
    if not cfg.enabled:
        return None

    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard is not available. Install it with: pip install tensorboard"
        )

    log_dir = Path(out_dir) / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _to_finite_scalar_dict(metrics: Mapping[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out[k] = fv
    return out


class TensorBoardLogger:
    def __init__(
        self,
        out_dir: Path | str,
        cfg: TensorBoardConfig = TensorBoardConfig(),
    ) -> None:
        self.cfg = cfg
        self.writer = create_tensorboard_writer(out_dir=out_dir, cfg=cfg)
        self._write_count = 0

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _maybe_flush(self) -> None:
        if self.writer is None:
            return
        self._write_count += 1
        if self.cfg.flush_every > 0 and self._write_count % self.cfg.flush_every == 0:
            self.writer.flush()

    def _log_fold_metric(
        self,
        chart_name: str,
        fold_index: int,
        step: int,
        value: float,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalars(
            chart_name,
            {f"fold_{fold_index:02d}": float(value)},
            step,
        )
        self._maybe_flush()

    def log_train_iteration(
        self,
        fold_index: int,
        iteration: int,
        train_stats: Mapping[str, object],
    ) -> None:
        if self.writer is None:
            return

        stats = _to_finite_scalar_dict(train_stats)
        for key in TRAIN_KEYS:
            if key in stats:
                self._log_fold_metric(
                    chart_name=f"train/{key}",
                    fold_index=fold_index,
                    step=iteration,
                    value=stats[key],
                )

    def log_eval_iteration(
        self,
        split: str,
        fold_index: int,
        iteration: int,
        eval_stats: Mapping[str, object],
    ) -> None:
        if self.writer is None:
            return

        if split != "val":
            return

        stats = _to_finite_scalar_dict(eval_stats)
        for key in VAL_KEYS:
            if key in stats:
                self._log_fold_metric(
                    chart_name=f"val/{key}",
                    fold_index=fold_index,
                    step=iteration,
                    value=stats[key],
                )

    def log_best_marker(
        self,
        fold_index: int,
        iteration: int,
        is_best: bool,
    ) -> None:
        return

    def log_fold_summary(
        self,
        fold_index: int,
        best_iter: int,
        best_val_stats: Mapping[str, object],
        final_test_stats: Mapping[str, object],
    ) -> None:
        return

    def log_run_summary(
        self,
        avg_best_val_stats: Mapping[str, object],
        avg_test_stats: Mapping[str, object],
        step: int = 0,
    ) -> None:
        return
