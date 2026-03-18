from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FeatureGroups:
    price_like: Tuple[int, ...] = ()
    volume_like: Tuple[int, ...] = ()
    onchain_like: Tuple[int, ...] = ()
    never_mask: Tuple[int, ...] = ()


@dataclass(frozen=True)
class AugmentConfig:
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


class TradingAugmenter(nn.Module):
    def __init__(self, cfg: AugmentConfig, groups: FeatureGroups) -> None:
        super().__init__()
        self.cfg = cfg
        self.groups = groups

    def forward(
        self,
        x_seq: torch.Tensor,
        x_port: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(x_seq, x_port)

        if not self.training:
            return x_seq, x_port

        x_seq_aug = x_seq.clone()
        x_port_aug = x_port.clone()

        device = x_seq.device

        if torch.rand(1, device=device).item() < self.cfg.p_jitter:
            x_seq_aug = self._jitter(x_seq_aug)

        if torch.rand(1, device=device).item() < self.cfg.p_feature_dropout:
            x_seq_aug = self._feature_dropout(x_seq_aug)

        if torch.rand(1, device=device).item() < self.cfg.p_time_mask:
            x_seq_aug = self._time_mask(x_seq_aug)

        if torch.rand(1, device=device).item() < self.cfg.p_scale:
            x_seq_aug = self._scale_selected_features(x_seq_aug)

        if self.cfg.augment_portfolio:
            x_port_aug = self._portfolio_jitter(x_port_aug)

        return x_seq_aug, x_port_aug

    def _validate_inputs(self, x_seq: torch.Tensor, x_port: torch.Tensor) -> None:
        if x_seq.ndim != 3:
            raise ValueError(
                f"x_seq must have shape [B, T, F], got {tuple(x_seq.shape)}"
            )
        if x_port.ndim != 2:
            raise ValueError(
                f"x_port must have shape [B, P], got {tuple(x_port.shape)}"
            )
        if x_seq.shape[0] != x_port.shape[0]:
            raise ValueError(
                f"Batch size mismatch: x_seq has B={x_seq.shape[0]}, "
                f"x_port has B={x_port.shape[0]}"
            )

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.cfg.jitter_std
        return x + noise

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        candidate = list(self.groups.volume_like) + list(self.groups.onchain_like)
        never_mask = set(self.groups.never_mask)
        candidate = [idx for idx in candidate if idx not in never_mask]

        if len(candidate) == 0:
            return x

        B, _, _ = x.shape
        device = x.device

        idx = torch.tensor(candidate, dtype=torch.long, device=device)
        selected = x[:, :, idx]  # [B, T, C]

        mask = (
            torch.rand(B, 1, len(candidate), device=device)
            < self.cfg.feature_dropout_prob
        )

        selected = selected.masked_fill(mask, 0.0)
        x[:, :, idx] = selected
        return x

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        usable_T = T - self.cfg.leave_last_n
        if usable_T <= 0:
            return x

        max_mask_len = max(1, int(T * self.cfg.max_time_mask_frac))
        max_mask_len = min(max_mask_len, usable_T)

        for b in range(B):
            mask_len = torch.randint(
                low=1,
                high=max_mask_len + 1,
                size=(1,),
                device=device,
            ).item()

            start_max = usable_T - mask_len
            if start_max < 0:
                continue

            start = torch.randint(
                low=0,
                high=start_max + 1,
                size=(1,),
                device=device,
            ).item()

            x[b, start : start + mask_len, :] = 0.0

        return x

    def _scale_selected_features(self, x: torch.Tensor) -> torch.Tensor:
        selected = list(self.groups.volume_like) + list(self.groups.onchain_like)
        if len(selected) == 0:
            return x

        device = x.device
        B, _, _ = x.shape

        idx = torch.tensor(selected, dtype=torch.long, device=device)

        scale = 1.0 + (
            (torch.rand(B, 1, len(selected), device=device) * 2.0 - 1.0)
            * self.cfg.scale_range
        )

        x[:, :, idx] = x[:, :, idx] * scale
        return x

    def _portfolio_jitter(self, x_port: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_port) * self.cfg.portfolio_jitter_std
        return x_port + noise
