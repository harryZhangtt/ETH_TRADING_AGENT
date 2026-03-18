from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    """
    Placeholder config for a PPO actor-critic model with a shared Transformer encoder.

    Expected input :
    - x_seq: 60-day window of features, shape [B, 60, seq_feature_dim]
    - x_port: portfolio state, shape [B, portfolio_dim]
    """

    seq_feature_dim: int
    portfolio_dim: int
    action_dim: int  # discrete actions for categorical policy

    max_seq_len: int = 60

    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    latent_dim: int = 256


class TimeSeriesPortfolioTransformerEncoder(nn.Module):
    """
    Shared encoder for PPO.

    Inputs:
    - x_seq: [B, T(=max_seq_len), F]
    - x_port: [B, P]
    Output:
    - h: [B, latent_dim]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.seq_proj = nn.Linear(cfg.seq_feature_dim, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_seq_len + 1, cfg.d_model))
        self.seq_drop = nn.Dropout(cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.seq_norm = nn.LayerNorm(cfg.d_model)

        self.port_mlp = nn.Sequential(
            nn.LayerNorm(cfg.portfolio_dim),
            nn.Linear(cfg.portfolio_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_model, cfg.latent_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_seq: torch.Tensor, x_port: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq must be [B,T,F], got {tuple(x_seq.shape)}")
        if x_port.dim() != 2:
            raise ValueError(f"x_port must be [B,P], got {tuple(x_port.shape)}")

        bsz, t, _ = x_seq.shape
        if t != self.cfg.max_seq_len:
            raise ValueError(f"Expected T={self.cfg.max_seq_len}, got T={t}")

        x = self.seq_proj(x_seq)  # [B,T,d_model]
        cls = self.cls_token.expand(bsz, -1, -1)  # [B,1,d_model]
        x = torch.cat([cls, x], dim=1)  # [B,T+1,d_model]
        x = x + self.pos_embed[:, : t + 1, :]
        x = self.seq_drop(x)

        x = self.transformer(x)  # [B,T+1,d_model]
        x = self.seq_norm(x)
        seq_emb = x[:, 0, :]  # CLS pooling, [B,d_model]

        port_emb = self.port_mlp(x_port)  # [B,d_model]
        h = self.fuse(torch.cat([seq_emb, port_emb], dim=-1))  # [B,latent_dim]
        return h


class PPOSharedTransformerActorCritic(nn.Module):
    """
    PPO actor-critic with a shared Transformer encoder.

    - Actor head outputs logits for a categorical policy over discrete actions.
    - Critic head outputs scalar V(s).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TimeSeriesPortfolioTransformerEncoder(cfg)

        self.actor = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.action_dim),
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, 1),
        )

    def forward(
        self, x_seq: torch.Tensor, x_port: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x_seq, x_port)
        logits = self.actor(h)  # [B, action_dim]
        value = self.critic(h).squeeze(-1)  # [B]
        return logits, value

    @torch.no_grad()
    def act(
        self, x_seq: torch.Tensor, x_port: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the current policy.
        Returns:
        - action: [B] (dtype long)
        - logprob: [B]
        - value: [B]
        """
        logits, value = self.forward(x_seq, x_port)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(
        self, x_seq: torch.Tensor, x_port: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO update.
        Returns:
        - logprob: [B]
        - entropy: [B]
        - value: [B]
        """
        logits, value = self.forward(x_seq, x_port)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value


__all__ = [
    "ModelConfig",
    "TimeSeriesPortfolioTransformerEncoder",
    "PPOSharedTransformerActorCritic",
]

if __name__ == "__main__":

    import torch
    from model import PPOSharedTransformerActorCritic

    B, T, F = 32, 60, 20  # batch=32，window size=60 days，20 features per day
    P = 3  # portfolio state dim
    A = 11  # discrete action here（0.0~1.0 step 0.1）

    net = PPOSharedTransformerActorCritic(
        seq_feature_dim=F, portfolio_dim=P, action_dim=A
    )

    x_seq = torch.randn(B, T, F)
    x_port = torch.randn(B, P)

    action, logprob, value = net.act(x_seq, x_port)
