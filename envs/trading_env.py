from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from ..rewards import *


@dataclass(frozen=True)
class TradingEnvConfig:
    window_len: int = 60
    alpha: float = 0.001
    lambda_risk: float = 0.1
    init_value: float = 1.0
    init_w: float = 0.0
    action_grid: Tuple[float, ...] = tuple(np.round(np.linspace(0.0, 1.0, 11), 1))
    eps: float = 1e-12
    sigma_is_std: bool = True  # if False, variance


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        X: Union[np.ndarray, Sequence[float]],
        r_mkt_log: Union[np.ndarray, Sequence[float]],
        sigma: Union[np.ndarray, Sequence[float]],
        dates: Optional[Sequence[Any]] = None,
        config: TradingEnvConfig = TradingEnvConfig(),
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = config

        self.X = np.asarray(X, dtype=np.float32)
        self.r_mkt_log = np.asarray(r_mkt_log, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)

        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D (T,F). Got shape {self.X.shape}")
        if self.r_mkt_log.ndim != 1:
            raise ValueError(
                f"r_mkt_log must be 1D (T,). Got shape {self.r_mkt_log.shape}"
            )
        if self.sigma.ndim != 1:
            raise ValueError(f"sigma must be 1D (T,). Got shape {self.sigma.shape}")

        T_x = self.X.shape[0]
        T_r = self.r_mkt_log.shape[0]
        T_s = self.sigma.shape[0]
        self.T = int(min(T_x, T_r, T_s))
        if self.T < self.cfg.window_len:
            raise ValueError(
                f"Not enough timesteps: T={self.T} < window_len={self.cfg.window_len}"
            )

        self.dates = list(dates) if dates is not None else None
        if self.dates is not None and len(self.dates) < self.T:
            raise ValueError(f"dates length {len(self.dates)} < T {self.T}")

        self.t_min = self.cfg.window_len - 1
        self.t_end = self.T - 1

        self._rng = np.random.default_rng(seed)

        self.action_grid = np.asarray(self.cfg.action_grid, dtype=np.float32)
        if self.action_grid.ndim != 1 or len(self.action_grid) < 2:
            raise ValueError(
                "action_grid must be a 1D sequence with at least 2 values."
            )
        if np.any(self.action_grid < 0.0) or np.any(self.action_grid > 1.0):
            raise ValueError("action_grid weights must be within [0,1].")
        self.action_space = spaces.Discrete(len(self.action_grid))

        F = self.X.shape[1]
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.cfg.window_len, F),
                    dtype=np.float32,
                ),
                "p": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )

        self.t: int = self.t_min
        self.V: float = float(self.cfg.init_value)
        self.w: float = float(self.cfg.init_w)

        self.last_portfolio_update: Optional[PortfolioUpdate] = None
        self.last_reward_breakdown: Optional[RewardBreakdown] = None

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _get_date(self, t: int) -> Any:
        if self.dates is None:
            return t
        return self.dates[t]

    def _obs(self, t: int) -> Dict[str, np.ndarray]:
        if t < self.t_min or t > self.t_end:
            raise IndexError(
                f"t={t} out of valid obs range [{self.t_min}, {self.t_end}]"
            )

        x = self.X[t - self.cfg.window_len + 1 : t + 1]  # (window_len, F)
        if x.shape[0] != self.cfg.window_len:
            raise RuntimeError(f"Bad window slice at t={t}: got {x.shape}")

        V_safe = max(self.cfg.eps, float(self.V))
        sigma_t = float(self.sigma[t])
        p = np.array([float(self.w), float(np.log(V_safe)), sigma_t], dtype=np.float32)

        return {"x": x.astype(np.float32, copy=False), "p": p}

    def _terminal_obs(self) -> Dict[str, np.ndarray]:
        F = self.X.shape[1]
        return {
            "x": np.zeros((self.cfg.window_len, F), dtype=np.float32),
            "p": np.zeros((3,), dtype=np.float32),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)

        options = options or {}
        random_start = bool(options.get("random_start", True))
        start_idx = options.get("start_idx", None)

        if start_idx is not None:
            t0 = int(start_idx)
            if t0 < self.t_min or t0 > self.t_end:
                raise ValueError(
                    f"start_idx={t0} out of range [{self.t_min}, {self.t_end}]"
                )
        else:
            if random_start:
                t0 = int(self._rng.integers(self.t_min, self.t_end + 1))
            else:
                t0 = self.t_min

        self.t = t0
        self.V = float(self.cfg.init_value)
        self.w = float(self.cfg.init_w)

        self.last_portfolio_update = None
        self.last_reward_breakdown = None

        obs = self._obs(self.t)
        info = {
            "t": self.t,
            "date": self._get_date(self.t),
            "V": self.V,
            "w": self.w,
            "sigma": float(self.sigma[self.t]),
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action} for action_space {self.action_space}"
            )

        if self.t < self.t_min or self.t > self.t_end:
            raise RuntimeError(
                "step() called in terminal/out-of-range state; call reset()"
            )

        w_next = float(self.action_grid[int(action)])
        r_log = float(self.r_mkt_log[self.t])
        sigma_t = float(self.sigma[self.t])

        pu = update_portfolio(
            V=float(self.V),
            w=float(self.w),
            w_next=w_next,
            r_mkt=r_log,
            alpha=float(self.cfg.alpha),
            r_is_log=True,
            clamp_cost=True,
            clamp_r_simple=True,
            weight_bounds=(0.0, 1.0),
            eps=self.cfg.eps,
        )
        rb = compute_reward(
            V=float(self.V),
            V_next=float(pu.V_next),
            w_next=w_next,
            sigma=sigma_t,
            lambda_risk=float(self.cfg.lambda_risk),
            eps=self.cfg.eps,
            sigma_is_std=self.cfg.sigma_is_std,
            weight_bounds=(0.0, 1.0),
            clamp_values=True,
        )

        V_prev = float(self.V)
        w_prev = float(self.w)
        self.V = float(pu.V_next)
        self.w = w_next
        self.last_portfolio_update = pu
        self.last_reward_breakdown = rb

        next_t = self.t + 1
        terminated = next_t > self.t_end
        truncated = False
        self.t = next_t

        obs = self._terminal_obs() if terminated else self._obs(self.t)

        info: Dict[str, Any] = {
            "t_prev": next_t - 1,
            "t": self.t,
            "date_prev": self._get_date(next_t - 1),
            "date": None if terminated else self._get_date(self.t),
            "action": int(action),
            "w_prev": w_prev,
            "w_next": w_next,
            "delta_w": float(pu.delta_w),
            "V_prev": V_prev,
            "V_next": float(self.V),
            "r_mkt_log": r_log,
            "r_simple": float(pu.r_simple),
            "cost_multiplier": float(pu.cost_multiplier),
            "market_multiplier": float(pu.market_multiplier),
            "sigma": sigma_t,
            "log_growth": float(rb.log_growth),
            "risk_penalty": float(rb.risk_penalty),
            "reward": float(rb.total),
        }

        return obs, float(rb.total), bool(terminated), bool(truncated), info

    def render(self) -> None:
        t_show = min(self.t, self.t_end)
        print(f"t={t_show} date={self._get_date(t_show)} V={self.V:.6f} w={self.w:.2f}")
