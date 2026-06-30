"""
Online Reinforcement Learning Agent (PPO) for adaptive position sizing.

Architecture:
  - The RL agent does NOT make buy/sell decisions.
  - It outputs a `size_scalar` in [0.0, 1.0] that multiplies the
    deterministic decision engine's position size.
  - Trained online from live tick-level PnL rewards using a rolling
    experience replay buffer (bounded memory, no historical data storage).

CPU-optimized: 3-layer MLP with 64 hidden units (~0.1ms inference).
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# ── Constants ──
STATE_DIM = 7       # [edge, risk_mod, spread_bps, book_depth_norm, volatility, prev_pnl, pattern_score]
ACTION_DIM = 1      # continuous scalar in [0, 1]
HIDDEN_DIM = 64
BUFFER_SIZE = 2048  # half-context rolling window
BATCH_SIZE = 256
TRAIN_EPOCHS = 4
GAMMA = 0.99
CLIP_EPS = 0.2
LR = 3e-4
ENTROPY_COEF = 0.01


# ── Experience Storage ──
@dataclass
class Experience:
    state: list[float]
    action: float
    log_prob: float
    reward: float
    done: bool


class RollingReplayBuffer:
    """Fixed-size deque that automatically discards oldest experiences."""

    def __init__(self, maxlen: int = BUFFER_SIZE):
        self._buf: deque[Experience] = deque(maxlen=maxlen)

    def push(self, exp: Experience) -> None:
        self._buf.append(exp)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        return len(self._buf) >= BATCH_SIZE

    def sample_all(self) -> list[Experience]:
        """Return entire buffer for on-policy PPO update, then keep rolling."""
        return list(self._buf)

    def clear_old(self) -> None:
        """Keep only the most recent half (the rolling window idea)."""
        keep = len(self._buf) // 2
        recent = list(self._buf)[-keep:]
        self._buf.clear()
        self._buf.extend(recent)


# ── Policy Network ──
class PolicyNetwork(nn.Module):
    """Tiny MLP: state → (mean, log_std) for Gaussian policy."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, ACTION_DIM)
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = torch.sigmoid(self.mean_head(h))  # clamp to [0, 1]
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, state: torch.Tensor) -> tuple[float, float]:
        """Sample an action and return (action, log_prob)."""
        with torch.no_grad():
            mean, std = self.forward(state)
            dist = torch.distributions.Normal(mean, std)
            raw = dist.sample()
            action = torch.clamp(raw, 0.0, 1.0)
            log_prob = dist.log_prob(raw).sum()
        return action.item(), log_prob.item()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log_probs and entropy for PPO loss."""
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


# ── Value Network (Critic) ──
class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── PPO Agent ──
class PPOAgent:
    """
    Proximal Policy Optimization agent for online position sizing.

    Usage:
        agent = PPOAgent()
        scalar = agent.act(state_vector)         # get size_scalar
        agent.observe(reward, done=False)         # feed PnL reward
        # ... training happens automatically when buffer is full
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        self.device = torch.device("cpu")
        self.policy = PolicyNetwork().to(self.device)
        self.value = ValueNetwork().to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.value_opt = optim.Adam(self.value.parameters(), lr=LR)
        self.buffer = RollingReplayBuffer(maxlen=BUFFER_SIZE)
        self.checkpoint_dir = checkpoint_dir or Path("data/rl")

        self._last_state: torch.Tensor | None = None
        self._last_action: float = 0.5
        self._last_log_prob: float = 0.0
        self._step_count = 0

        # Try to load existing checkpoint
        self._load_checkpoint()

    def act(self, state: list[float]) -> float:
        """Given a state vector, return a size_scalar in [0, 1]."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, log_prob = self.policy.get_action(s)

        self._last_state = state
        self._last_action = action
        self._last_log_prob = log_prob
        return action

    def observe(self, reward: float, done: bool = False) -> None:
        """Feed the PnL reward from the last action."""
        if self._last_state is None:
            return

        self.buffer.push(Experience(
            state=self._last_state,
            action=self._last_action,
            log_prob=self._last_log_prob,
            reward=reward,
            done=done,
        ))
        self._step_count += 1

        # Train every BATCH_SIZE steps if buffer is ready
        if self.buffer.ready and self._step_count % BATCH_SIZE == 0:
            self._train()
            self.buffer.clear_old()  # keep rolling window
            self._save_checkpoint()

    def _train(self) -> None:
        """Run PPO update on the current buffer."""
        exps = self.buffer.sample_all()

        states = torch.tensor([e.state for e in exps], dtype=torch.float32)
        actions = torch.tensor([[e.action] for e in exps], dtype=torch.float32)
        old_log_probs = torch.tensor([e.log_prob for e in exps], dtype=torch.float32)
        rewards_raw = [e.reward for e in exps]
        dones = [e.done for e in exps]

        # Compute discounted returns
        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards_raw), reversed(dones)):
            if d:
                G = 0.0
            G = r + GAMMA * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        for _ in range(TRAIN_EPOCHS):
            # Value loss
            values = self.value(states)
            advantages = returns_t - values.detach()
            value_loss = nn.functional.mse_loss(values, returns_t)

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            # Policy loss (PPO clip)
            new_log_probs, entropy = self.policy.evaluate(states, actions)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            policy_loss -= ENTROPY_COEF * entropy.mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

        logger.info(
            f"PPO update | steps={self._step_count} | "
            f"policy_loss={policy_loss.item():.4f} | value_loss={value_loss.item():.4f}"
        )

    def _save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "step": self._step_count,
        }, self.checkpoint_dir / "ppo_checkpoint.pt")

    def _load_checkpoint(self) -> None:
        ckpt = self.checkpoint_dir / "ppo_checkpoint.pt"
        if ckpt.exists():
            try:
                data = torch.load(ckpt, map_location=self.device, weights_only=True)
                self.policy.load_state_dict(data["policy"])
                self.value.load_state_dict(data["value"])
                self._step_count = data.get("step", 0)
                logger.info(f"Loaded RL checkpoint (step {self._step_count})")
            except Exception as e:
                logger.warning(f"Failed to load RL checkpoint: {e}")


# ── Convenience: build state vector from pipeline data ──
def build_state_vector(
    signal: dict[str, Any],
    news: dict[str, Any] | None = None,
    market_spread_bps: float = 1.0,
    book_depth_usd: float = 100000.0,
    volatility: float = 0.02,
    prev_pnl: float = 0.0,
    pattern_score: float = 0.5,
) -> list[float]:
    """
    Assembles the 7-dim state vector from pipeline data.
    All values are normalized to roughly [-1, 1] or [0, 1].
    """
    edge = float(signal.get("net_edge", 0))
    risk_mod = float(news.get("risk_modifier", 1.0)) if news else 1.0
    spread_norm = min(market_spread_bps / 10.0, 1.0)       # normalize to [0, 1]
    depth_norm = min(book_depth_usd / 200000.0, 1.0)       # normalize to [0, 1]
    vol_norm = min(volatility / 0.1, 1.0)                   # normalize to [0, 1]
    pnl_norm = max(min(prev_pnl / 100.0, 1.0), -1.0)       # normalize to [-1, 1]

    return [edge * 100, risk_mod, spread_norm, depth_norm, vol_norm, pnl_norm, pattern_score]
