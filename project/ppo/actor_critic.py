"""
ppo/actor_critic.py
====================
The shared actor-critic network and rollout buffer used by train.py /
test.py / evaluate.py. Implements PPO exactly as specified in Algorithm 1
and Eq. 9 of Schulman et al., 2017 -- adapted from N parallel Atari-game
copies to N=n_agents junctions inside one TrafficGridEnv (see config.py's
docstring for why this is a faithful mapping of the paper's actor
structure, not a simplification of it).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


NEG_LOGIT = -1e9


def mask_switch_logits(logits: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    """Ineligible junction (yellow, or elapsed < min_green): the switch logit (index 1) is set to a
    large negative value, so P(switch)=0 and the sampled/argmax action is always hold."""
    block = torch.zeros_like(logits, dtype=torch.bool)
    block[..., 1] = ~eligible
    return logits.masked_fill(block, NEG_LOGIT)


class ActorCritic(nn.Module):
    """Shared trunk, two heads: policy logits (2 actions) and a scalar
    value estimate. Matches the paper's Eq. 9 combined-objective setting,
    where policy and value share parameters."""

    def __init__(self, input_dim: int, hidden_sizes: tuple, n_actions: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, n_actions)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, x: torch.Tensor, eligible: torch.Tensor):
        """Samples an action from the current policy (stochastic, matching
        the paper's on-policy rollout collection -- NOT epsilon-greedy;
        PPO's exploration comes from the policy's own entropy, encouraged
        by entropy_coeff during training)."""
        logits, value = self.forward(x)
        logits = mask_switch_logits(logits, eligible)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    @torch.no_grad()
    def act_greedy(self, x: torch.Tensor, eligible: torch.Tensor):
        """Deterministic action (argmax over the masked logits) for evaluation. DQN is evaluated
        fully greedy too (epsilon=0), so both algorithms use the same mode."""
        logits, _ = self.forward(x)
        return mask_switch_logits(logits, eligible).argmax(dim=-1)


class RolloutBuffer:
    """Stores one horizon-length rollout across all n_agents junctions
    (n_agents acting as the paper's N parallel actors) and computes
    Generalized Advantage Estimation (Schulman et al. 2015a, referenced by
    this paper's Eq. 11-12) per-agent, since each junction's reward stream
    is its own independent trajectory under the shared policy."""

    def __init__(self, horizon: int, n_agents: int, obs_dim: int):
        self.horizon = horizon
        self.n_agents = n_agents
        self.obs = np.zeros((horizon, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, n_agents), dtype=np.int64)
        self.log_probs = np.zeros((horizon, n_agents), dtype=np.float32)
        self.values = np.zeros((horizon, n_agents), dtype=np.float32)
        self.rewards = np.zeros((horizon, n_agents), dtype=np.float32)
        self.eligible = np.zeros((horizon, n_agents), dtype=bool)  # could this junction switch at this step?
        self.dones = np.zeros((horizon, n_agents), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, actions, log_probs, values, rewards, done, eligible):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.eligible[t] = eligible
        self.dones[t] = done  # same scalar `done` broadcast to all agents:
        # the environment resets globally, so every junction's episode
        # boundary is simultaneous -- there is no per-agent-only done.
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.horizon

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """last_values: (n_agents,) bootstrap value estimate for the state
        immediately after the last stored step (V=0 wherever that step was
        terminal, matching the paper's Eq. 10 finite-horizon estimator)."""
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.n_agents, dtype=np.float32)
        for t in reversed(range(self.horizon)):
            next_values = last_values if t == self.horizon - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def get_flat(self, advantages, returns):
        """Flattens (horizon, n_agents, ...) -> (horizon*n_agents, ...) for
        minibatch sampling, matching the paper's "construct the surrogate
        loss on these NT timesteps of data" step in Algorithm 1."""
        b_obs = self.obs.reshape(-1, self.obs.shape[-1])
        b_actions = self.actions.reshape(-1)
        b_log_probs = self.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_eligible = self.eligible.reshape(-1)
        return b_obs, b_actions, b_log_probs, b_advantages, b_returns, b_eligible

    def reset(self):
        self.ptr = 0
