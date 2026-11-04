"""
ac/network.py
=============
Separate actor and critic MLPs (two-timescale algorithms need independent parameters and optimizers), shared by ac/ and ca/.
Both map ONE junction's 11-dim local observation to their output, exactly like dqn.QNetwork / ppo.ActorCritic, and are
applied to every junction with shared weights. tanh hidden units (same as PPO; DQN uses ReLU -- disclosed in LIMITATIONS).

The actor is MASKED: when a junction is not eligible to switch (yellow, or elapsed < min_green) the switch logit is set
to -1e9 with the SAME function PPO uses (ppo.actor_critic.mask_switch_logits), so P(switch) = 0 there, the sampled /
argmax action is 'hold', its log-probability is 0 and its entropy is 0. `eligible` is a required argument so an
unmasked policy cannot be built by accident.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ppo.actor_critic import mask_switch_logits   # single definition of the masking rule, shared with PPO


def _mlp(input_dim: int, hidden_sizes: tuple, output_dim: int) -> nn.Sequential:
    layers, prev = [], input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple, n_actions: int):
        super().__init__()
        self.net = _mlp(input_dim, hidden_sizes, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def masked_logits(self, x: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        return mask_switch_logits(self.forward(x), eligible)

    def act(self, x: torch.Tensor, eligible: torch.Tensor):
        """Sample from the masked policy. Returns (action, log_prob, entropy); gradients flow through log_prob / entropy."""
        dist = Categorical(logits=self.masked_logits(x, eligible))
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def act_greedy(self, x: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        """Deterministic evaluation action: argmax of the masked logits (same mode as PPO and DQN at evaluation)."""
        return self.masked_logits(x, eligible).argmax(dim=-1)


class CriticNetwork(nn.Module):
    """State-value approximator V_w(s) of one junction's observation."""

    def __init__(self, input_dim: int, hidden_sizes: tuple):
        super().__init__()
        self.net = _mlp(input_dim, hidden_sizes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
