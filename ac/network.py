"""
ac/network.py
=============
Separate actor and critic networks for the two-timescale AC/CA algorithms
(see config.py's docstring for why these must be separate networks/
optimizers, unlike ppo/actor_critic.py's shared-trunk ActorCritic).

Both networks map one junction's local 11-dim observation to their
respective output (matching dqn/q_network.py's QNetwork and
ppo/actor_critic.py's ActorCritic in input/output shape, for a fair
apples-to-apples comparison across all four algorithms trained on this
environment).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(input_dim: int, hidden_sizes: tuple, output_dim: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """Boltzmann/Gibbs policy (paper's Eq. 6: pi_theta(i,a) = softmax over
    theta(i,a)) approximated by a neural network instead of a per-state-
    action tabular parameter theta(i,a). The network outputs the logits
    that Eq. 6's softmax is applied to; Categorical(logits=...) below IS
    that softmax."""

    def __init__(self, input_dim: int, hidden_sizes: tuple, n_actions: int):
        super().__init__()
        self.net = _mlp(input_dim, hidden_sizes, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, x: torch.Tensor):
        """Samples an action per row of x (one row per junction) from the
        current stochastic policy -- the paper's phi_n(i) ~ pi_theta_n(i, .)
        (Section III, just above Eq. 7)."""
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    @torch.no_grad()
    def act_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic argmax action for test.py/evaluate.py, mirroring
        the paper's theta0 -> infinity limit (Theorem 3) where the softmax
        policy collapses onto the greedy/optimal action."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


class CriticNetwork(nn.Module):
    """Value-function approximator V_w(i) (paper's Vn(i), Eq. 7), replacing
    the paper's per-state tabular value Vn(i) with a neural network over
    the 11-dim feature vector -- same role as ppo/actor_critic.py's
    value_head, but its own independent network here since it has its
    own optimizer/step-size sequence a(n)."""

    def __init__(self, input_dim: int, hidden_sizes: tuple):
        super().__init__()
        self.net = _mlp(input_dim, hidden_sizes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
