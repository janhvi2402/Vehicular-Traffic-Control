"""
dqn/q_network.py
=================
The shared Q-network and replay buffer used by train.py / test.py /
evaluate.py. This implements vanilla DQN exactly as specified in
Algorithm 1 of Mnih et al., 2015 (experience replay + a periodically
-synced target network, no Double-DQN / Dueling / PER extensions, which
postdate this paper) -- adapted from per-pixel CNN inputs to the 11-dim
per-junction feature vector produced by TrafficGridEnv, and from a single
Atari agent to a parameter-shared network applied independently to all 16
junctions (see USAGE.md's "shared Q-network" design).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Maps one junction's local 11-dim observation to Q(hold), Q(switch).

    Paper's stated architectural principle (kept here even though the
    input is a feature vector, not pixels): "there is a separate output
    unit for each possible action, and only the state representation is
    an input to the neural network" -- i.e. one forward pass yields
    Q-values for every action, rather than requiring one pass per action.
    Hidden layers use ReLU ("rectifier nonlinearity"), matching the paper.
    """

    def __init__(self, input_dim: int, hidden_sizes: tuple, n_actions: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniform-random experience replay, exactly as in the paper: "sample
    random minibatch of transitions ... from D, drawn uniformly at random
    from the pool of stored samples." Fixed capacity, oldest transitions
    are overwritten first (a deque), matching the paper's finite-memory
    behavior ("stores the last N experience tuples... always overwrites
    with recent transitions owing to the finite memory size N")."""

    def __init__(self, capacity: int, seed: int = 0):
        self.buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buffer, batch_size)
        obs = np.stack([t.obs for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return obs, actions, rewards, next_obs, dones


def select_actions_epsilon_greedy(
    q_net: QNetwork, node_features: np.ndarray, epsilon: float, rng: np.random.Generator, device: str
) -> np.ndarray:
    """Epsilon-greedy action selection applied independently to every
    junction using the ONE shared Q-network (parameter-sharing /
    independent-DQN scaling described in USAGE.md). node_features has
    shape (n_agents, input_dim). Returns an (n_agents,) int array, ready
    to hand directly to TrafficGridEnv.step() as the signal-vote portion
    of the action."""
    n_agents = node_features.shape[0]
    actions = np.zeros(n_agents, dtype=np.int64)
    explore_mask = rng.random(n_agents) < epsilon
    n_explore = int(explore_mask.sum())
    if n_explore > 0:
        actions[explore_mask] = rng.integers(0, q_net.net[-1].out_features, size=n_explore)
    n_greedy = n_agents - n_explore
    if n_greedy > 0:
        with torch.no_grad():
            x = torch.as_tensor(node_features[~explore_mask], dtype=torch.float32, device=device)
            q_values = q_net(x)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        actions[~explore_mask] = greedy_actions
    return actions