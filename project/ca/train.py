"""
ca/train.py
===========
Trains the average-reward two-timescale critic-actor (ca/agent.py) with the shared online trainer
(environment/online_trainer.py): same seeds, masking, reward clip, truncation handling, budget and logging as DQN, PPO and AC.

    python -m ca.train --seed 0                       # -> runs/ca_seed0/ (500 episodes = 360,000 env steps)
    python -m ca.train --seed 1 --episodes 2 --out runs/smoke_ca
Or press Play (edit DEFAULT_SEED).
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.online_trainer import cli_run  # noqa: E402
from ca.config import CAConfig  # noqa: E402
from ca.agent import CAAgent  # noqa: E402

DEFAULT_SEED = 0                 # Play button: EDIT THIS to train another seed (output folder follows: runs/ca_seed<N>)
DEFAULT_TOTAL_EPISODES = 500     # must equal dqn / ppo / ac


if __name__ == "__main__":
    cli_run("ca", CAConfig, CAAgent, PROJECT_ROOT, default_seed=DEFAULT_SEED, default_episodes=DEFAULT_TOTAL_EPISODES)
