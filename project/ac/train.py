"""
ac/train.py
===========
Trains the two-timescale actor-critic (ac/agent.py) with the shared online trainer (environment/online_trainer.py), so the
training protocol (seeds, masking, reward clip, truncation handling, budget, logging) is the same as for DQN, PPO and CA.

    python -m ac.train --seed 0                      # -> runs/ac_seed0/ (500 episodes = 360,000 env steps)
    python -m ac.train --seed 1 --episodes 2 --out runs/smoke_ac
    python -m ac.train --variant ca --seed 0         # discounted critic-actor ABLATION -> runs/ca_discounted_seed0/
Or press Play (edit DEFAULT_SEED).
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.online_trainer import cli_run  # noqa: E402
from ac.config import ACConfig  # noqa: E402
from ac.agent import ACAgent  # noqa: E402

DEFAULT_SEED = 0                 # Play button: EDIT THIS to train another seed (output folder follows: runs/ac_seed<N>)
DEFAULT_TOTAL_EPISODES = 500     # must equal dqn / ppo (check with scripts/check_matched_budget.py)


def _extra(p):
    p.add_argument("--variant", choices=["ac", "ca"], default="ac",
                   help="'ac' (default, the study's AC) or 'ca' = discounted critic-actor ABLATION (not part of the main comparison)")


if __name__ == "__main__":
    import argparse
    pre = argparse.ArgumentParser(add_help=False); _extra(pre); a, _ = pre.parse_known_args()
    cli_run("ac", ACConfig, ACAgent, PROJECT_ROOT, default_seed=DEFAULT_SEED, default_episodes=DEFAULT_TOTAL_EPISODES,
            default_out_name="ac" if a.variant == "ac" else "ca_discounted", extra_args=_extra,
            cfg_kwargs=lambda args: {"variant": args.variant})
