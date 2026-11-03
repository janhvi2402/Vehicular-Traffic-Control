"""
ca/test.py
=============
Single-episode sanity check of a trained CA checkpoint (NOT the evaluation protocol -- that is evaluate.py). Same metrics as dqn/test.py,
ppo/test.py (from the shared eval_common.run_episode; switches are counted from info["executed_actions"]).

    python -m ca.test --checkpoint runs/ca_seed0/actor_final.pt        (or press Play)
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.eval_protocol import default_env_files, run_sanity_check  # noqa: E402
from ca.config import CAConfig  # noqa: E402
from ac.network import ActorNetwork  # noqa: E402
from ca.evaluate import ca_policy_fn, DEFAULT_CHECKPOINT  # noqa: E402


def main():
    net, route = default_env_files("4x4")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--net_file", default=net)
    p.add_argument("--route_file", default=route)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--gui", action="store_true")
    args, _ = p.parse_known_args()
    if not os.path.exists(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    cfg = CAConfig()
    actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions)
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    actor.eval()
    run_sanity_check(ca_policy_fn(actor, "cpu", greedy=not args.stochastic), args, cfg.episode_seconds, cfg.decision_interval)


if __name__ == "__main__":
    main()
