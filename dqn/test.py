"""
dqn/test.py
===========
A quick, single-episode sanity check for a trained checkpoint -- NOT the
paper-style evaluation protocol (that's evaluate.py). Use this to confirm
a checkpoint loads and behaves sensibly (no crashes, reasonable waiting
times, plausible switch counts) before running the full evaluation.

Usage (from a terminal, if you have one):
    python -m dqn.test --checkpoint runs/dqn_run_matched500/qnet_final.pt

Or just press the "Run Python File" / Play button in VS Code with no
arguments at all -- every argument below has a hardcoded default (edit
the DEFAULT_* constants if your checkpoint/network files live somewhere
else), and every path is resolved relative to the project root (the
folder that contains this "dqn" folder), not to whatever the current
working directory happens to be. That's what makes the Play button work
the same way no matter where VS Code decides to launch the script from.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Project root = the folder that CONTAINS this "dqn" package (one level
# up from this file). Every default path below is built from this, so
# the script behaves identically whether it's launched with the Play
# button, "python dqn/test.py", or "python -m dqn.test" -- regardless of
# what directory VS Code happens to set as the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.q_network import QNetwork, select_actions_epsilon_greedy  # noqa: E402

# ------------------------------------------------------------------ #
# EDIT THESE IF YOUR FILES LIVE SOMEWHERE ELSE
# ------------------------------------------------------------------ #
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", "dqn_run_matched500", "qnet_final.pt")
DEFAULT_NET_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml")


def run_episode(q_net, env, epsilon, seed, device, rng):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_switches = 0
    n_forced = 0
    wait_sum = 0.0
    steps = 0
    while True:
        actions = select_actions_epsilon_greedy(q_net, obs["node_features"], epsilon, rng, device)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        n_switches += int(actions.sum())
        n_forced += int(info["forced_switches"].sum())
        # UNCAPPED, from TrafficGridEnv's own accumulator -- NOT
        # obs["node_features"][:, 4:8].sum()*env.WAIT_NORM, which is
        # clipped to [0,1] before WAIT_NORM in _get_obs() and silently
        # caps each junction's contribution at WAIT_NORM.
        wait_sum += info["total_waiting_time"]
        steps += 1
        if terminated or truncated:
            break
    return {
        "steps": steps,
        "total_reward": total_reward,
        "mean_waiting_time": wait_sum / steps,
        "final_step_waiting_time": info["total_waiting_time"],
        "throughput": info["throughput"],
        "n_switch_votes": n_switches,
        "n_forced_switches": n_forced,
        # per-vehicle, whole-trip wait -- see grid_env.py's
        # _update_vehicle_wait_tracking for what this actually measures
        "mean_vehicle_wait": info["mean_vehicle_wait"],
        "max_vehicle_wait": info["max_vehicle_wait"],
        "n_vehicles_completed": info["n_vehicles_completed"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--net_file", default=DEFAULT_NET_FILE)
    parser.add_argument("--route_file", default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--seed", type=int, default=123)
    # NOTE: default 0.05 deliberately keeps Mnih et al.'s original
    # evaluation protocol for this quick sanity check -- unlike
    # dqn/evaluate.py, which defaults to 0.0 specifically so its numbers
    # are fair to compare against ppo/evaluate.py (see the fairness note
    # there). ppo/test.py is always fully greedy with no such flag, so
    # DO NOT compare this script's printed numbers against ppo/test.py's
    # -- they use different exploration noise by default. Only
    # dqn/evaluate.py vs ppo/evaluate.py is an apples-to-apples
    # comparison. Pass --eval_epsilon 0.0 here if you specifically want
    # this script's output to be greedy-comparable to ppo/test.py anyway.
    parser.add_argument("--eval_epsilon", type=float, default=0.05,
                         help="epsilon for epsilon-greedy action selection. Default 0.05 "
                              "matches Mnih et al.'s evaluation protocol for this single-"
                              "checkpoint sanity check -- see the note above this argument "
                              "for why this differs from dqn/evaluate.py's default and why "
                              "you shouldn't compare this script's output directly against "
                              "ppo/test.py's without passing --eval_epsilon 0.0.")
    parser.add_argument("--gui", action="store_true")
    # parse_known_args (not parse_args) so the script never errors out when
    # VS Code's Play button runs it with zero arguments.
    args, _unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at:\n  {args.checkpoint}\n"
              f"Edit DEFAULT_CHECKPOINT near the top of this file to point at your "
              f"actual checkpoint (e.g. runs/dqn_run_matched500/qnet_final.pt), or pass "
              f"--checkpoint <path> if you're running from a terminal.")
        sys.exit(1)

    cfg = DQNConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    q_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    q_net.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file,
        route_file=args.route_file,
        episode_seconds=cfg.episode_seconds,
        decision_interval=cfg.decision_interval,
        use_gui=args.gui,
    )
    env = TrafficGridEnv(env_cfg)

    if args.eval_epsilon != 0.0:
        print(f"NOTE: running with eval_epsilon={args.eval_epsilon} (not fully greedy) -- "
              f"do not compare this run's numbers against ppo/test.py, which is always "
              f"greedy. Use dqn/evaluate.py vs ppo/evaluate.py for a fair comparison.")
    result = run_episode(q_net, env, args.eval_epsilon, args.seed, device, rng)
    env.close()

    print("Single-episode sanity check")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k:>22s}: {v}")


if __name__ == "__main__":
    main()