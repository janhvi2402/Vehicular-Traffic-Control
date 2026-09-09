"""
dqn/test.py
===========
A quick, single-episode sanity check for a trained checkpoint -- NOT the
paper-style evaluation protocol (that's evaluate.py). Use this to confirm
a checkpoint loads and behaves sensibly (no crashes, reasonable waiting
times, plausible switch counts) before running the full evaluation.

Usage:
    python -m dqn.test --checkpoint runs/dqn_run1/qnet_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.q_network import QNetwork, select_actions_epsilon_greedy  # noqa: E402


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
        wait_sum += float(obs["node_features"][:, 4:8].sum()) * env.WAIT_NORM
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

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

    # paper's evaluation protocol: fixed low epsilon (0.05), not fully greedy
    result = run_episode(q_net, env, cfg.eval_epsilon, args.seed, device, rng)
    env.close()

    print("Single-episode sanity check")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k:>22s}: {v}")


if __name__ == "__main__":
    main()
