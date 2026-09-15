"""
ac/test.py
==========
A quick, single-episode sanity check for a trained AC or CA actor
checkpoint -- NOT the full evaluation protocol (that's evaluate.py).
Mirrors dqn/test.py and ppo/test.py exactly.

Usage:
    python -m ac.test --checkpoint runs/ac_run1/actor_final.pt
    python -m ac.test --checkpoint runs/ca_run1/actor_final.pt --stochastic
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from ac.config import ACConfig  # noqa: E402
from ac.network import ActorNetwork  # noqa: E402


def run_episode(actor, env, seed, device, greedy=True):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_switches = 0
    n_forced = 0
    wait_sum = 0.0
    steps = 0
    while True:
        x = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        if greedy:
            actions = actor.act_greedy(x).cpu().numpy()
        else:
            actions, _, _ = actor.act(x)
            actions = actions.cpu().numpy()
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to an actor_*.pt checkpoint")
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true",
                         help="sample from the policy instead of taking the argmax action")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    cfg = ACConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
    actor.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file, route_file=args.route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
        use_gui=args.gui,
    )
    env = TrafficGridEnv(env_cfg)

    result = run_episode(actor, env, args.seed, device, greedy=not args.stochastic)
    env.close()

    print("Single-episode sanity check")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k:>22s}: {v}")


if __name__ == "__main__":
    main()