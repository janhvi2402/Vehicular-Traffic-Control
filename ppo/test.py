"""
ppo/test.py
===========
A quick, single-episode sanity check for a trained PPO checkpoint -- NOT
the full evaluation protocol (that's evaluate.py). Mirrors dqn/test.py.

Usage:
    python -m ppo.test --checkpoint runs/ppo_run_matched500/actor_critic_final.pt

    Or just hit Play/F5 with zero arguments -- DEFAULT_CHECKPOINT below
    points at runs/ppo_run_matched500/actor_critic_final.pt already.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Project root = the folder that CONTAINS this "ppo" package. Mirrors the
# pattern used throughout dqn/*.py, so the VS Code Play button works here
# too regardless of the working directory VS Code happens to launch from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from ppo.config import PPOConfig  # noqa: E402
from ppo.actor_critic import ActorCritic  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", "ppo_run_matched500", "actor_critic_final.pt")
DEFAULT_NET_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml")


def run_episode(net, env, seed, device, greedy=True):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_switches = 0
    n_forced = 0
    wait_sum = 0.0
    steps = 0
    while True:
        x = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        if greedy:
            actions = net.act_greedy(x).cpu().numpy()
        else:
            actions, _, _ = net.act(x)
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
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--net_file", default=DEFAULT_NET_FILE)
    parser.add_argument("--route_file", default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true",
                         help="sample from the policy instead of taking the argmax action")
    parser.add_argument("--gui", action="store_true")
    # parse_known_args (not parse_args) so the VS Code Play button, which
    # passes zero arguments, never errors out here -- matches dqn/test.py.
    args, _unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at:\n  {args.checkpoint}\n"
              f"Edit DEFAULT_CHECKPOINT near the top of this file to point at your "
              f"actual checkpoint, or pass --checkpoint <path> if you're running from a terminal.")
        sys.exit(1)

    cfg = PPOConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ActorCritic(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file, route_file=args.route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
        use_gui=args.gui,
    )
    env = TrafficGridEnv(env_cfg)

    result = run_episode(net, env, args.seed, device, greedy=not args.stochastic)
    env.close()

    print("Single-episode sanity check")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k:>22s}: {v}")


if __name__ == "__main__":
    main()