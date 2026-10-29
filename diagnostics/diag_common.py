"""
diagnostics/diag_common.py
Shared helpers for the manually-run diagnostic scripts in this folder. These scripts
write PNG files only and print key numbers to the console. Each one runs its own
episode with the exact checkpoint given and stamps checkpoint / seed / env-file hash
into the figure footer, so figures from different checkpoints cannot be mixed up.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402

def _first_existing(folder, names):
    for n in names:
        p = os.path.join(PROJECT_ROOT, folder, n)
        if os.path.exists(p):
            return p
    sys.exit(f"ERROR: none of {names} found in {folder}/")


DEFAULT_NET = _first_existing("sumo_4x4_network", ["grid4x4.net.xml", "grid4x4_net.xml"])
DEFAULT_ROUTE = _first_existing("sumo_4x4_network", ["routes.rou.xml", "routes_rou.xml"])

# Play-button defaults (edit these two): used when a script is run without command-line arguments
DEFAULT_ALGO = "dqn"
DEFAULT_CHECKPOINTS = {
    "dqn": os.path.join(PROJECT_ROOT, "runs", "dqn_seed0", "qnet_final.pt"),
    "ppo": os.path.join(PROJECT_ROOT, "runs", "ppo_seed0", "actor_critic_final.pt"),
}


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--algo", default=DEFAULT_ALGO, choices=["dqn", "ppo"])
    p.add_argument("--checkpoint", default=None, help="default: DEFAULT_CHECKPOINTS[algo]")
    p.add_argument("--seed", type=int, default=1000, help="SUMO seed of the episode that is run (eval seeds start at 1000)")
    p.add_argument("--net_file", default=DEFAULT_NET)
    p.add_argument("--route_file", default=DEFAULT_ROUTE)
    p.add_argument("--out", default=None, help="PNG path (default: <checkpoint dir>/figures/<script>.png)")
    _orig = p.parse_known_args

    def _parse_known_args(*a, **k):
        args, rest = _orig(*a, **k)
        if args.checkpoint is None:
            args.checkpoint = DEFAULT_CHECKPOINTS[args.algo]
        if not os.path.exists(args.checkpoint):
            raise SystemExit(f"ERROR: checkpoint not found:\n  {args.checkpoint}\n"
                             f"Train first, or edit DEFAULT_CHECKPOINTS in diagnostics/diag_common.py.")
        return args, rest
    p.parse_known_args = _parse_known_args


def default_out(args, name):
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        return args.out
    d = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "figures")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def file_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def footer(fig, args, extra=""):
    txt = (f"algo={args.algo}  checkpoint={os.path.abspath(args.checkpoint)}  seed={args.seed}  "
           f"env_hash(net+routes)={file_hash(args.net_file, args.route_file)} {extra}")
    fig.text(0.005, 0.004, txt, fontsize=6, color="0.35", ha="left", va="bottom")


def load_policy(algo, checkpoint):
    """Deterministic policy: DQN greedy (epsilon=0), PPO argmax. Same choice as the evaluate.py scripts."""
    import torch  # lazy: compare_algorithms.py does not need torch
    device = "cpu"
    if algo == "dqn":
        from dqn.config import DQNConfig
        from dqn.q_network import QNetwork
        cfg = DQNConfig()
        net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions)
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        net.eval()

        from dqn.q_network import select_actions_epsilon_greedy
        rng = np.random.default_rng(0)

        def policy(obs):   # epsilon=0 (greedy) with the same eligibility masking the trainer/evaluate use
            return select_actions_epsilon_greedy(net, obs["node_features"], 0.0, rng, device)
    else:
        from ppo.config import PPOConfig
        from ppo.actor_critic import ActorCritic
        cfg = PPOConfig()
        net = ActorCritic(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions)
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        net.eval()

        from environment.eligibility import eligibility_from_features

        def policy(obs):   # argmax over masked logits
            x = torch.as_tensor(obs["node_features"], dtype=torch.float32)
            return net.act_greedy(x, torch.as_tensor(eligibility_from_features(obs["node_features"]))).numpy()
    return policy, cfg


def make_env(args, cfg):
    return TrafficGridEnv(GridEnvConfig(net_file=args.net_file, route_file=args.route_file,
                                        episode_seconds=cfg.episode_seconds,
                                        decision_interval=cfg.decision_interval))


def run_logged_episode(env, policy, seed):
    """One episode, logging everything the diagnostics need, all from env events/info
    (never from phase sampled at decision boundaries)."""
    obs, _ = env.reset(seed=seed)
    log = {"greens": [], "queue": [], "wait": [], "feats": [], "mask": [], "actions": [],
           "n_forced_info": 0, "n_honored_info": 0}
    while True:
        act = np.asarray(policy(obs))
        log["feats"].append(obs["node_features"].copy())
        log["mask"].append(obs["action_mask"].copy())
        log["actions"].append(act.copy())
        obs, r, term, trunc, info = env.step(act)
        log["greens"].extend(info["completed_greens"])
        log["queue"].append(info["total_queue_length"])
        log["wait"].append(info["total_waiting_time"])
        log["n_forced_info"] += int(info["forced_switches"].sum())
        log["n_honored_info"] += int(info["honored_switches"].sum())
        log["n_teleports"] = info["n_teleports"]; log["ended_early"] = info["ended_early"]
        if term or trunc:
            break
    log["feats"] = np.asarray(log["feats"]); log["mask"] = np.asarray(log["mask"])
    log["actions"] = np.asarray(log["actions"])
    return log
