"""
ppo/evaluate.py
===============
Full evaluation protocol for a trained PPO checkpoint, run over multiple
episodes with different random seeds, against the same two baselines used
for DQN (see environment/eval_common.py) -- random and swept fixed-cycle
-- so results are directly comparable across algorithms.

Usage:
    python -m ppo.evaluate --checkpoint runs/ppo_run_matched500/actor_critic_final.pt \\
        --episodes 10 --out runs/ppo_run_matched500/eval

    # generalization test on the 3x3 network (same pattern as DQN's):
    python -m ppo.evaluate --checkpoint runs/ppo_run_matched500/actor_critic_final.pt \\
        --net_file sumo_3x3_network/grid3x3_net.xml \\
        --route_file sumo_3x3_network/routes3x3_rou.xml \\
        --episodes 2 --out runs/ppo_run_matched500/eval_3x3

    Or just hit Play/F5 with zero arguments -- DEFAULT_CHECKPOINT below
    points at runs/ppo_run_matched500/actor_critic_final.pt already.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

# Project root = the folder that CONTAINS this "ppo" package. Mirrors the
# pattern used throughout dqn/*.py, so the VS Code Play button works here
# too regardless of the working directory VS Code happens to launch from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from environment.eval_common import (  # noqa: E402
    random_policy_fn, fixed_cycle_policy_fn, run_episode,
    run_policy_over_seeds, summarize, plot_comparison,
    build_results_table, print_results_table, save_results_table_csv,
)
from ppo.config import PPOConfig  # noqa: E402
from ppo.actor_critic import ActorCritic  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", "ppo_run_matched500", "actor_critic_final.pt")
DEFAULT_NET_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "runs", "ppo_run_matched500", "eval")


def ppo_policy_fn(net, device, greedy=True):
    def _policy(obs):
        x = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        if greedy:
            return net.act_greedy(x).cpu().numpy()
        actions, _, _ = net.act(x)
        return actions.cpu().numpy()
    return _policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--net_file", default=DEFAULT_NET_FILE)
    parser.add_argument("--route_file", default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--episodes", type=int, default=10, help="paper uses 3 seeds x multiple envs; reduce for faster iteration")
    parser.add_argument("--stochastic", action="store_true",
                         help="sample from the policy instead of taking the argmax action at eval time")
    parser.add_argument("--fixed_cycle_seconds", type=float, default=None)
    parser.add_argument("--sweep_fixed_cycle", type=float, nargs="+", default=[15, 20, 25, 30, 45, 60])
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--out", default=DEFAULT_OUT)
    # parse_known_args (not parse_args) so the VS Code Play button, which
    # passes zero arguments, never errors out here -- matches dqn/evaluate.py.
    args, _unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at:\n  {args.checkpoint}\n"
              f"Edit DEFAULT_CHECKPOINT near the top of this file, or pass "
              f"--checkpoint <path> if you're running from a terminal.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    cfg = PPOConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ActorCritic(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file, route_file=args.route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
    )
    env = TrafficGridEnv(env_cfg)
    seeds = [args.base_seed + i for i in range(args.episodes)]

    print(f"Evaluating over {args.episodes} episodes (seeds {seeds[0]}..{seeds[-1]})")

    ppo_rows = run_policy_over_seeds(env, ppo_policy_fn(net, device, greedy=not args.stochastic), seeds)
    print("PPO done.")

    rng_rand = np.random.default_rng(args.base_seed + 9999)
    random_rows = run_policy_over_seeds(env, random_policy_fn(env.n_agents, rng_rand), seeds)
    print("Random baseline done.")

    if args.fixed_cycle_seconds is not None:
        chosen_cycle = args.fixed_cycle_seconds
    else:
        print(f"No --fixed_cycle_seconds given; sweeping {args.sweep_fixed_cycle} on one probe episode...")
        probe_seed = seeds[0]
        sweep_results = {}
        for cand in args.sweep_fixed_cycle:
            r = run_episode(env, fixed_cycle_policy_fn(cand, env.cfg.max_green), probe_seed)
            sweep_results[cand] = r["mean_waiting_time"]
            print(f"  cycle={cand:>5.1f}s -> mean_waiting_time={r['mean_waiting_time']:.1f}")
        chosen_cycle = min(sweep_results, key=sweep_results.get)
        print(f"Selected fixed-cycle baseline: {chosen_cycle}s (lowest mean waiting time in sweep)")

    fixed_rows = run_policy_over_seeds(env, fixed_cycle_policy_fn(chosen_cycle, env.cfg.max_green), seeds)
    print("Fixed-cycle baseline done.")

    env.close()

    summaries = {
        "algo": summarize(ppo_rows),
        "ppo": summarize(ppo_rows),
        "random": summarize(random_rows),
        "fixed_cycle": summarize(fixed_rows),
        "fixed_cycle_seconds_used": chosen_cycle,
    }

    ppo_r = summaries["ppo"]["total_reward"]["mean"]
    rand_r = summaries["random"]["total_reward"]["mean"]
    fixed_r = summaries["fixed_cycle"]["total_reward"]["mean"]
    denom = fixed_r - rand_r
    normalized_pct = 100.0 * (ppo_r - rand_r) / denom if abs(denom) > 1e-9 else float("nan")
    summaries["normalized_performance_pct_vs_fixed_cycle"] = normalized_pct

    for name, rows in [("ppo", ppo_rows), ("random", random_rows), ("fixed_cycle", fixed_rows)]:
        with open(os.path.join(args.out, f"{name}_episodes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'metric':<22s}{'PPO':>15s}{'Fixed-cycle':>17s}{'Random':>15s}")
    print("-" * 72)
    for metric in ["total_reward", "mean_waiting_time", "mean_vehicle_wait", "max_vehicle_wait",
                   "throughput", "mean_queue_length"]:
        d = summaries["ppo"][metric]
        f_ = summaries["fixed_cycle"][metric]
        r = summaries["random"][metric]
        print(f"{metric:<22s}{d['mean']:>10.1f}±{d['std']:<4.0f}{f_['mean']:>12.1f}±{f_['std']:<4.0f}{r['mean']:>10.1f}±{r['std']:<4.0f}")
    print("=" * 72)
    for name in ["ppo", "fixed_cycle", "random"]:
        n_done = summaries[name]["n_vehicles_completed"]["mean"]
        print(f"  [{name}] vehicles completed their trip this episode (mean): {n_done:.0f} "
              f"-- mean_vehicle_wait/max_vehicle_wait above are computed ONLY over these; a "
              f"vehicle still on the road when the episode ends is not counted.")
    print(f"Normalized performance vs. fixed-cycle baseline (paper's Fig. 3-style formula, "
          f"random=0%, fixed-cycle={chosen_cycle}s=100%): {normalized_pct:.1f}%")
    if fixed_r < rand_r:
        print("CAUTION: the fixed-cycle baseline scored WORSE than the random policy on total reward "
              "for this run -- report the raw metrics above as the primary comparison; treat the "
              "normalized percentage as a secondary statistic only.")
    print(f"\nFull results saved to: {args.out}/summary.json (+ per-episode CSVs)")

    plot_comparison(summaries, args.out, algo_name="PPO")

    # ---- report-style tables (Method / Avg. Waiting Time / Std Dev / Improvement %),
    # matching dqn/evaluate.py's tables exactly, so both algorithms' results
    # can be tabulated the same way for the report (Table 8.6-style).
    wait_table = build_results_table(summaries, algo_name="PPO", metric="mean_waiting_time")
    print_results_table(wait_table, metric_label="Avg. Waiting Time (s)")
    save_results_table_csv(wait_table, os.path.join(args.out, "results_table_grid_waiting_time.csv"))

    veh_table = build_results_table(summaries, algo_name="PPO", metric="mean_vehicle_wait")
    print_results_table(veh_table, metric_label="Avg. Wait/Vehicle (s)",
                         title="Per-Vehicle Waiting Time Comparison Against Fixed-Time Baseline")
    save_results_table_csv(veh_table, os.path.join(args.out, "results_table_per_vehicle_wait.csv"))


if __name__ == "__main__":
    main()