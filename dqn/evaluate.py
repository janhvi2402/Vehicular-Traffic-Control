"""
dqn/evaluate.py
===============
Full evaluation protocol, run over multiple episodes with different random
seeds, comparing the trained DQN against two baselines:

  1. "Random" baseline -- a policy that votes to switch uniformly at
     random each step. This is the direct analogue of the paper's 0%
     reference point: "a policy that selects actions uniformly at random"
     (Methods, Evaluation procedure).

  2. "Fixed-cycle" baseline -- a canonical fixed-time signal controller
     (switches every `fixed_cycle_seconds` regardless of traffic
     conditions), the standard baseline used in traffic-engineering
     practice. This stands in for the paper's "professional human games
     tester" reference point: there is no human-equivalent for signal
     control, so we use the field's own conventional baseline instead.

Following the paper's normalized-performance formula (Fig. 3 / Methods):
    normalized % = 100 * (DQN_score - random_score) / (baseline_score - random_score)
we compute the same ratio using total episode reward as the "score"
(reward is already a higher-is-better quantity here, exactly like the
paper's game score), substituting the fixed-cycle controller for the
paper's human tester.

Each policy is run for `n_eval_episodes` episodes (paper: 30, for up to
5 minutes each with different initial conditions -- we use different SUMO
seeds per episode as our source of varied initial conditions, since our
episodes are full 3600s demand profiles rather than short Atari episodes)
at the paper's evaluation epsilon of 0.05 for the DQN policy.

Usage:
    python -m dqn.evaluate --checkpoint runs/dqn_run1/qnet_final.pt \\
        --episodes 10 --out runs/dqn_run1/eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from environment.eval_common import (  # noqa: E402
    random_policy_fn, fixed_cycle_policy_fn, run_episode,
    run_policy_over_seeds, summarize, plot_comparison,
)
from dqn.config import DQNConfig  # noqa: E402
from dqn.q_network import QNetwork, select_actions_epsilon_greedy  # noqa: E402


# --------------------------------------------------------------------------- #
# DQN policy
# --------------------------------------------------------------------------- #

def dqn_policy_fn(q_net, epsilon, device, rng):
    def _policy(obs):
        return select_actions_epsilon_greedy(q_net, obs["node_features"], epsilon, rng, device)
    return _policy


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--episodes", type=int, default=10, help="paper uses 30; reduce for faster iteration")
    parser.add_argument("--fixed_cycle_seconds", type=float, default=None,
                         help="phase duration for the fixed-time baseline controller. If omitted, "
                              "--sweep_fixed_cycle candidates are swept and the best is used instead.")
    parser.add_argument("--sweep_fixed_cycle", type=float, nargs="+", default=[15, 20, 25, 30, 45, 60],
                         help="candidate fixed-cycle durations (s) to sweep when --fixed_cycle_seconds "
                              "is not given; the candidate with the lowest mean waiting time (on one "
                              "quick probe episode) is selected as the reported fixed-cycle baseline. "
                              "This matters: a fixed-time controller's performance is highly sensitive "
                              "to its cycle length (in our tests, 15s cycles cut mean waiting time by "
                              "more than half versus 30s on the same network/demand) -- comparing DQN "
                              "against an arbitrarily-chosen, untuned cycle length would not be a fair "
                              "or defensible baseline for a paper.")
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--out", default="runs/eval")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = DQNConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    q_net.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file,
        route_file=args.route_file,
        episode_seconds=cfg.episode_seconds,
        decision_interval=cfg.decision_interval,
    )
    env = TrafficGridEnv(env_cfg)
    seeds = [args.base_seed + i for i in range(args.episodes)]

    print(f"Evaluating over {args.episodes} episodes (seeds {seeds[0]}..{seeds[-1]})")

    rng_dqn = np.random.default_rng(args.base_seed)
    dqn_rows = run_policy_over_seeds(env, dqn_policy_fn(q_net, cfg.eval_epsilon, device, rng_dqn), seeds)
    print("DQN done.")

    rng_rand = np.random.default_rng(args.base_seed + 9999)
    random_rows = run_policy_over_seeds(env, random_policy_fn(env.n_agents, rng_rand), seeds)
    print("Random baseline done.")

    if args.fixed_cycle_seconds is not None:
        chosen_cycle = args.fixed_cycle_seconds
    else:
        # Sweep candidates on one probe episode and pick the best-performing
        # cycle length by mean waiting time, so the reported "fixed-cycle"
        # baseline is the strongest fair comparison, not an arbitrary pick.
        print(f"No --fixed_cycle_seconds given; sweeping {args.sweep_fixed_cycle} on one probe episode...")
        probe_seed = seeds[0]
        sweep_results = {}
        for cand in args.sweep_fixed_cycle:
            r = run_episode(env, fixed_cycle_policy_fn(cand, env.cfg.max_green), probe_seed)
            sweep_results[cand] = r["mean_waiting_time"]
            print(f"  cycle={cand:>5.1f}s -> mean_waiting_time={r['mean_waiting_time']:.1f}")
        chosen_cycle = min(sweep_results, key=sweep_results.get)
        print(f"Selected fixed-cycle baseline: {chosen_cycle}s (lowest mean waiting time in sweep)")

    fixed_rows = run_policy_over_seeds(
        env, fixed_cycle_policy_fn(chosen_cycle, env.cfg.max_green), seeds
    )
    print("Fixed-cycle baseline done.")

    env.close()

    summaries = {
        "algo": summarize(dqn_rows),
        "dqn": summarize(dqn_rows),  # kept for backward-compatible key name in this script's own output
        "random": summarize(random_rows),
        "fixed_cycle": summarize(fixed_rows),
        "fixed_cycle_seconds_used": chosen_cycle,
    }

    # Paper's normalized-performance formula (Fig. 3 / Methods), using total
    # reward as the "score" and the fixed-cycle controller in place of the
    # human reference point.
    dqn_r = summaries["dqn"]["total_reward"]["mean"]
    rand_r = summaries["random"]["total_reward"]["mean"]
    fixed_r = summaries["fixed_cycle"]["total_reward"]["mean"]
    denom = (fixed_r - rand_r)
    normalized_pct = 100.0 * (dqn_r - rand_r) / denom if abs(denom) > 1e-9 else float("nan")
    summaries["normalized_performance_pct_vs_fixed_cycle"] = normalized_pct

    # ---- save raw per-episode rows + summary ----
    for name, rows in [("dqn", dqn_rows), ("random", random_rows), ("fixed_cycle", fixed_rows)]:
        with open(os.path.join(args.out, f"{name}_episodes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    # ---- console report ----
    print("\n" + "=" * 72)
    print(f"{'metric':<22s}{'DQN':>15s}{'Fixed-cycle':>17s}{'Random':>15s}")
    print("-" * 72)
    for metric in ["total_reward", "mean_waiting_time", "throughput", "mean_queue_length"]:
        d = summaries["dqn"][metric]
        f_ = summaries["fixed_cycle"][metric]
        r = summaries["random"][metric]
        print(f"{metric:<22s}{d['mean']:>10.1f}±{d['std']:<4.0f}{f_['mean']:>12.1f}±{f_['std']:<4.0f}{r['mean']:>10.1f}±{r['std']:<4.0f}")
    print("=" * 72)
    print(f"Normalized performance vs. fixed-cycle baseline (paper's Fig. 3 formula, "
          f"random=0%, fixed-cycle={chosen_cycle}s=100%): {normalized_pct:.1f}%")
    if fixed_r < rand_r:
        print("CAUTION: the fixed-cycle baseline scored WORSE than the random policy on total reward "
              "for this run -- unlike the paper's Atari setting, there is no guarantee a fixed-time "
              "controller beats random switching, so this normalized percentage can be negative even "
              "when DQN outperforms both baselines on every raw metric. Report the raw metrics above "
              "(mean_waiting_time, mean_queue_length, throughput) as the primary comparison; treat the "
              "normalized percentage as a secondary, paper-methodology-matching statistic only.")
    print(f"\nFull results saved to: {args.out}/summary.json (+ per-episode CSVs)")

    plot_comparison(summaries, args.out, algo_name="DQN")


if __name__ == "__main__":
    main()
