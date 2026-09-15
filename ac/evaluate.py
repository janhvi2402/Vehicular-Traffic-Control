"""
ac/evaluate.py
==============
Full evaluation protocol for a trained AC or CA actor checkpoint, run over
multiple episodes with different random seeds, against the same two
baselines used for DQN and PPO (see environment/eval_common.py) -- random
and swept fixed-cycle -- so results are directly comparable across all
four algorithms.

Usage:
    python -m ac.evaluate --checkpoint runs/ac_run1/actor_final.pt \\
        --algo_name AC --episodes 10 --out runs/ac_run1/eval

    python -m ac.evaluate --checkpoint runs/ca_run1/actor_final.pt \\
        --algo_name CA --episodes 10 --out runs/ca_run1/eval

    # generalization test on the 3x3 network (same pattern as DQN/PPO's):
    python -m ac.evaluate --checkpoint runs/ac_run1/actor_final.pt \\
        --algo_name AC \\
        --net_file sumo_3x3_network/grid3x3.net.xml \\
        --route_file sumo_3x3_network/routes3x3.rou.xml \\
        --episodes 2 --out runs/ac_run1/eval_3x3
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
from ac.config import ACConfig  # noqa: E402
from ac.network import ActorNetwork  # noqa: E402


def ac_policy_fn(actor, device, greedy=True):
    def _policy(obs):
        x = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        if greedy:
            return actor.act_greedy(x).cpu().numpy()
        actions, _, _ = actor.act(x)
        return actions.cpu().numpy()
    return _policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to an actor_*.pt checkpoint")
    parser.add_argument("--algo_name", default="AC", help="label used in printed report and plot legend, e.g. 'AC' or 'CA'")
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true",
                         help="sample from the policy instead of taking the argmax action at eval time")
    parser.add_argument("--fixed_cycle_seconds", type=float, default=None)
    parser.add_argument("--sweep_fixed_cycle", type=float, nargs="+", default=[15, 20, 25, 30, 45, 60])
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--out", default="runs/eval")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = ACConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
    actor.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file, route_file=args.route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
    )
    env = TrafficGridEnv(env_cfg)
    seeds = [args.base_seed + i for i in range(args.episodes)]

    print(f"Evaluating {args.algo_name} over {args.episodes} episodes (seeds {seeds[0]}..{seeds[-1]})")

    algo_rows = run_policy_over_seeds(env, ac_policy_fn(actor, device, greedy=not args.stochastic), seeds)
    print(f"{args.algo_name} done.")

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
        "algo": summarize(algo_rows),
        "random": summarize(random_rows),
        "fixed_cycle": summarize(fixed_rows),
        "fixed_cycle_seconds_used": chosen_cycle,
        "algo_name": args.algo_name,
    }

    algo_r = summaries["algo"]["total_reward"]["mean"]
    rand_r = summaries["random"]["total_reward"]["mean"]
    fixed_r = summaries["fixed_cycle"]["total_reward"]["mean"]
    denom = fixed_r - rand_r
    normalized_pct = 100.0 * (algo_r - rand_r) / denom if abs(denom) > 1e-9 else float("nan")
    summaries["normalized_performance_pct_vs_fixed_cycle"] = normalized_pct

    for name, rows in [("algo", algo_rows), ("random", random_rows), ("fixed_cycle", fixed_rows)]:
        with open(os.path.join(args.out, f"{name}_episodes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'metric':<22s}{args.algo_name:>15s}{'Fixed-cycle':>17s}{'Random':>15s}")
    print("-" * 72)
    for metric in ["total_reward", "mean_waiting_time", "throughput", "mean_queue_length"]:
        d = summaries["algo"][metric]
        f_ = summaries["fixed_cycle"][metric]
        r = summaries["random"][metric]
        print(f"{metric:<22s}{d['mean']:>10.1f}±{d['std']:<4.0f}{f_['mean']:>12.1f}±{f_['std']:<4.0f}{r['mean']:>10.1f}±{r['std']:<4.0f}")
    print("=" * 72)
    print(f"Normalized performance vs. fixed-cycle baseline (random=0%, fixed-cycle={chosen_cycle}s=100%): "
          f"{normalized_pct:.1f}%")
    if fixed_r < rand_r:
        print("CAUTION: the fixed-cycle baseline scored WORSE than the random policy on total reward "
              "for this run -- report the raw metrics above as the primary comparison; treat the "
              "normalized percentage as a secondary statistic only.")
    print(f"\nFull results saved to: {args.out}/summary.json (+ per-episode CSVs)")

    plot_comparison(summaries, args.out, algo_name=args.algo_name)


if __name__ == "__main__":
    main()
