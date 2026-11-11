"""
environment/eval_protocol.py
============================
The evaluation protocol of ppo/evaluate.py / dqn/evaluate.py as ONE function, used by ac/evaluate.py and ca/evaluate.py, so a
learned policy and every baseline are run exactly as for DQN and PPO:
  seeds base_seed.. (default 1000..1000+episodes-1); random baseline rng = default_rng(base_seed + 9999); tuned fixed-cycle baseline
  (given with --fixed_cycle_seconds, or swept on seeds 2000-2004); always-switch and max-pressure baselines; greedy learned policy.
Writes <out>/summary.json (+ per-episode CSVs, comparison plot, results tables) with the SAME keys as ppo/evaluate.py:
  algo, <algo_key>, random, fixed_cycle, always_switch, max_pressure, fixed_cycle_seconds_used, normalized_performance_pct_vs_fixed_cycle.
The realistic fixed-time plans (90 s / 60 s) are added later by scripts/make_fixed_time_figures.py from runs/realistic_fixed_time.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from environment.eval_common import (  # noqa: E402
    random_policy_fn, fixed_cycle_policy_fn, always_switch_policy_fn, max_pressure_policy_fn, run_episode,
    run_policy_over_seeds, summarize, plot_comparison, build_results_table, print_results_table, save_results_table_csv)
from environment.sumo_info import sumo_info  # noqa: E402

_NETS = {"4x4": ("sumo_4x4_network", ["grid4x4.net.xml", "grid4x4_net.xml"], ["routes.rou.xml", "routes_rou.xml"]),
         "3x3": ("sumo_3x3_network", ["grid3x3.net.xml", "grid3x3_net.xml"], ["routes3x3.rou.xml", "routes3x3_rou.xml"])}


def _first_existing(folder, names):
    for n in names:
        p = os.path.join(PROJECT_ROOT, folder, n)
        if os.path.exists(p):
            return p
    return os.path.join(PROJECT_ROOT, folder, names[0])


def default_env_files(network: str = "4x4"):
    folder, nets, routes = _NETS[network]
    return _first_existing(folder, nets), _first_existing(folder, routes)


def add_standard_args(parser: argparse.ArgumentParser, default_checkpoint: str, default_out: str, network: str = "4x4"):
    net, route = default_env_files(network)
    parser.add_argument("--checkpoint", default=default_checkpoint, help=f"actor checkpoint, default: {default_checkpoint}")
    parser.add_argument("--net_file", default=net)
    parser.add_argument("--route_file", default=route)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true", help="sample from the policy instead of the argmax action")
    parser.add_argument("--fixed_cycle_seconds", type=float, default=None)
    parser.add_argument("--sweep_fixed_cycle", type=float, nargs="+", default=[10, 15, 20, 25, 30, 45, 60])
    parser.add_argument("--base_seed", type=int, default=1000)
    parser.add_argument("--out", default=default_out)
    return parser


def run_standard_evaluation(policy_fn, *, algo_key: str, algo_label: str, args, episode_seconds: int, decision_interval: int):
    os.makedirs(args.out, exist_ok=True)
    env = TrafficGridEnv(GridEnvConfig(net_file=args.net_file, route_file=args.route_file,
                                       episode_seconds=episode_seconds, decision_interval=decision_interval))
    seeds = [args.base_seed + i for i in range(args.episodes)]
    print(f"Evaluating {algo_label} over {args.episodes} episodes (seeds {seeds[0]}..{seeds[-1]})")

    algo_rows = run_policy_over_seeds(env, policy_fn, seeds)
    print(f"{algo_label} done.")
    rng_rand = np.random.default_rng(args.base_seed + 9999)
    random_rows = run_policy_over_seeds(env, random_policy_fn(env.n_agents, rng_rand), seeds)
    print("Random baseline done.")

    if args.fixed_cycle_seconds is not None:
        chosen_cycle = args.fixed_cycle_seconds
    else:
        print(f"No --fixed_cycle_seconds given; sweeping {args.sweep_fixed_cycle} on seeds 2000-2004...")
        sweep_seeds = [2000 + i for i in range(5)]
        sweep = {}
        for cand in args.sweep_fixed_cycle:
            vals = [run_episode(env, fixed_cycle_policy_fn(cand, env.cfg.max_green), s)["mean_waiting_time"] for s in sweep_seeds]
            sweep[cand] = float(np.mean(vals))
            print(f"  cycle={cand:>5.1f}s -> mean_waiting_time={sweep[cand]:.1f}")
        chosen_cycle = min(sweep, key=sweep.get)
        print(f"Selected fixed-cycle baseline: {chosen_cycle}s (lowest mean waiting time in sweep)")
        with open(os.path.join(args.out, "fixed_cycle_sweep.json"), "w") as f:
            json.dump({"seeds": sweep_seeds, "mean_waiting_time_by_cycle": {str(k): v for k, v in sweep.items()},
                       "chosen": chosen_cycle}, f, indent=2)

    fixed_rows = run_policy_over_seeds(env, fixed_cycle_policy_fn(chosen_cycle, env.cfg.max_green), seeds)
    print("Fixed-cycle baseline done.")
    always_rows = run_policy_over_seeds(env, always_switch_policy_fn(env.n_agents), seeds)
    print("Always-switch baseline done.")
    maxp_rows = run_policy_over_seeds(env, max_pressure_policy_fn(), seeds)
    print("Max-pressure baseline done.")
    env.close()

    summaries = {"always_switch": summarize(always_rows), "max_pressure": summarize(maxp_rows),
                 "algo": summarize(algo_rows), algo_key: summarize(algo_rows), "random": summarize(random_rows),
                 "fixed_cycle": summarize(fixed_rows), "fixed_cycle_seconds_used": chosen_cycle, "sumo": sumo_info()}
    algo_r, rand_r, fixed_r = (summaries[k]["total_reward"]["mean"] for k in ("algo", "random", "fixed_cycle"))
    denom = fixed_r - rand_r
    normalized_pct = 100.0 * (algo_r - rand_r) / denom if abs(denom) > 1e-9 else float("nan")
    summaries["normalized_performance_pct_vs_fixed_cycle"] = normalized_pct

    for name, rows in [(algo_key, algo_rows), ("random", random_rows), ("fixed_cycle", fixed_rows),
                       ("always_switch", always_rows), ("max_pressure", maxp_rows)]:
        with open(os.path.join(args.out, f"{name}_episodes.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'metric':<22s}{algo_label:>15s}{'Fixed-cycle':>17s}{'Random':>15s}")
    print("-" * 72)
    for metric in ["total_reward", "mean_waiting_time", "mean_vehicle_wait", "max_vehicle_wait", "throughput", "mean_queue_length"]:
        d, f_, r = summaries["algo"][metric], summaries["fixed_cycle"][metric], summaries["random"][metric]
        print(f"{metric:<22s}{d['mean']:>10.1f}±{d['std']:<4.0f}{f_['mean']:>12.1f}±{f_['std']:<4.0f}{r['mean']:>10.1f}±{r['std']:<4.0f}")
    print("=" * 72)
    print(f"[secondary only] normalized performance on total reward (random=0%, fixed-cycle={chosen_cycle}s=100%): {normalized_pct:.1f}%")
    cols = ["mean_waiting_time", "mean_vehicle_wait", "mean_queue_length", "throughput", "n_switch_votes", "n_forced_switches", "n_teleports", "ended_early"]
    print("\nAll policies (means over eval episodes):")
    print(f"{'policy':<14s}" + "".join(f"{c[:16]:>17s}" for c in cols))
    names = ["algo", "fixed_cycle", "max_pressure", "always_switch", "random"]
    for n in names:
        print(f"{(algo_key if n == 'algo' else n):<14s}" + "".join(f"{summaries[n][c]['mean']:>17.2f}" for c in cols))
    if any(summaries[n]["n_teleports"]["mean"] > 0 or summaries[n]["ended_early"]["mean"] > 0 for n in names):
        print("WARNING: some policy had SUMO teleports or ended early -- report n_teleports and treat its waiting numbers with care.")
    print(f"\nFull results saved to: {args.out}/summary.json (+ per-episode CSVs)")

    plot_comparison(summaries, args.out, algo_name=algo_label)
    wait_table = build_results_table(summaries, algo_name=algo_label, metric="mean_waiting_time")
    print_results_table(wait_table, metric_label="Avg. Waiting Time (s)")
    save_results_table_csv(wait_table, os.path.join(args.out, "results_table_grid_waiting_time.csv"))
    veh_table = build_results_table(summaries, algo_name=algo_label, metric="mean_vehicle_wait")
    print_results_table(veh_table, metric_label="Avg. Wait/Vehicle (s)", title="Per-Vehicle Waiting Time Comparison Against Fixed-Time Baseline")
    save_results_table_csv(veh_table, os.path.join(args.out, "results_table_per_vehicle_wait.csv"))
    return summaries


def run_sanity_check(policy_fn, args, episode_seconds: int, decision_interval: int):
    """Single-episode check (the body of dqn/test.py and ppo/test.py), metrics from the shared run_episode."""
    env = TrafficGridEnv(GridEnvConfig(net_file=args.net_file, route_file=args.route_file, episode_seconds=episode_seconds,
                                       decision_interval=decision_interval, use_gui=args.gui))
    try:
        row = run_episode(env, policy_fn, args.seed)
    finally:
        env.close()
    print("Single-episode sanity check")
    print("-" * 40)
    for k, v in row.items():
        print(f"{k:>26s}: {v}")
    return row
