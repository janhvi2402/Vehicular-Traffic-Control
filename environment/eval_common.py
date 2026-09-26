"""
environment/eval_common.py
============================
Baseline policies, the episode-runner, and the comparison-plot helper used
by every algorithm's evaluate.py (dqn/evaluate.py, ppo/evaluate.py, and
future actor_critic/critic_actor evaluate.py scripts). Kept here, next to
grid_env.py, for the same reason grid_env.py itself is centralized: every
algorithm must be judged against IDENTICAL baselines and an IDENTICAL
episode-runner, or a "DQN beats PPO" comparison could really just be
measuring differently-implemented evaluation code.

Only algorithm-agnostic logic lives here: baseline policies operate purely
on `obs` / `env` and know nothing about any particular network. Each
algorithm's own evaluate.py supplies its own `<algo>_policy_fn(...)` and
calls into `run_policy_over_seeds` / `summarize` / `plot_comparison` here.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Baseline policies
# --------------------------------------------------------------------------- #

def random_policy_fn(n_agents, rng):
    def _policy(obs):
        return rng.integers(0, 2, size=n_agents)
    return _policy


def fixed_cycle_policy_fn(fixed_cycle_seconds: float, max_green: float):
    """Votes to switch once a phase has been green for >= fixed_cycle_seconds,
    ignoring queue/wait state entirely -- the defining property of a
    real-world fixed-time controller. obs feature index 10 is elapsed
    phase time normalized by max_green; we recover raw seconds from it."""
    def _policy(obs):
        elapsed_norm = obs["node_features"][:, 10]
        elapsed_seconds = elapsed_norm * max_green
        return (elapsed_seconds >= fixed_cycle_seconds).astype(np.int64)
    return _policy


# --------------------------------------------------------------------------- #
# Episode runner (shared by every policy, for a fair comparison)
# --------------------------------------------------------------------------- #

def run_episode(env, policy_fn, seed):
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    n_switch_votes = 0
    n_forced = 0
    queue_sum = 0.0
    wait_sum = 0.0
    n_steps = 0
    while True:
        actions = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        n_switch_votes += int(np.sum(actions))
        n_forced += int(info["forced_switches"].sum())
        # UNCAPPED grid totals from info, not reconstructed from
        # node_features. node_features[:, 0:4]/[:, 4:8] are clipped to
        # [0,1] before *_NORM in _get_obs() (see TrafficGridEnv), so
        # summing them back up silently caps each junction's contribution
        # at QUEUE_NORM/WAIT_NORM and understates real congestion once any
        # junction exceeds that -- exactly the case for a bad checkpoint,
        # the random baseline, or any run with sustained congestion, which
        # would previously look artificially closer to a good policy than
        # it really was. info["total_queue_length"]/["total_waiting_time"]
        # come straight from TrafficGridEnv's own per-junction accumulators
        # and carry no such cap.
        queue_sum += info["total_queue_length"]
        wait_sum += info["total_waiting_time"]
        n_steps += 1
        if terminated or truncated:
            break
    return {
        "total_reward": total_reward,
        # averaged over every decision step in the episode, NOT a final-step
        # snapshot -- a policy can't look artificially good by draining the
        # queue right at the end; this is the metric to report as "waiting
        # time" in the paper.
        "mean_waiting_time": wait_sum / n_steps,
        "final_step_waiting_time": info["total_waiting_time"],
        "throughput": info["throughput"],
        "mean_queue_length": queue_sum / n_steps,
        "n_switch_votes": n_switch_votes,
        "n_forced_switches": n_forced,
        "n_steps": n_steps,
        # PER-VEHICLE metrics -- how long an individual car actually waited,
        # not a per-step grid-wide sum like mean_waiting_time above. Read
        # from the final step's info since these accumulate over the whole
        # episode already (same pattern as "throughput"). NOTE: a vehicle
        # still on the road when the episode truncates (didn't arrive in
        # time) contributes NOTHING here -- same undercount risk flagged
        # elsewhere for throughput -- so n_vehicles_completed matters: if
        # it's well below the total number of vehicles the route file
        # injects, these two numbers are a partial-episode picture only.
        "mean_vehicle_wait": info["mean_vehicle_wait"],
        "max_vehicle_wait": info["max_vehicle_wait"],
        "n_vehicles_completed": info["n_vehicles_completed"],
    }


def run_policy_over_seeds(env, policy_fn, seeds):
    rows = []
    for s in seeds:
        rows.append(run_episode(env, policy_fn, s))
    return rows


def summarize(rows):
    keys = rows[0].keys()
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return summary


# --------------------------------------------------------------------------- #
# Comparison plot
# --------------------------------------------------------------------------- #

def plot_comparison(summaries: dict, out_dir: str, algo_name: str = "DQN"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = ["random", "fixed_cycle", "algo"]
    labels = ["Random", "Fixed-cycle", algo_name]
    colors = ["#993556", "#854F0B", "#0F6E56"]
    metrics = [
        ("mean_waiting_time", "Mean waiting time per step (s)\n(grid-wide, per-step sum)"),
        ("mean_vehicle_wait", "Mean wait per vehicle (s)\n(per-vehicle, whole trip)"),
        ("throughput", "Throughput (vehicles arrived)"),
        ("mean_queue_length", "Mean queue length (vehicles)"),
        ("total_reward", "Total episode reward"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        means = [summaries[p][metric]["mean"] for p in policies]
        stds = [summaries[p][metric]["std"] for p in policies]
        ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle(f"{algo_name} vs. baselines (mean ± std over evaluation episodes)", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "baseline_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to: {out_path}")


# --------------------------------------------------------------------------- #
# Report-style results table
# --------------------------------------------------------------------------- #
# Mirrors the "Overall Performance Comparison Against Fixed-Time Baseline"
# table from the earlier 2-junction report (Method / Avg. Waiting Time (s) /
# Std. Dev. (s) / Improvement (%)) so results from this 4x4/3x3 grid work
# can be dropped into the same table format. NOTE this uses the simple,
# directly-interpretable formula from that report --
#     Improvement % = 100 * (fixed_cycle_mean - algo_mean) / fixed_cycle_mean
# -- which is NOT the same number as evaluate.py's
# normalized_performance_pct_vs_fixed_cycle (that one anchors 0% at the
# RANDOM baseline's reward, per the Mnih et al. Atari methodology, and is
# computed from total_reward rather than waiting time). Both are valid;
# report whichever your reader expects, and don't mix the two numbers
# together as if they were the same statistic.

def improvement_pct(baseline_mean: float, algo_mean: float) -> float:
    if abs(baseline_mean) < 1e-9:
        return float("nan")
    return 100.0 * (baseline_mean - algo_mean) / baseline_mean


def build_results_table(summaries: dict, algo_name: str = "DQN", metric: str = "mean_waiting_time"):
    """One row per method: mean/std of `metric`, plus % improvement over the
    fixed-cycle baseline using the report's formula above. `metric` defaults
    to mean_waiting_time (grid-wide per-step average, what the earlier
    report called "Avg. Waiting Time"); pass metric="mean_vehicle_wait" for
    the per-vehicle whole-trip version instead -- they are NOT the same
    number, see grid_env.py's _update_vehicle_wait_tracking docstring."""
    baseline_mean = summaries["fixed_cycle"][metric]["mean"]
    order = [("fixed_cycle", "Fixed-Time Baseline"), ("random", "Random"), ("algo", algo_name)]
    rows = []
    for key, label in order:
        m = summaries[key][metric]["mean"]
        s = summaries[key][metric]["std"]
        pct = None if key == "fixed_cycle" else improvement_pct(baseline_mean, m)
        rows.append({"method": label, "avg_waiting_time_s": m, "std_dev_s": s, "improvement_pct": pct})
    return rows


def print_results_table(rows, metric_label: str = "Avg. Waiting Time (s)",
                         title: str = "Overall Performance Comparison Against Fixed-Time Baseline"):
    print(f"\n{title}")
    header = f"{'Method':<22s}{metric_label:>24s}{'Std. Dev. (s)':>16s}{'Improvement (%)':>18s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        pct_str = "-" if r["improvement_pct"] is None else f"{r['improvement_pct']:.1f}"
        print(f"{r['method']:<22s}{r['avg_waiting_time_s']:>24.2f}{r['std_dev_s']:>16.2f}{pct_str:>18s}")


def save_results_table_csv(rows, out_path):
    import csv
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results table saved to: {out_path}")


# --------------------------------------------------------------------------- #
# Multi-algorithm comparison (Q-Learning, DQN, PPO, PPO_Routing, ...)
# --------------------------------------------------------------------------- #
# Combines several ALREADY-RUN evaluate.py summary.json files into the one
# master table (Table 8.6 shape). Deliberately takes exactly ONE baseline
# summary rather than one-per-algorithm: if each algorithm's evaluate.py
# independently swept --sweep_fixed_cycle and landed on a different chosen
# cycle length, "Improvement (%)" would be computed against a different
# denominator per row -- the numbers would print side by side but would not
# actually be comparable. Reuse one run's fixed_cycle/random result (e.g.
# the DQN run's summary.json, or a dedicated baseline-only run) for every
# row instead.

def build_multi_algo_table(baseline_summary: dict, algo_summaries: dict,
                            metric: str = "mean_waiting_time"):
    """
    baseline_summary: one LOADED summary.json dict -- supplies the single
        "Fixed-Time Baseline" and "Random" rows for the whole table.
    algo_summaries: {label: summary_dict}, one per algorithm (e.g.
        {"Q-Learning": ..., "DQN": ..., "PPO": ..., "PPO_Routing": ...}).
        Each summary_dict's ["algo"][metric] is used for that row. All of
        these, and baseline_summary, must come from evaluate.py runs against
        the SAME net_file/route_file/seeds -- otherwise the rows describe
        different conditions, not different algorithms under the same one.
    Returns rows in the same shape as build_results_table's output, so
    print_results_table / save_results_table_csv work unchanged on either.
    """
    baseline_mean = baseline_summary["fixed_cycle"][metric]["mean"]
    rows = [
        {
            "method": "Fixed-Time Baseline",
            "avg_waiting_time_s": baseline_summary["fixed_cycle"][metric]["mean"],
            "std_dev_s": baseline_summary["fixed_cycle"][metric]["std"],
            "improvement_pct": None,
        },
        {
            "method": "Random",
            "avg_waiting_time_s": baseline_summary["random"][metric]["mean"],
            "std_dev_s": baseline_summary["random"][metric]["std"],
            "improvement_pct": improvement_pct(baseline_mean, baseline_summary["random"][metric]["mean"]),
        },
    ]
    for label, summ in algo_summaries.items():
        m = summ["algo"][metric]["mean"]
        s = summ["algo"][metric]["std"]
        rows.append({
            "method": label,
            "avg_waiting_time_s": m,
            "std_dev_s": s,
            "improvement_pct": improvement_pct(baseline_mean, m),
        })
    return rows


def plot_multi_algo_comparison(rows, out_path, metric_label: str = "Avg. Waiting Time (s)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["method"] for r in rows]
    means = [r["avg_waiting_time_s"] for r in rows]
    stds = [r["std_dev_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(1.8 * len(rows) + 2, 5))
    ax.bar(labels, means, yerr=stds, capsize=5, color="#0F6E56")
    ax.set_ylabel(metric_label)
    ax.set_title("Overall Performance Comparison Against Fixed-Time Baseline")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")