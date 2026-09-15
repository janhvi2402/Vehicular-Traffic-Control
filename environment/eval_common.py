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
        ("mean_waiting_time", "Mean waiting time per step (s)"),
        ("throughput", "Throughput (vehicles arrived)"),
        ("mean_queue_length", "Mean queue length (vehicles)"),
        ("total_reward", "Total episode reward"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
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