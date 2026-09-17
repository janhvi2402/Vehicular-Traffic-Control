"""
dqn/switch_timing_plot.py
==========================
Runs one evaluation episode and visualizes exactly how the signal-control
policy behaves over time:

  1. A Gantt/swimlane chart, one row per junction, showing green vs.
     yellow intervals across the whole episode -- this is the direct
     answer to "how is the model switching the lights".
  2. A histogram of the green-phase durations the policy actually chose,
     with reference lines at min_green and max_green (the hard cap) so
     it's visible whether the policy is switching proactively or just
     riding the cap.
  3. A grid-wide time series of total queue length and total waiting time
     across the episode, to see how congestion evolves under the policy.
  4. A per-junction bar chart of how many times the 90s hard cap had to
     force a switch (ideally near zero for a well-trained policy -- a
     high count means the agent isn't learning to switch on its own).

Works for the trained DQN checkpoint AND for the two evaluate.py baselines
(random / fixed-cycle), so the same figure can be produced side-by-side
for the report.

Usage:
    python -m dqn.switch_timing_plot --checkpoint runs/dqn_run1/qnet_final.pt --out runs/dqn_run1/timing
    python -m dqn.switch_timing_plot --policy random --out runs/timing_random
    python -m dqn.switch_timing_plot --policy fixed_cycle --fixed_cycle_seconds 30 --out runs/timing_fixed
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# Project root = the folder that CONTAINS this "dqn" package. Building
# every default path from this makes the script runnable from the VS
# Code Play button with no arguments, regardless of the working
# directory VS Code happens to launch it from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.q_network import QNetwork, select_actions_epsilon_greedy  # noqa: E402
from dqn.evaluate import random_policy_fn, fixed_cycle_policy_fn  # noqa: E402

import traci  # noqa: E402

# ------------------------------------------------------------------ #
# EDIT THESE IF YOUR FILES LIVE SOMEWHERE ELSE
# ------------------------------------------------------------------ #
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", "dqn_run1", "qnet_final.pt")
DEFAULT_NET_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "runs", "dqn_run1", "timing")


def run_and_record(env, policy_fn, seed):
    """Runs one episode, returning per-junction phase-interval logs plus
    grid-wide queue/waiting time series, sampled at every decision step
    (so timing resolution == env.cfg.decision_interval, 5s by default)."""
    obs, _ = env.reset(seed=seed)
    tl_ids = env.tl_ids
    n = env.n_agents

    # one open interval per junction: [start_time, label] ; closed and
    # appended to `intervals[i]` whenever the label changes
    intervals = {i: [] for i in range(n)}
    open_start = [0.0] * n
    open_label = [None] * n

    times, queue_series, wait_series = [], [], []
    forced_counts = np.zeros(n, dtype=int)

    def label_of(tid):
        state = traci.trafficlight.getRedYellowGreenState(tid)
        return "yellow" if "y" in state.lower() else "green"

    for i, tid in enumerate(tl_ids):
        open_label[i] = label_of(tid)

    t = 0.0
    while True:
        actions = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(actions)
        t += env.cfg.decision_interval

        forced_counts += info["forced_switches"].astype(int)
        queue_series.append(float(obs["node_features"][:, 0:4].sum()) * env.QUEUE_NORM)
        wait_series.append(float(obs["node_features"][:, 4:8].sum()) * env.WAIT_NORM)
        times.append(t)

        for i, tid in enumerate(tl_ids):
            lbl = label_of(tid)
            if lbl != open_label[i]:
                intervals[i].append((open_start[i], t, open_label[i]))
                open_start[i] = t
                open_label[i] = lbl

        if terminated or truncated:
            break

    for i in range(n):
        if t > open_start[i]:  # skip a spurious zero-length trailing segment if a
            intervals[i].append((open_start[i], t, open_label[i]))  # switch landed exactly on the final recorded step

    return {
        "tl_ids": tl_ids,
        "intervals": intervals,
        "times": np.array(times),
        "queue_series": np.array(queue_series),
        "wait_series": np.array(wait_series),
        "forced_counts": forced_counts,
        "episode_seconds": t,
    }


def plot_gantt(record, out_path, min_green, max_green):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tl_ids = record["tl_ids"]
    n = len(tl_ids)
    fig, ax = plt.subplots(figsize=(16, 0.42 * n + 1.5))
    colors = {"green": "#3FA66B", "yellow": "#E0B23A"}

    for i, tid in enumerate(tl_ids):
        bars = [(s, e - s) for (s, e, lbl) in record["intervals"][i] if lbl == "green"]
        ybars = [(s, e - s) for (s, e, lbl) in record["intervals"][i] if lbl == "yellow"]
        ax.broken_barh(bars, (i - 0.4, 0.8), facecolors=colors["green"])
        ax.broken_barh(ybars, (i - 0.4, 0.8), facecolors=colors["yellow"])

    ax.set_yticks(range(n))
    ax.set_yticklabels(tl_ids, fontsize=8)
    ax.set_xlabel("Episode time (s)")
    ax.set_title("Signal phase timeline per junction (green vs. yellow)")
    ax.set_xlim(0, record["episode_seconds"])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["green"], label="Green"), Patch(color=colors["yellow"], label="Yellow")],
              loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_green_duration_histogram(record, out_path, min_green, max_green):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    durations = []
    for i in record["intervals"]:
        for (s, e, lbl) in record["intervals"][i]:
            if lbl == "green":
                durations.append(e - s)
    durations = np.array(durations)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(durations, bins=30, color="#0F6E56", edgecolor="white")
    ax.axvline(min_green, color="#993556", linestyle="--", label=f"min_green = {min_green:.0f}s")
    ax.axvline(max_green, color="#854F0B", linestyle="--", label=f"max_green (hard cap) = {max_green:.0f}s")
    ax.set_xlabel("Green-phase duration chosen (s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of green-phase durations across all junctions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return durations


def plot_congestion_timeseries(record, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(record["times"], record["queue_series"], color="#0F6E56")
    ax1.set_ylabel("Total queue length\n(vehicles, all junctions)")
    ax2.plot(record["times"], record["wait_series"], color="#993556")
    ax2.set_ylabel("Total waiting time\n(s, all junctions)")
    ax2.set_xlabel("Episode time (s)")
    fig.suptitle("Grid-wide congestion over the episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forced_switches(record, out_path, max_green):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(record["tl_ids"], record["forced_counts"], color="#854F0B")
    ax.set_ylabel(f"# forced switches (hit {max_green:.0f}s cap)")
    ax.set_xlabel("Junction")
    ax.set_title("Hard-cap (max_green) trigger count per junction")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["dqn", "random", "fixed_cycle"], default="dqn")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"used when --policy dqn; default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--fixed_cycle_seconds", type=float, default=30.0)
    parser.add_argument("--net_file", default=DEFAULT_NET_FILE)
    parser.add_argument("--route_file", default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DEFAULT_OUT)
    # parse_known_args (not parse_args) so the Play button (zero args)
    # never errors out.
    args, _unknown = parser.parse_known_args()

    if args.policy == "dqn" and not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at:\n  {args.checkpoint}\n"
              f"Edit DEFAULT_CHECKPOINT near the top of this file, or pass "
              f"--checkpoint <path> if you're running from a terminal, or "
              f"switch --policy to 'random' or 'fixed_cycle' which don't need one.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    cfg = DQNConfig()
    env_cfg = GridEnvConfig(
        net_file=args.net_file,
        route_file=args.route_file,
        episode_seconds=cfg.episode_seconds,
        decision_interval=cfg.decision_interval,
    )
    env = TrafficGridEnv(env_cfg)

    if args.policy == "dqn":
        assert args.checkpoint, "--checkpoint is required for --policy dqn"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
        q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        q_net.eval()
        rng = np.random.default_rng(args.seed)
        policy_fn = lambda obs: select_actions_epsilon_greedy(q_net, obs["node_features"], cfg.eval_epsilon, rng, device)
    elif args.policy == "random":
        rng = np.random.default_rng(args.seed)
        policy_fn = random_policy_fn(env.n_agents, rng)
    else:
        policy_fn = fixed_cycle_policy_fn(args.fixed_cycle_seconds, env.cfg.max_green)

    print(f"Running one episode under policy={args.policy} (seed={args.seed})...")
    record = run_and_record(env, policy_fn, args.seed)
    env.close()

    plot_gantt(record, os.path.join(args.out, "phase_timeline.png"), env_cfg.min_green, env_cfg.max_green)
    durations = plot_green_duration_histogram(
        record, os.path.join(args.out, "green_duration_histogram.png"), env_cfg.min_green, env_cfg.max_green
    )
    plot_congestion_timeseries(record, os.path.join(args.out, "congestion_timeseries.png"))
    plot_forced_switches(record, os.path.join(args.out, "forced_switches_per_junction.png"), env_cfg.max_green)

    print(f"\nGreen-phase duration stats (s): mean={durations.mean():.1f}, "
          f"median={np.median(durations):.1f}, min={durations.min():.1f}, max={durations.max():.1f}")
    print(f"NOTE: durations are logged at {env_cfg.decision_interval}s resolution (one sample per "
          f"decision step), so true phase boundaries that don't align with this sampling grid can "
          f"appear up to {env_cfg.decision_interval}s shorter/longer here than the actual enforced "
          f"value -- a logged duration below min_green ({env_cfg.min_green}s) reflects this sampling "
          f"quantization, not an environment violation of the min_green constraint.")
    print(f"Total forced (hard-cap) switches across all junctions: {int(record['forced_counts'].sum())}")
    print(f"Plots saved under: {args.out}/")


if __name__ == "__main__":
    main()