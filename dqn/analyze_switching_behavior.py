"""
dqn/analyze_switching_behavior.py
==================================
Answers a specific, important question about a trained checkpoint: is the
learned policy actually reading queue/waiting-time state to decide when to
switch, or has it degenerated into a trivial rule (e.g. "always switch the
instant min_green allows" or "always ride the max_green cap")?

Method: run one GREEDY (epsilon=0) rollout, and at every decision point
where a junction is in a stable green phase and legally eligible to switch,
record the traffic state split by which approaches are currently being
served (green) vs. waiting (red) -- using this network's known phase
convention (phase 0/1 = E-W green, phase 2/3 = N-S green, verified via
traci.trafficlight.getAllProgramLogics on this net file; re-verify if you
change the network) -- alongside the action the network actually chose.

Reports:
  - switch rate broken down by elapsed time (tests the "just a timer" theory)
  - switch rate broken down by congestion imbalance at FIXED elapsed time
    (isolates the effect of traffic state from the effect of the clock)
  - correlations between the switch decision and each state variable
  - mean state values for held vs. switched decisions
  - a two-panel comparison figure

Usage:
    python -m dqn.analyze_switching_behavior --checkpoint runs/dqn_run1/qnet_final.pt --out runs/dqn_run1/behavior
"""

from __future__ import annotations

import argparse
import json
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
from dqn.q_network import QNetwork  # noqa: E402

# ------------------------------------------------------------------ #
# EDIT THESE IF YOUR FILES LIVE SOMEWHERE ELSE
# ------------------------------------------------------------------ #
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", "dqn_run1", "qnet_final.pt")
DEFAULT_NET_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "runs", "dqn_run1", "behavior")

APPROACH_ORDER = ["N", "E", "S", "W"]  # must match TrafficGridEnv.APPROACH_ORDER


def collect_rows(q_net, env, env_cfg, seed):
    """Runs one greedy (epsilon=0) episode, returning one row per junction
    per eligible stable-green decision point."""
    obs, _ = env.reset(seed=seed)
    rows = []
    with torch.no_grad():
        while True:
            feats = obs["node_features"]
            x = torch.as_tensor(feats, dtype=torch.float32)
            q_values = q_net(x).numpy()
            actions = q_values.argmax(axis=1)  # greedy, no exploration noise

            for i in range(env.n_agents):
                phase_norm = float(feats[i, 8])
                elapsed = float(feats[i, 10] * env_cfg.max_green)
                is_yellow = feats[i, 9] > 0.5
                mask = obs["action_mask"][i]
                queue = feats[i, 0:4] * env.QUEUE_NORM
                wait = feats[i, 4:8] * env.WAIT_NORM

                # phase_norm in {0, 1/3, 2/3, 1} for a 4-phase program
                phase_idx = round(phase_norm * 3)
                if is_yellow or mask <= 0.5:
                    continue  # only analyze eligible, stable-green decisions
                served = ["E", "W"] if phase_idx == 0 else ["N", "S"] if phase_idx == 2 else None
                if served is None:
                    continue

                served_idx = [APPROACH_ORDER.index(a) for a in served]
                waiting_idx = [j for j in range(4) if j not in served_idx]
                rows.append({
                    "elapsed": elapsed,
                    "served_queue": float(queue[served_idx].sum()),
                    "waiting_queue": float(queue[waiting_idx].sum()),
                    "served_wait": float(wait[served_idx].sum()),
                    "waiting_wait": float(wait[waiting_idx].sum()),
                    "action": int(actions[i]),
                })

            obs, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break
    return rows


def analyze(rows):
    elapsed = np.array([r["elapsed"] for r in rows])
    served_q = np.array([r["served_queue"] for r in rows])
    waiting_q = np.array([r["waiting_queue"] for r in rows])
    served_w = np.array([r["served_wait"] for r in rows])
    waiting_w = np.array([r["waiting_wait"] for r in rows])
    action = np.array([r["action"] for r in rows])
    imbalance = waiting_w - served_w

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    report = {
        "n_decision_points": len(rows),
        "overall_switch_rate": float(action.mean()),
        "correlations": {
            "action_vs_elapsed": corr(action, elapsed),
            "action_vs_served_queue": corr(action, served_q),
            "action_vs_waiting_queue": corr(action, waiting_q),
            "action_vs_served_wait": corr(action, served_w),
            "action_vs_waiting_wait": corr(action, waiting_w),
            "action_vs_imbalance": corr(action, imbalance),
        },
        "switch_rate_by_elapsed_bin": {},
        "mean_state_hold_vs_switch": {},
    }

    bins = [(10, 15), (15, 20), (20, 30), (30, 50), (50, 90)]
    for lo, hi in bins:
        m = (elapsed >= lo) & (elapsed < hi)
        if m.sum() > 0:
            report["switch_rate_by_elapsed_bin"][f"{lo}-{hi}s"] = {
                "n": int(m.sum()), "switch_rate": float(action[m].mean())
            }

    for name, arr in [("elapsed", elapsed), ("served_queue", served_q), ("waiting_queue", waiting_q),
                       ("served_wait", served_w), ("waiting_wait", waiting_w), ("imbalance", imbalance)]:
        report["mean_state_hold_vs_switch"][name] = {
            "hold_mean": float(arr[action == 0].mean()) if (action == 0).any() else None,
            "switch_mean": float(arr[action == 1].mean()) if (action == 1).any() else None,
        }

    return report, elapsed, imbalance, action


def plot_behavior(elapsed, imbalance, action, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = [(10, 15), (15, 20), (20, 30), (30, 50), (50, 90)]
    labels = [f"{lo}-{hi}s" for lo, hi in bins]
    rates = []
    for lo, hi in bins:
        m = (elapsed >= lo) & (elapsed < hi)
        rates.append(action[m].mean() if m.sum() > 0 else 0)
    axes[0].bar(labels, rates, color="#0F6E56")
    axes[0].set_title("Switch rate vs. elapsed green time")
    axes[0].set_xlabel("Elapsed time in phase")
    axes[0].set_ylabel("P(switch | eligible)")
    axes[0].set_ylim(0, 1)

    m10 = (elapsed >= 10) & (elapsed < 15)
    imb_here, act_here = imbalance[m10], action[m10]
    q1, q3 = np.percentile(imb_here, [33, 67])
    groups = [("Low\n(waiting<<served)", imb_here < q1), ("Mid", (imb_here >= q1) & (imb_here < q3)),
              ("High\n(waiting>>served)", imb_here >= q3)]
    labels2 = [g[0] for g in groups]
    rates2 = [act_here[g[1]].mean() if g[1].sum() > 0 else 0 for g in groups]
    axes[1].bar(labels2, rates2, color="#993556")
    axes[1].set_title("Switch rate vs. congestion imbalance\n(at fixed elapsed = 10-15s)")
    axes[1].set_xlabel("Waiting-side minus served-side wait time")
    axes[1].set_ylabel("P(switch | eligible)")
    axes[1].set_ylim(0, 1)

    fig.suptitle("Is switching congestion-driven or timer-driven?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help=f"default: {DEFAULT_CHECKPOINT}")
    parser.add_argument("--net_file", default=DEFAULT_NET_FILE)
    parser.add_argument("--route_file", default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=DEFAULT_OUT)
    # parse_known_args (not parse_args) so the Play button (zero args)
    # never errors out.
    args, _unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at:\n  {args.checkpoint}\n"
              f"Edit DEFAULT_CHECKPOINT near the top of this file, or pass "
              f"--checkpoint <path> if you're running from a terminal.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    cfg = DQNConfig()
    q_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions)
    q_net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    q_net.eval()

    env_cfg = GridEnvConfig(
        net_file=args.net_file, route_file=args.route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
    )
    env = TrafficGridEnv(env_cfg)

    print(f"Running one greedy episode (seed={args.seed}) to log state+action pairs...")
    rows = collect_rows(q_net, env, env_cfg, args.seed)
    env.close()
    print(f"Collected {len(rows)} eligible decision points.")

    report, elapsed, imbalance, action = analyze(rows)

    with open(os.path.join(args.out, "behavior_rows.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(args.out, "behavior_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_behavior(elapsed, imbalance, action, os.path.join(args.out, "switching_behavior_analysis.png"))

    print("\n=== Switch rate by elapsed-time bin ===")
    for k, v in report["switch_rate_by_elapsed_bin"].items():
        print(f"  {k:>8s}: n={v['n']:5d}  switch_rate={v['switch_rate']:.3f}")
    print("\n=== Correlations with the switch decision ===")
    for k, v in report["correlations"].items():
        print(f"  {k:30s} = {v:+.3f}")
    print(f"\nSaved: {args.out}/behavior_rows.json, behavior_report.json, switching_behavior_analysis.png")


if __name__ == "__main__":
    main()