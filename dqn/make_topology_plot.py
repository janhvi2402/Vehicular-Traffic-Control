"""
make_topology_plot.py
======================
Builds the "same frozen DQN checkpoint: trained 4x4 grid vs. an unseen 3x3
grid topology" comparison plot from two dqn/evaluate.py runs.

No command-line args -- everything is hardcoded below. Just edit the four
paths to match your own folders, then run:

    python make_topology_plot.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root = the folder that CONTAINS this "dqn" package. Basing the
# four paths below on this (instead of leaving them as bare relative
# strings) means clicking VS Code's Play button works no matter what
# directory VS Code happens to set as the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ #
# EDIT THESE FOUR LINES
# ------------------------------------------------------------------ #
# FIX: was runs/dqn_run2 -- a different, unrelated checkpoint from the
# runs/dqn_run_matched500 checkpoint used everywhere else in this project's
# DQN-vs-PPO comparison (evaluate.py, run_matched_eval.sh,
# make_dqn_ppo_comparison.py). Pointing this figure at a different model
# than every other figure describes means the topology-generalization
# story and the algorithm-comparison story would silently be about two
# different trained policies.
SUMMARY_4X4 = os.path.join(PROJECT_ROOT, "runs", "dqn_run_matched500", "eval_4x4", "summary.json")
SUMMARY_3X3 = os.path.join(PROJECT_ROOT, "runs", "dqn_run_matched500", "eval_3x3", "summary.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "runs", "dqn_run_matched500")
OUT_FILENAME = "topology_generalization.png"
# ------------------------------------------------------------------ #

os.makedirs(OUT_DIR, exist_ok=True)

for path in (SUMMARY_4X4, SUMMARY_3X3):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected an evaluate.py summary.json at:\n  {path}\n"
            f"Run dqn/evaluate.py first (once with the 4x4 net, once with the "
            f"3x3 net, --out pointed at eval_4x4 / eval_3x3 respectively), or "
            f"edit SUMMARY_4X4 / SUMMARY_3X3 near the top of this file."
        )

with open(SUMMARY_4X4) as f:
    s4 = json.load(f)
with open(SUMMARY_3X3) as f:
    s3 = json.load(f)

# --- pull the three metrics for DQN and fixed-cycle on both networks ---
dqn_wait   = [s4["dqn"]["mean_waiting_time"]["mean"],  s3["dqn"]["mean_waiting_time"]["mean"]]
fixed_wait = [s4["fixed_cycle"]["mean_waiting_time"]["mean"], s3["fixed_cycle"]["mean_waiting_time"]["mean"]]

dqn_queue   = [s4["dqn"]["mean_queue_length"]["mean"],  s3["dqn"]["mean_queue_length"]["mean"]]
fixed_queue = [s4["fixed_cycle"]["mean_queue_length"]["mean"], s3["fixed_cycle"]["mean_queue_length"]["mean"]]

dqn_switches   = [s4["dqn"]["n_switch_votes"]["mean"],  s3["dqn"]["n_switch_votes"]["mean"]]
fixed_switches = [s4["fixed_cycle"]["n_switch_votes"]["mean"], s3["fixed_cycle"]["n_switch_votes"]["mean"]]

labels = ["4x4 grid\n(16 junctions, trained on)", "3x3 grid\n(9 junctions, never seen)"]
x = range(len(labels))
width = 0.35
dqn_color, fixed_color = "#0F6E56", "#854F0B"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.bar([i - width/2 for i in x], dqn_wait, width, label="DQN", color=dqn_color)
ax.bar([i + width/2 for i in x], fixed_wait, width, label="Fixed-cycle", color=fixed_color)
ax.set_title("Mean waiting time (s)\n(lower = better)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.legend()

ax = axes[1]
ax.bar([i - width/2 for i in x], dqn_queue, width, label="DQN", color=dqn_color)
ax.bar([i + width/2 for i in x], fixed_queue, width, label="Fixed-cycle", color=fixed_color)
ax.set_title("Mean queue length (vehicles)\n(lower = better)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.legend()

ax = axes[2]
ax.bar([i - width/2 for i in x], dqn_switches, width, label="DQN", color=dqn_color)
ax.bar([i + width/2 for i in x], fixed_switches, width, label="Fixed-cycle", color=fixed_color)
ax.set_title("Total switch count per episode\n(more switches = more reward penalty)")
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.legend()

fig.suptitle("Same frozen DQN checkpoint: trained 4x4 grid vs. an unseen 3x3 grid topology")
fig.tight_layout()

out_path = os.path.join(OUT_DIR, OUT_FILENAME)
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")