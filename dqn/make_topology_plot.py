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

# ------------------------------------------------------------------ #
# EDIT THESE FOUR LINES
# ------------------------------------------------------------------ #
SUMMARY_4X4 = "runs/dqn_run2/eval_4x4/summary.json"
SUMMARY_3X3 = "runs/dqn_run2/eval_3x3/summary.json"
OUT_DIR = "runs/dqn_run2"
OUT_FILENAME = "topology_generalization.png"
# ------------------------------------------------------------------ #

os.makedirs(OUT_DIR, exist_ok=True)

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
