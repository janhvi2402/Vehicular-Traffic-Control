"""
make_dqn_ppo_comparison.py
===========================
Regenerates dqn_ppo_comparison.png: DQN vs PPO vs fixed-cycle, on both the
trained 4x4 grid and the unseen 3x3 grid, from the four evaluate.py runs
produced by run_matched_eval.sh.

No command-line args -- edit the four summary.json paths below, then run:
    python make_dqn_ppo_comparison.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# EDIT THESE FOUR PATHS
# ------------------------------------------------------------------ #
DQN_4X4 = "runs/dqn_run_25k/eval_4x4/summary.json"
DQN_3X3 = "runs/dqn_run_25k/eval_3x3/summary.json"
PPO_4X4 = "runs/ppo_run1/eval_4x4/summary.json"
PPO_3X3 = "runs/ppo_run1/eval_3x3/summary.json"
OUT_PATH = "runs/dqn_ppo_comparison.png"
TITLE_SUFFIX = "(both algorithms trained for a matched ~25,000-env-step validation budget)"
# ------------------------------------------------------------------ #

with open(DQN_4X4) as f: d4 = json.load(f)
with open(DQN_3X3) as f: d3 = json.load(f)
with open(PPO_4X4) as f: p4 = json.load(f)
with open(PPO_3X3) as f: p3 = json.load(f)

# dqn/evaluate.py's summary.json uses the "dqn" key; ppo/evaluate.py's uses "ppo".
# Both use "fixed_cycle" -- but pull it separately per algorithm's own run
# since each run's --fixed_cycle_seconds sweep/override could differ if you
# didn't pin them to the same value (run_matched_eval.sh pins both to the
# same CYCLE, so d4["fixed_cycle"] and p4["fixed_cycle"] should already match
# here; this script doesn't assume that, it just reads each independently).

def series(metric):
    return {
        "dqn":   [d4["dqn"][metric]["mean"],   d3["dqn"][metric]["mean"]],
        "ppo":   [p4["ppo"][metric]["mean"],   p3["ppo"][metric]["mean"]],
        "fixed": [d4["fixed_cycle"][metric]["mean"], d3["fixed_cycle"][metric]["mean"]],
    }

wait = series("mean_waiting_time")
queue = series("mean_queue_length")
switches = series("n_switch_votes")

labels = ["4x4 grid\n(16 junctions)", "3x3 grid\n(9 junctions, unseen)"]
x = range(len(labels))
width = 0.27
colors = {"dqn": "#0F6E56", "ppo": "#3B5EA8", "fixed": "#854F0B"}
names = {"dqn": "DQN", "ppo": "PPO", "fixed": "Fixed-cycle"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, data, title in [
    (axes[0], wait, "Mean waiting time (s)"),
    (axes[1], queue, "Mean queue length (vehicles)"),
    (axes[2], switches, "Total switch votes per episode"),
]:
    for i, key in enumerate(["dqn", "ppo", "fixed"]):
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], data[key], width, label=names[key], color=colors[key])
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()

fig.suptitle(f"DQN vs. PPO vs. fixed-cycle baseline, on the trained 4x4 grid and the unseen 3x3 grid\n{TITLE_SUFFIX}")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
