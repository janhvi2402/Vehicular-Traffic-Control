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
DQN_4X4 = "runs/dqn_run_matched500/eval_4x4/summary.json"
DQN_3X3 = "runs/dqn_run_matched500/eval_3x3/summary.json"
PPO_4X4 = "runs/ppo_run_matched500/eval_4x4/summary.json"
PPO_3X3 = "runs/ppo_run_matched500/eval_3x3/summary.json"
OUT_PATH = "runs/dqn_ppo_comparison.png"
TITLE_SUFFIX = "(both algorithms trained for a matched 500-episode / 360,000-env-step budget)"
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
#
# SANITY CHECK: if the fixed-cycle runs used the same --fixed_cycle_seconds,
# the same eval seeds, and the same net/route files (all true when driven by
# run_matched_eval.sh), then DQN's and PPO's OWN fixed_cycle baseline numbers
# describe the exact same policy on the exact same episodes -- they should
# come out numerically identical, since the fixed-cycle controller doesn't
# know or care which learned algorithm it's being compared against. If they
# don't match, something in the setup (net/route file, seeds, decision
# granularity, environment version) silently differs between the two runs,
# and NOTHING else this script computes can be trusted as a fair comparison
# until that's resolved.
_FIXED_CYCLE_TOLERANCE = 1e-6
for metric in ("mean_waiting_time", "mean_vehicle_wait", "mean_queue_length", "n_switch_votes"):
    for label, a, b in [("4x4", d4, p4), ("3x3", d3, p3)]:
        va, vb = a["fixed_cycle"][metric]["mean"], b["fixed_cycle"][metric]["mean"]
        if abs(va - vb) > _FIXED_CYCLE_TOLERANCE:
            print(f"WARNING: fixed_cycle[{metric}] mismatch on {label} between the DQN run "
                  f"({va}) and the PPO run ({vb}) -- these should be identical if both runs "
                  f"truly share the same env/seeds/fixed_cycle_seconds. Something in the two "
                  f"setups differs; treat this comparison as unverified until you find it.")

def series(metric):
    return {
        "dqn":   [d4["dqn"][metric]["mean"],   d3["dqn"][metric]["mean"]],
        "ppo":   [p4["ppo"][metric]["mean"],   p3["ppo"][metric]["mean"]],
        "fixed": [d4["fixed_cycle"][metric]["mean"], d3["fixed_cycle"][metric]["mean"]],
    }

wait = series("mean_waiting_time")
veh_wait = series("mean_vehicle_wait")
queue = series("mean_queue_length")
switches = series("n_switch_votes")

labels = ["4x4 grid\n(16 junctions)", "3x3 grid\n(9 junctions, unseen)"]
x = range(len(labels))
width = 0.27
colors = {"dqn": "#0F6E56", "ppo": "#3B5EA8", "fixed": "#854F0B"}
names = {"dqn": "DQN", "ppo": "PPO", "fixed": "Fixed-cycle"}

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

for ax, data, title in [
    (axes[0], wait, "Mean waiting time (s)\n(grid-wide, per-step sum)"),
    (axes[1], veh_wait, "Mean wait per vehicle (s)\n(per-vehicle, whole trip)"),
    (axes[2], queue, "Mean queue length (vehicles)"),
    (axes[3], switches, "Total switch votes per episode"),
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