"""
dqn/make_topology_plot.py   (keep this file inside dqn/ or scripts/: it finds the project root two levels up)
=========================
"Same frozen DQN checkpoint: trained 4x4 grid vs. an unseen 3x3 grid" -- from the evaluate.py summaries written by
scripts/run_eval.py (runs/dqn_seed<N>/eval_4x4 and eval_3x3). Press the Play button; edit SEEDS below.

Absolute numbers are NOT comparable between the two grids (9 vs 16 junctions, different load), so every panel shows
the result RELATIVE to the fixed-cycle baseline of the SAME network, or per junction.
Bars = mean over the training seeds in SEEDS; error bars = std over training seeds (only drawn for >= 2 seeds).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------ #
SEEDS = [0]                                   # EDIT: training seeds that have been trained AND evaluated, e.g. [0, 1, 2]
NETS = [("4x4", "4x4 grid\n(trained on)", 16), ("3x3", "3x3 grid\n(unseen topology)", 9)]   # (folder tag, label, n_junctions)
OUT_DIR = os.path.join(PROJECT_ROOT, "runs", "aggregate")
OUT_FILENAME = "topology_generalization.png"
# ------------------------------------------------------------------ #

missing, data = [], {}
for net, _, _ in NETS:
    data[net] = []
    for s in SEEDS:
        p = os.path.join(PROJECT_ROOT, "runs", f"dqn_seed{s}", f"eval_{net}", "summary.json")
        if os.path.exists(p):
            data[net].append(json.load(open(p)))
        else:
            missing.append(p)
if missing:
    raise FileNotFoundError("Missing evaluate.py summaries (run scripts/run_eval.py first):\n  " + "\n  ".join(missing))

POLICIES = [("dqn", "DQN", "#0F6E56"), ("max_pressure", "Max-pressure", "#2B5FA8")]
POLICIES = [p for p in POLICIES if all(p[0] in r for net in data for r in data[net])]
METRICS = [("mean_waiting_time", "Mean waiting time"), ("mean_vehicle_wait", "Mean wait per vehicle"),
           ("mean_queue_length", "Mean queue length")]


def improvement(r, pol, m):
    fx = r["fixed_cycle"][m]["mean"]
    return 100.0 * (fx - r[pol][m]["mean"]) / fx


def changes_per_junction(r, pol, n_j):
    return (r[pol]["n_switch_votes"]["mean"] + r[pol]["n_forced_switches"]["mean"]) / n_j


fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
x = np.arange(len(NETS)); n_pol = len(POLICIES); width = 0.8 / max(n_pol, 1)
print(f"{'metric':28s}{'network':>8s}" + "".join(f"{lab:>16s}" for _, lab, _ in POLICIES))
for ax, (m, title) in zip(axes[:3], METRICS):
    for k, (pol, lab, col) in enumerate(POLICIES):
        vals = [[improvement(r, pol, m) for r in data[net]] for net, _, _ in NETS]
        mu = [np.mean(v) for v in vals]; sd = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals]
        ax.bar(x + (k - (n_pol - 1) / 2) * width, mu, width, yerr=sd if len(SEEDS) > 1 else None, capsize=4, label=lab, color=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in NETS])
    ax.set_title(f"{title}: improvement over fixed-cycle (%)\n(higher = better; 0 = same as fixed-cycle)")
    ax.legend()
    for net, _, _ in NETS:
        print(f"{m:28s}{net:>8s}" + "".join(f"{np.mean([improvement(r, pol, m) for r in data[net]]):>15.1f}%" for pol, _, _ in POLICIES))

ax = axes[3]
ALL = [("dqn", "DQN", "#0F6E56"), ("max_pressure", "Max-pressure", "#2B5FA8"), ("fixed_cycle", "Fixed-cycle", "#854F0B")]
ALL = [p for p in ALL if all(p[0] in r for net in data for r in data[net])]
w3 = 0.8 / max(len(ALL), 1)
for k, (pol, lab, col) in enumerate(ALL):
    vals = [[changes_per_junction(r, pol, nj) for r in data[net]] for net, _, nj in NETS]
    mu = [np.mean(v) for v in vals]; sd = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals]
    ax.bar(x + (k - (len(ALL) - 1) / 2) * w3, mu, w3, yerr=sd if len(SEEDS) > 1 else None, capsize=4, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in NETS])
ax.set_title("Phase changes per junction per episode\n(voted + forced)")
ax.legend()

fixed_used = {net: sorted({str(r.get("fixed_cycle_seconds_used")) for r in data[net]}) for net, _, _ in NETS}
fig.suptitle(f"Same frozen DQN checkpoint(s) (train seeds {SEEDS}): trained 4x4 grid vs. unseen 3x3 grid", fontsize=12)
fig.text(0.005, 0.005, "NOTE: the 3x3 test changes topology AND load (see routes files); fixed-cycle baseline tuned per network "
         f"(cycle used: {fixed_used}); evaluation episodes differ only by SUMO seed (same demand file).", fontsize=7, color="0.3")
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, OUT_FILENAME)
fig.savefig(out_path, dpi=150); plt.close(fig)
print(f"Saved: {out_path}")