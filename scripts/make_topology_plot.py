"""
scripts/make_topology_plot.py  (replaces dqn/make_topology_plot.py -- delete that one)
Trained 4x4 grid vs the unseen 3x3 grid, for ONE algorithm or several, from the evaluate.py summaries written by
scripts/run_eval.py (runs/<algo>_seed<N>/eval_4x4 and eval_3x3). Press the Play button; edit ALGOS / SEEDS.

Absolute numbers are NOT comparable between the grids (9 vs 16 junctions, different load), so every panel shows the
result RELATIVE to the fixed-cycle baseline of the SAME network, or per junction.
Bars = mean over training seeds; error bars = std over training seeds (drawn only for >= 2 seeds).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------ #
ALGOS = ["dqn", "ppo"]        # EDIT: one algorithm ("ppo") gives the single-algorithm figure
SEEDS = [0, 1, 2]                   # EDIT: training seeds that were trained AND evaluated, e.g. [0, 1, 2]
NETS = [("4x4", "4x4 grid\n(trained on)", 16), ("3x3", "3x3 grid\n(unseen topology)", 9)]   # (tag, label, n_junctions)
OUT_DIR = os.path.join(ROOT, "runs", "aggregate")
# ------------------------------------------------------------------ #
COLORS = {"dqn": "#0F6E56", "ppo": "#2B5FA8", "ac": "#B5651D", "ca": "#7B3FA0", "max_pressure": "#7A7A7A", "fixed_cycle": "#854F0B"}
LABELS = {"dqn": "DQN", "ppo": "PPO", "ac": "AC", "ca": "CA", "max_pressure": "Max-pressure", "fixed_cycle": "Tuned fixed-time"}
LABELS.update({"fixed_time_90": "Fixed-time 90 s", "fixed_time_60": "Fixed-time 60 s"})
COLORS.update({"fixed_time_90": "#C98A00", "fixed_time_60": "#E6B422"})
# Set by scripts/make_fixed_time_figures.py: improvement against one fixed-time plan (BASELINE_KEY), read summary_fixedtime.json
SUMMARY_NAME = os.environ.get("SUMMARY_NAME", "summary.json")
BASELINE_KEY = os.environ.get("BASELINE_KEY", "fixed_cycle")
SUFFIX = os.environ.get("OUT_SUFFIX") or ("" if SUMMARY_NAME == "summary.json" else "_" + SUMMARY_NAME[len("summary_"):-len(".json")])   # summary_fixedtime.json -> _fixedtime
METRICS = [("mean_waiting_time", "Mean waiting time"), ("mean_vehicle_wait", "Mean wait per vehicle"), ("mean_queue_length", "Mean queue length")]


def load(algo):
    out = {}
    for net, _, _ in NETS:
        out[net] = []
        for s in SEEDS:
            p = os.path.join(ROOT, "runs", f"{algo}_seed{s}", f"eval_{net}", SUMMARY_NAME)
            if os.path.exists(p): out[net].append(json.load(open(p)))
    return out


data = {}
for a in ALGOS:
    d = load(a)
    if all(len(d[n]) == len(SEEDS) for n, _, _ in NETS): data[a] = d
    else: print(f"SKIP {a}: eval summaries missing for some (network, seed) -- run scripts/run_eval.py")
if not data:
    raise FileNotFoundError("No algorithm has complete evaluate.py summaries for every network/seed in SEEDS.")
algos = list(data)
ref = data[algos[0]]                                  # baselines (fixed-cycle, max-pressure) are identical across algorithms
has_mp = all("max_pressure" in r for n, _, _ in NETS for r in ref[n])


def improvement(r, key, m):
    fx = r[BASELINE_KEY][m]["mean"]
    return 100.0 * (fx - r[key][m]["mean"]) / fx


def per_junction(r, key, nj):
    return (r[key]["n_switch_votes"]["mean"] + r[key]["n_forced_switches"]["mean"]) / nj


def bars(ax, entries, value_fn, x):
    """entries: [(key, label, color, runs_by_net)] ; value_fn(run, net_index) -> float"""
    w = 0.8 / len(entries)
    for k, (key, lab, col, runs) in enumerate(entries):
        vals = [[value_fn(r, key, i) for r in runs[net]] for i, (net, _, _) in enumerate(NETS)]
        mu = [np.mean(v) for v in vals]; sd = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals]
        ax.bar(x + (k - (len(entries) - 1) / 2) * w, mu, w, yerr=sd if len(SEEDS) > 1 else None, capsize=4, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab, _ in NETS]); ax.legend(fontsize=8)


x = np.arange(len(NETS))
fig, axes = plt.subplots(1, 4, figsize=(21, 5.2))
imp_entries = [(a, LABELS[a], COLORS[a], data[a]) for a in algos]
if has_mp: imp_entries.append(("max_pressure", LABELS["max_pressure"], COLORS["max_pressure"], ref))
print(f"{'metric':26s}{'network':>8s}" + "".join(f"{LABELS[k]:>15s}" for k, *_ in imp_entries))
for ax, (m, title) in zip(axes[:3], METRICS):
    bars(ax, imp_entries, lambda r, key, i, m=m: improvement(r, "algo" if key in data else key, m), x)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{title}: improvement over {LABELS[BASELINE_KEY]} (%)\n(higher = better; 0 = same as {LABELS[BASELINE_KEY]})", fontsize=10)
    for net, _, _ in NETS:
        print(f"{m:26s}{net:>8s}" + "".join(f"{np.mean([improvement(r, 'algo' if k in data else k, m) for r in runs[net]]):>14.1f}%" for k, _, _, runs in imp_entries))
ent = [(a, LABELS[a], COLORS[a], data[a]) for a in algos]
ft_keys = [k for k in ("fixed_time_90", "fixed_time_60") if all(k in r for n, _, _ in NETS for r in ref[n])]
ent += [(k, LABELS[k], COLORS[k], ref) for k in ft_keys]
if not ft_keys: ent.append(("fixed_cycle", LABELS["fixed_cycle"], COLORS["fixed_cycle"], ref))   # old summary.json only
if has_mp: ent.insert(len(algos), ("max_pressure", LABELS["max_pressure"], COLORS["max_pressure"], ref))
bars(axes[3], ent, lambda r, key, i: per_junction(r, "algo" if key in data else key, NETS[i][2]), x)
axes[3].set_title("Phase changes per junction per episode\n(voted + forced)", fontsize=10)

base_txt = "; ".join(f"{n}: " + ", ".join(f"{k[-2:]} s plan (main {ref[n][0][k]['split_main_green']}/side {ref[n][0][k]['split_side_green']})" for k in ft_keys)
                     for n, _, _ in NETS) if ft_keys else "tuned fixed-time cycle"
fig.suptitle(f"{', '.join(LABELS[a] for a in algos)}: trained 4x4 grid vs unseen 3x3 grid (train seeds {SEEDS})", fontsize=12)
fig.text(0.005, 0.005, "NOTE: 3x3 differs from 4x4 in topology AND load (about 1.8x vehicles per junction: 4,067 vehicles on 9 junctions vs 4,027 on 16). "
         f"Baselines: {base_txt}. Evaluation episodes differ only by SUMO seed.", fontsize=7, color="0.3")
fig.tight_layout(rect=(0, 0.03, 1, 0.93))
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"topology_generalization_{'_'.join(algos)}{SUFFIX}.png")
fig.savefig(out, dpi=150); plt.close(fig)
print(f"Saved: {out}")
