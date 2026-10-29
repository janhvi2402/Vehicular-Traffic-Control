"""
scripts/make_fixed_time_figures.py -- Play button, no arguments, a few minutes. Run it AFTER scripts/eval_fixed_time.py.
Shows BOTH fixed-time plans (90 s and 60 s) in every combined figure/table. It does NOT re-run run_eval and touches no trained model or
summary.json: for every run it writes a COPY, eval_<net>/summary_fixedtime.json, in which the old fixed_cycle / always_switch results are renamed
fixed_cycle_old / always_switch_old and fixed_time_90 / fixed_time_60 are added. Outputs:
  runs/<algo>_seed<N>/figures_fixedtime/baseline_comparison_<net>.png
  runs/aggregate/compare_algorithms_<net>_fixedtime.png
  runs/aggregate/topology_generalization_dqn_ppo_fixedtime_vs90.png and ..._vs60.png
  runs/aggregate/results_table_fixedtime_vs90.md/.csv and ..._vs60.md/.csv      (improvement against each plan)
and finally prints the objective-J table. Training curves / behaviour diagnostics do not depend on the baseline.
"""
import copy, json, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.eval_fixed_time import CYCLES, plan_label
# ------------------------------------------------------------------ #
ALGOS, SEEDS = ["dqn", "ppo"], [0, 1, 2]
NETS = ["4x4", "3x3"]
# ------------------------------------------------------------------ #
KEYS = [f"fixed_time_{c}" for c in CYCLES]
COLORS_FT = {"fixed_time_90": "#C98A00", "fixed_time_60": "#E6B422"}
src = os.path.join(ROOT, "runs", "realistic_fixed_time.json")
if not os.path.exists(src): sys.exit("Run scripts/sweep_fixed_time_split.py and scripts/eval_fixed_time.py first.")
field = json.load(open(src))
for n in NETS:
    for c in CYCLES:
        if plan_label(c) not in field.get(n, {}):
            sys.exit(f"SKIP: '{plan_label(c)}' missing for {n} in runs/realistic_fixed_time.json -- run scripts/eval_fixed_time.py.")


def plot_run(s, path, algo, net):
    nj = s["fixed_time_90"]["n_agents"]
    ent = [("random", "Random", "#993556")]
    for k in KEYS:
        ent.append((k, f"Fixed-time {s[k]['cycle_seconds']} s\n(main {s[k]['split_main_green']} / side {s[k]['split_side_green']} s)", COLORS_FT[k]))
    ent += [("max_pressure", "Max-pressure", "#7B4FA0"), ("algo", algo.upper(), "#0F6E56")]
    ent = [e for e in ent if e[0] in s]
    mets = [("mean_waiting_time", "Mean waiting time per step\n(grid-wide sum)"), ("mean_vehicle_wait", "Mean wait per vehicle (s)\n(completed trips)"),
            ("throughput", "Throughput (vehicles arrived)"), ("mean_queue_length", "Mean queue length (vehicles)"),
            ("phase_changes", "Phase changes per junction\nper episode")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, (m, title) in zip(axes, mets):
        if m == "phase_changes":
            mu = [(s[k]["n_switch_votes"]["mean"] + s[k]["n_forced_switches"]["mean"]) / nj for k, _, _ in ent]; sd = [s[k]["n_switch_votes"]["std"] / nj for k, _, _ in ent]
        else:
            mu = [s[k][m]["mean"] for k, _, _ in ent]; sd = [s[k][m]["std"] for k, _, _ in ent]
        ax.bar([lab for _, lab, _ in ent], mu, yerr=sd, capsize=4, color=[c for _, _, c in ent]); ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle(f"{algo.upper()} vs baselines on the {net} grid (mean ± std over evaluation episodes)", fontsize=12); fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig); print("saved", path)


made = 0
for a in ALGOS:
    for sd in SEEDS:
        for n in NETS:
            d = os.path.join(ROOT, "runs", f"{a}_seed{sd}", f"eval_{n}"); p = os.path.join(d, "summary.json")
            if not os.path.exists(p): print("missing", p); continue
            s = json.load(open(p))
            s["fixed_cycle_old"] = s.pop("fixed_cycle")
            if "always_switch" in s: s["always_switch_old"] = s.pop("always_switch")
            for c in CYCLES: s[f"fixed_time_{c}"] = copy.deepcopy(field[n][plan_label(c)])
            json.dump(s, open(os.path.join(d, "summary_fixedtime.json"), "w"), indent=2)
            plot_run(s, os.path.join(ROOT, "runs", f"{a}_seed{sd}", "figures_fixedtime", f"baseline_comparison_{n}.png"), a, n); made += 1
print(f"{made} run summaries written (summary_fixedtime.json)")

results = []
steps = [(f"compare algorithms {n}", {}, ["diagnostics/compare_algorithms.py", "--algos", *ALGOS, "--seeds", *map(str, SEEDS),
          "--summary_name", f"eval_{n}/summary_fixedtime.json", "--out", os.path.join("runs", "aggregate", f"compare_algorithms_{n}_fixedtime.png")]) for n in NETS]
for c in CYCLES:
    e = {"SUMMARY_NAME": "summary_fixedtime.json", "BASELINE_KEY": f"fixed_time_{c}", "OUT_SUFFIX": f"_fixedtime_vs{c}"}
    steps += [(f"topology generalization vs {c} s", e, ["scripts/make_topology_plot.py"]), (f"results table vs {c} s", e, ["scripts/make_results_table.py"])]
steps.append(("objective J table", {}, ["scripts/compute_objective_J.py"]))
for label, extra, args in steps:
    print(f"\n=== {label}", flush=True); results.append((label, subprocess.run([sys.executable, *args], cwd=ROOT, env=dict(os.environ, **extra)).returncode))
print("\n" + "=" * 50)
for label, rc in results: print(("ok   " if rc == 0 else "FAIL ") + label)
print("Outputs: runs/<algo>_seed<N>/figures_fixedtime/, runs/aggregate/compare_algorithms_<net>_fixedtime.png, "
      "topology_generalization_dqn_ppo_fixedtime_vs90.png / _vs60.png, results_table_fixedtime_vs90.md/.csv / _vs60.md/.csv")
