"""
scripts/make_field_baseline_figures.py -- Play button, no arguments, a few minutes.
Run it AFTER scripts/eval_field_fixed_time.py. It does NOT re-run run_eval and NO trained model is touched:
the DQN/PPO/random/max-pressure/tuned-fixed results already in runs/ stay valid, because the baseline does not depend on the agents.
Steps:
  1. adds the field plan to a COPY of every run's summary (eval_<net>/summary_field.json; summary.json is left unchanged);
  2. per-run bar charts with the field plan:        runs/<algo>_seed<N>/figures_field/baseline_comparison_<net>.png
  3. all algorithms x seeds vs baselines:           runs/aggregate/compare_algorithms_<net>_field.png
  4. 4x4 vs 3x3 generalization (improvement over the field plan): runs/aggregate/topology_generalization_<algos>_field.png
  5. numbers table:                                 runs/aggregate/results_table_field.md / .csv
  6. objective J table (printed).
Training curves and behaviour diagnostics do not depend on the baseline, so the earlier ones stay as they are.
"""
import copy, json, os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------ #
ALGOS, SEEDS = ["dqn", "ppo"], [0, 1, 2]
MAIN_GREEN, SIDE_GREEN, YELLOW = 50, 30, 4        # YELLOW=4: same simulator as the agents (fair); 5 = your field spec
NETS = ["4x4", "3x3"]
# ------------------------------------------------------------------ #
LABEL = f"Field plan {MAIN_GREEN}/{SIDE_GREEN} s green, {YELLOW} s yellow"
src = os.path.join(ROOT, "runs", "realistic_fixed_time.json")
if not os.path.exists(src): sys.exit("Run scripts/eval_field_fixed_time.py first.")
field = json.load(open(src))
for n in NETS:
    if LABEL not in field.get(n, {}): sys.exit(f"'{LABEL}' missing for {n} in runs/realistic_fixed_time.json (variant failed or not run).")


def plot_run(s, path, algo, net):
    nj = s["field_plan"]["n_agents"]; tuned = s.get("fixed_cycle_seconds_used")
    ent = [("random", "Random", "#993556"), ("field_plan", f"Field plan\n{MAIN_GREEN}/{SIDE_GREEN} s", "#C98A00"),
           ("fixed_cycle", f"Tuned fixed\n({tuned:g} s)" if isinstance(tuned, (int, float)) else "Tuned fixed", "#854F0B"),
           ("max_pressure", "Max-pressure", "#7B4FA0"), ("algo", algo.upper(), "#0F6E56")]
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
            s = json.load(open(p)); s["field_plan"] = copy.deepcopy(field[n][LABEL])
            json.dump(s, open(os.path.join(d, "summary_field.json"), "w"), indent=2)
            plot_run(s, os.path.join(ROOT, "runs", f"{a}_seed{sd}", "figures_field", f"baseline_comparison_{n}.png"), a, n); made += 1
print(f"{made} run summaries extended")

env = dict(os.environ, SUMMARY_NAME="summary_field.json", BASELINE_KEY="field_plan")
steps = [("compare algorithms " + n, ["diagnostics/compare_algorithms.py", "--algos", *ALGOS, "--seeds", *map(str, SEEDS),
          "--summary_name", f"eval_{n}/summary_field.json", "--out", os.path.join("runs", "aggregate", f"compare_algorithms_{n}_field.png")]) for n in NETS]
steps += [("topology generalization", ["scripts/make_topology_plot.py"]), ("results table", ["scripts/make_results_table.py"]), ("objective J table", ["scripts/compute_objective_J.py"])]
res = []
for label, args in steps:
    print(f"\n=== {label}", flush=True); res.append((label, subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode))
print("\n" + "=" * 50)
for label, rc in res: print(("ok   " if rc == 0 else "FAIL ") + label)
print("Outputs: runs/aggregate/*_field.* and runs/<algo>_seed<N>/figures_field/")
