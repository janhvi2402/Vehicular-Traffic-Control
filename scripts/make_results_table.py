"""
scripts/make_results_table.py -- Play button, no arguments. Run it AFTER scripts/run_eval.py has finished.
Builds the numbers table for the report from runs/<algo>_seed<N>/eval_<net>/summary.json:
  learned algorithms: mean +- std over the training seeds;  baselines: mean (+- std over evaluation episodes).
Writes runs/aggregate/results_table.md and results_table.csv and prints the markdown.
"""
import csv, json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGOS = ["dqn", "ppo"]
SEEDS = [0, 1, 2]
NETS = [("4x4", 16), ("3x3", 9)]
BASELINES = [("random", "Random"), ("always_switch", "Always-switch"), ("fixed_cycle", "Fixed-cycle"), ("max_pressure", "Max-pressure")]
METRICS = [("mean_waiting_time", "Waiting time (veh*s, per-step mean)"), ("mean_vehicle_wait", "Wait per vehicle (s)"),
           ("mean_queue_length", "Queue length (veh)"), ("throughput", "Throughput (veh)"),
           ("phase_changes", "Phase changes / junction / episode"), ("n_teleports", "Teleports / episode")]
IMPROVE = ["mean_waiting_time", "mean_vehicle_wait", "mean_queue_length"]


def value(r, key, m, nj):
    if m == "phase_changes": return (r[key]["n_switch_votes"]["mean"] + r[key]["n_forced_switches"]["mean"]) / nj
    return r[key][m]["mean"]


md, rows = [], []
for net, nj in NETS:
    runs = {}
    for a in ALGOS:
        for s in SEEDS:
            p = os.path.join(ROOT, "runs", f"{a}_seed{s}", f"eval_{net}", "summary.json")
            if os.path.exists(p): runs.setdefault(a, []).append(json.load(open(p)))
            else: print(f"missing: {p}")
    if not runs: continue
    ref = next(iter(runs.values()))[0]
    allr = [r for v in runs.values() for r in v]
    for b, _ in BASELINES:
        if all(b in r for r in allr):
            vals = [r[b]["mean_waiting_time"]["mean"] for r in allr]
            if max(vals) - min(vals) > 1e-6 * max(1, abs(np.mean(vals))): print(f"WARNING [{net}] baseline {b} differs between runs -- not evaluated identically")
    fx = {m: np.mean([value(r, "fixed_cycle", m, nj) for r in allr]) for m, _ in METRICS}
    head = ["Policy"] + [lab for _, lab in METRICS] + [f"Improvement vs fixed-cycle: {lab.split(' (')[0]} (%)" for m, lab in METRICS if m in IMPROVE]
    md += [f"### {net} grid  (fixed-cycle = {ref.get('fixed_cycle_seconds_used')} s; algorithms: mean ± std over {', '.join(f'{a}: {len(v)}' for a, v in runs.items())} training seeds)", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    entries = [(b, lab, None) for b, lab in BASELINES if all(b in r for r in allr)] + [(a, a.upper(), runs[a]) for a in runs]
    for key, lab, rr in entries:
        cells, flat = [lab], {"network": net, "policy": lab}
        for m, mlab in METRICS:
            if rr is None:
                v = np.mean([value(r, key, m, nj) for r in allr]); sd = ref[key][m]["std"] if m != "phase_changes" and m in ref[key] else None
                cells.append(f"{v:.2f}" + (f" (±{sd:.2f}, eval eps)" if sd is not None else "")); flat[m] = v
            else:
                x = [value(r, "algo", m, nj) for r in rr]; v, sd = np.mean(x), (np.std(x, ddof=1) if len(x) > 1 else 0.0)
                cells.append(f"{v:.2f} ± {sd:.2f}"); flat[m] = v; flat[m + "_std_over_seeds"] = sd
        for m in IMPROVE:
            if rr is None: x = [100 * (fx[m] - value(r, key, m, nj)) / fx[m] for r in allr[:1]]
            else: x = [100 * (fx[m] - value(r, "algo", m, nj)) / fx[m] for r in rr]
            cells.append(f"{np.mean(x):.1f}" + (f" ± {np.std(x, ddof=1):.1f}" if rr is not None and len(x) > 1 else "")); flat["improve_" + m] = np.mean(x)
        md.append("| " + " | ".join(cells) + " |"); rows.append(flat)
    md.append("")
md += ["Notes: waiting time is the sum of SUMO per-vehicle waiting time over all incoming lanes (default 100 s memory), averaged over steps; "
       "wait per vehicle counts only completed trips; baselines are deterministic given the evaluation seeds."]
out_dir = os.path.join(ROOT, "runs", "aggregate"); os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir, "results_table.md"), "w").write("\n".join(md) + "\n")
if rows:
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("network", "policy"), k))
    with open(os.path.join(out_dir, "results_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print("\n".join(md)); print(f"\nsaved to {out_dir}")
