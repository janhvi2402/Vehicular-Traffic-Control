"""
diagnostics/compare_algorithms.py  -- PNG only; numbers printed to console.
DQN vs PPO vs baselines from the EXISTING summary.json of each training-seed run
(runs/<algo>_seed<N>/summary.json, written by evaluate.py). Algorithm bars: mean over
training seeds of each run's eval-episode mean, error bar = std over training seeds
(ddof=1). Baseline bars: mean over runs, error bar = eval-episode std of the first run.
Reads no checkpoint, so the footer lists the summary files used with a short content hash.

  python -m diagnostics.compare_algorithms --algos dqn ppo --seeds 0 1 2
"""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.diag_common import PROJECT_ROOT, file_hash

METRICS = [("mean_waiting_time", "Mean waiting time per step (s)"),
           ("mean_vehicle_wait", "Mean wait per vehicle (s)"),
           ("mean_queue_length", "Mean queue length (vehicles)"),
           ("throughput", "Throughput (vehicles arrived)")]
COLORS = {"random": "#993556", "fixed_cycle": "#854F0B", "dqn": "#0F6E56", "ppo": "#2B5FA8"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=os.path.join(PROJECT_ROOT, "runs"))
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--summary_name", default="summary.json")
    p.add_argument("--out", default=None)
    args, _ = p.parse_known_args()

    S, files = {}, []
    for a in args.algos:
        S[a] = []
        for s in args.seeds:
            f = os.path.join(args.runs_dir, f"{a}_seed{s}", args.summary_name)
            if not os.path.exists(f):
                print(f"MISSING: {f}"); continue
            S[a].append(json.load(open(f))); files.append(f)
    if not files:
        raise SystemExit("no summary.json files found")

    base_runs = [s for a in S for s in S[a]]
    for pol in ("random", "fixed_cycle"):
        for m, _ in METRICS:
            v = [r[pol][m]["mean"] for r in base_runs if m in r[pol]]
            if v and (max(v) - min(v)) > 1e-6 * max(1.0, abs(np.mean(v))):
                print(f"WARNING: {pol}/{m} differs between runs (min {min(v):.3f}, max {max(v):.3f}) "
                      f"- runs were not evaluated against identical baselines")
    cyc = {r.get("fixed_cycle_seconds_used") for r in base_runs}
    if len(cyc) > 1:
        print(f"WARNING: different fixed-cycle lengths used across runs: {sorted(map(str, cyc))}")

    policies = ["random", "fixed_cycle"] + [a for a in args.algos if S[a]]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.8))
    print(f"\n{'metric':22s}" + "".join(f"{p_:>22s}" for p_ in policies))
    for ax, (m, title) in zip(axes, METRICS):
        means, errs, pts = [], [], []
        for pol in policies:
            if pol in ("random", "fixed_cycle"):
                vals = [r[pol][m]["mean"] for r in base_runs if m in r[pol]]
                err = base_runs[0][pol][m]["std"] if vals else 0.0
            else:
                vals = [r["algo"][m]["mean"] for r in S[pol] if m in r["algo"]]
                err = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            means.append(float(np.mean(vals)) if vals else np.nan); errs.append(err); pts.append(vals)
        ax.bar(policies, means, yerr=errs, capsize=5, color=[COLORS[p_] for p_ in policies])
        for i, v in enumerate(pts):
            if policies[i] in S and len(v) > 1:
                ax.scatter([i] * len(v), v, color="k", s=12, zorder=3)
        ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelrotation=15)
        print(f"{m:22s}" + "".join(f"{(f'{mu:.2f}±{e:.2f}'):>22s}" for mu, e in zip(means, errs)))
        if m == "mean_waiting_time" and "fixed_cycle" in policies:
            fx = means[policies.index("fixed_cycle")]
            print(f"{'  improvement vs fixed %':22s}" + "".join(f"{100 * (fx - mu) / fx:>22.1f}" for mu in means))
    fig.suptitle("Algorithms vs baselines (algo error bars: std over training seeds; dots: individual seeds; "
                 "baseline error bars: std over eval episodes)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.text(0.005, 0.004, "summaries: " + " | ".join(f"{os.path.relpath(f, args.runs_dir)}#{file_hash(f)}" for f in files),
             fontsize=5.5, color="0.35")
    out = args.out or os.path.join(args.runs_dir, "aggregate", "compare_algorithms.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nsaved: {out}   (n runs: {{{', '.join(f'{a}: {len(S[a])}' for a in S)}}})")


if __name__ == "__main__":
    main()
