"""
diagnostics/compare_algorithms.py  -- PNG only; numbers printed to console.
DQN vs PPO vs baselines from the EXISTING summary.json of each training-seed run
(runs/<algo>_seed<N>/summary.json, written by evaluate.py). Algorithm bars: mean over
training seeds of each run's eval-episode mean, error bar = std over training seeds
(ddof=1). Baseline bars: mean over runs, error bar = eval-episode std of the first run.
Reads no checkpoint, so the footer lists the summary files used with a short content hash.

  python -m diagnostics.compare_algorithms --algos dqn ppo --seeds 0 1 2
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # Play button: find project root
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
LABELS = {"fixed_cycle": "tuned fixed-time (old)", "fixed_time_90": "Fixed-time 90 s", "fixed_time_60": "Fixed-time 60 s", "max_pressure": "max-pressure"}
COLORS = {"fixed_time_90": "#C98A00", "fixed_time_60": "#E6B422", "random": "#993556", "fixed_cycle": "#854F0B",
          "max_pressure": "#7A7A7A", "dqn": "#0F6E56", "ppo": "#2B5FA8", "ac": "#B5651D", "ca": "#7B3FA0"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default=os.path.join(PROJECT_ROOT, "runs"))
    p.add_argument("--algos", nargs="+", default=["dqn", "ppo"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--summary_name", default="eval_4x4/summary.json")
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
    BASELINES = [b for b in ("random", "fixed_cycle", "fixed_time_90", "fixed_time_60", "max_pressure") if all(b in r for r in base_runs)]
    for pol in BASELINES:
        for m, _ in METRICS:
            v = [r[pol][m]["mean"] for r in base_runs if m in r[pol]]
            if v and (max(v) - min(v)) > 1e-6 * max(1.0, abs(np.mean(v))):
                print(f"WARNING: {pol}/{m} differs between runs (min {min(v):.3f}, max {max(v):.3f}) "
                      f"- runs were not evaluated against identical baselines")
    for key in ("fixed_time_90", "fixed_time_60"):
        if key in BASELINES:
            splits = {(r[key].get("split_main_green"), r[key].get("split_side_green")) for r in base_runs}
            if len(splits) > 1:
                print(f"WARNING: {key} uses different main/side splits across runs: {sorted(map(str, splits))}")
    splits_txt = "   ".join(f"{k}: main {base_runs[0][k].get('split_main_green')}/side {base_runs[0][k].get('split_side_green')}"
                          for k in ("fixed_time_90", "fixed_time_60") if k in BASELINES)

    policies = BASELINES + [a for a in args.algos if S[a]]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4.8))
    print(f"\n{'metric':36s}" + "".join(f"{p_:>22s}" for p_ in policies))
    for ax, (m, title) in zip(axes, METRICS):
        means, errs, pts = [], [], []
        for pol in policies:
            if pol in BASELINES:
                vals = [r[pol][m]["mean"] for r in base_runs if m in r[pol]]
                err = base_runs[0][pol][m]["std"] if vals else 0.0
            else:
                vals = [r["algo"][m]["mean"] for r in S[pol] if m in r["algo"]]
                err = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            means.append(float(np.mean(vals)) if vals else np.nan); errs.append(err); pts.append(vals)
        ax.bar([LABELS.get(p_, p_) for p_ in policies], means, yerr=errs, capsize=5, color=[COLORS.get(p_, "#777777") for p_ in policies])
        for i, v in enumerate(pts):
            if policies[i] in S and len(v) > 1:
                ax.scatter([i] * len(v), v, color="k", s=12, zorder=3)
        ax.set_title(title, fontsize=10); ax.tick_params(axis="x", labelrotation=15)
        print(f"{m:36s}" + "".join(f"{(f'{mu:.2f}±{e:.2f}'):>22s}" for mu, e in zip(means, errs)))
        if m == "mean_waiting_time":
            for ref_key in ("fixed_cycle", "fixed_time_90", "fixed_time_60"):
                if ref_key in policies:
                    rv = means[policies.index(ref_key)]
                    print(f"{'  improvement vs ' + LABELS[ref_key] + ' %':36s}" + "".join(f"{100 * (rv - mu) / rv:>22.1f}" for mu in means))
    fig.suptitle("Algorithms vs baselines (algo error bars: std over training seeds; dots: individual seeds; "
                 "baseline error bars: std over eval episodes)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    if splits_txt:
        fig.text(0.005, 0.03, splits_txt, fontsize=8, color="0.25")
    fig.text(0.005, 0.004, "summaries: " + " | ".join(f"{os.path.relpath(f, args.runs_dir)}#{file_hash(f)}" for f in files),
             fontsize=5.5, color="0.35")
    out = args.out or os.path.join(args.runs_dir, "aggregate", "compare_algorithms.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nsaved: {out}   (n runs: {{{', '.join(f'{a}: {len(S[a])}' for a in S)}}})")


if __name__ == "__main__":
    main()
