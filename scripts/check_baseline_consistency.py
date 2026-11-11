"""
scripts/check_baseline_consistency.py -- Play button / `python scripts/check_baseline_consistency.py`, takes a second, no SUMO.
Reads every runs/<run>/eval_<net>/summary.json and checks that all runs of the same network were evaluated against IDENTICAL baselines
(random, old fixed-cycle, always-switch, max-pressure). Baselines are deterministic given the evaluation seeds, so any difference means
the runs were evaluated with a different evaluator, route file or setting and must be re-evaluated before they are compared.
Also lists the file date of each summary (old dates = produced by an older evaluate.py) and the fixed-cycle seconds that were used.
"""
import datetime, glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["random", "fixed_cycle", "always_switch", "max_pressure"]
METRICS = ["mean_waiting_time", "mean_vehicle_wait", "mean_queue_length", "throughput"]
bad = 0
for net in ("4x4", "3x3"):
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "*", f"eval_{net}", "summary.json")))
    if not files:
        continue
    print(f"\n=== {net}: {len(files)} summaries")
    data, sumos = {}, {}
    for f in files:
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        s = json.load(open(f)); data[run] = s
        when = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%d-%m-%Y %H:%M")
        missing = [b for b in BASE + ["algo"] if b not in s]
        sm = s.get("sumo")
        print(f"  {run:12s} written {when}  fixed_cycle_seconds_used={s.get('fixed_cycle_seconds_used')}"
              f"  sumo={(sm['version'] + ' @ ' + str(sm['binary'])) if sm else 'unrecorded'}"
              + (f"  MISSING KEYS: {missing}" if missing else ""))
        if sm:
            sumos.setdefault((sm["version"], sm["binary"]), []).append(run)
        bad += bool(missing)
    if len(sumos) > 1:
        bad += 1
        print("  SUMO MISMATCH: runs were evaluated under different simulators: "
              + "; ".join(f"{k[0]} @ {k[1]} -> {v}" for k, v in sumos.items()))
    ref_run = next(iter(data))
    for b in BASE:
        for m in METRICS:
            vals = {r: s[b][m]["mean"] for r, s in data.items() if b in s and m in s[b]}
            if len(vals) < 2:
                continue
            lo, hi = min(vals.values()), max(vals.values())
            if hi - lo > 1e-6 * max(1.0, abs(hi)):
                bad += 1
                print(f"  MISMATCH {b}/{m}: " + ", ".join(f"{r}={v:.4f}" for r, v in sorted(vals.items())))
print("\nOK: every run of each network used identical baselines." if not bad else
      f"\n{bad} problem(s): runs that differ from the rest (or lack keys) should be re-evaluated before comparison.")
sys.exit(1 if bad else 0)
