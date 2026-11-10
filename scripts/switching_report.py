"""
scripts/switching_report.py -- Play button / `python scripts/switching_report.py`, a second, no SUMO.
Per run (algorithm, training seed, network) from runs/<run>/eval_<net>/summary.json: how often it switches and what the reward objective says.
  changes/junction = (n_switch_votes + n_forced_switches) / n_junctions   per 3600 s episode
  cycle            = 2 * 3600 / changes  (a full two-phase cycle, yellow included; always-switch = 30 s, 60 s plan = 60 s, 90 s plan = 90 s)
  J                = waiting per junction + lambda_s * (changes per junction per step) / (1 - gamma)      lower is better
                     (the same J as scripts/compute_objective_J.py: lambda_s = 0.3, gamma = 0.99, 720 steps per episode)
Fixed-time 90 s / 60 s rows are added from runs/realistic_fixed_time.json when it exists.
"""
import glob, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAM, GAMMA, STEPS, NJ = 0.3, 0.99, 720, {"4x4": 16, "3x3": 9}


def row(label, s, nj):
    ch = (s["n_switch_votes"]["mean"] + s["n_forced_switches"]["mean"]) / nj
    w = s["mean_waiting_time"]["mean"] / nj
    j = w + LAM * (ch / STEPS) / (1 - GAMMA)
    return f"{label:22s}{w:9.2f}{s['mean_vehicle_wait']['mean']:9.2f}{ch:10.1f}{(2 * 3600 / ch if ch else float('inf')):9.1f}{j:9.2f}"


rep_path = os.path.join(ROOT, "runs", "realistic_fixed_time.json")
rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
for net, nj in NJ.items():
    files = sorted(glob.glob(os.path.join(ROOT, "runs", "*_seed*", f"eval_{net}", "summary.json")))
    if not files:
        continue
    print(f"\n=== {net} ({nj} junctions)\n{'policy':22s}{'wait/jct':>9s}{'veh wait':>9s}{'chg/jct':>10s}{'cycle s':>9s}{'J':>9s}")
    ref = json.load(open(files[0]))
    for k, lab in (("random", "Random"), ("always_switch", "Always-switch"), ("max_pressure", "Max-pressure")):
        if k in ref:
            print(row(lab, ref[k], nj))
    for lab, s in rep.get(net, {}).items():
        if isinstance(s, dict) and "n_switch_votes" in s:
            print(row(lab, s, s.get("n_agents", nj)))
    for f in files:
        s = json.load(open(f))
        print(row(os.path.basename(os.path.dirname(os.path.dirname(f))), s["algo"], nj))
print("\nchg/jct: always-switch 240, 60 s plan 120, 90 s plan 80.  J: the reward objective (waiting + discounted switch cost).")
