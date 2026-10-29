"""
scripts/compute_objective_J.py -- Play button, no arguments, takes seconds. Run after scripts/run_eval.py (and scripts/eval_fixed_time.py).
Shows the trade-off the agents were actually trained on. The reward is  -(change in waiting time) - lambda_s per phase change,
so the discounted return is about  -J  with
    J = mean waiting time per junction  +  lambda_s * (phase changes per junction per step) / (1 - gamma)      (lower is better)
(derivation in scripts/switch_cost_probe.py). A policy can lose on waiting time alone and still win on J by switching less.
Reads runs/<algo>_seed<N>/eval_<net>/summary.json and runs/realistic_fixed_time.json (the Fixed-time 90 s / 60 s plans).
The table is printed twice per network: once with Fixed-time 90 s as the reference row, once with Fixed-time 60 s.
"""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGOS, SEEDS = ["dqn", "ppo"], [0, 1, 2]
NETS = [("4x4", 16), ("3x3", 9)]
LAMBDA_S, GAMMA, STEPS = 0.3, 0.99, 720
BASE = [("random", "Random"), ("always_switch", "Always-switch (10 s, old baseline)"), ("fixed_cycle", "Tuned fixed-cycle (10 s, old baseline)"), ("max_pressure", "Max-pressure")]


def terms(s, nj):
    w = s["mean_waiting_time"]["mean"] / nj
    ch = s["n_switch_votes"]["mean"] + s["n_forced_switches"]["mean"]
    sw = LAMBDA_S * (ch / nj / STEPS) / (1 - GAMMA)
    return w, ch / nj, sw, w + sw


rep_path = os.path.join(ROOT, "runs", "realistic_fixed_time.json")
rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
for net, nj in NETS:
    runs = {}
    for a in ALGOS:
        for s in SEEDS:
            p = os.path.join(ROOT, "runs", f"{a}_seed{s}", f"eval_{net}", "summary.json")
            if os.path.exists(p): runs.setdefault(a, []).append(json.load(open(p)))
    if not runs: print(f"[{net}] no summaries"); continue
    ref = next(iter(runs.values()))[0]
    rows = [(lab, [terms(ref[k], nj)]) for k, lab in BASE if k in ref]
    for label, s in rep.get(net, {}).items():
        rows.append((f"{label} (main {s['split_main_green']}/side {s['split_side_green']})", [terms(s, s.get("n_agents", nj))]))
    rows += [(a.upper(), [terms(r["algo"], nj) for r in v]) for a, v in runs.items()]
    for ref_name in ("Fixed-time 90 s", "Fixed-time 60 s"):
        ref_rows = [r for r in rows if r[0].startswith(ref_name)]
        if not ref_rows:
            print(f"\n[{net}] reference '{ref_name}' not found in runs/realistic_fixed_time.json -- run scripts/eval_fixed_time.py"); continue
        wf, jf = ref_rows[0][1][0][0], ref_rows[0][1][0][3]
        print(f"\n=== {net}  reference row: {ref_name}  (lambda_s={LAMBDA_S}, gamma={GAMMA}; per junction; algorithms: mean over seeds)")
        print(f"{'policy':54s}{'waiting':>9s}{'changes':>9s}{'switch term':>12s}{'J':>8s}{'vs ref: waiting':>17s}{'J':>8s}")
        for lab, tl in rows:
            m = np.mean(np.array(tl), axis=0)
            print(f"{lab:54s}{m[0]:9.2f}{m[1]:9.1f}{m[2]:12.2f}{m[3]:8.2f}{100 * (wf - m[0]) / wf:16.1f}%{100 * (jf - m[3]) / jf:7.1f}%")
print("\nwaiting = mean waiting per junction; changes = phase changes per junction per episode; positive 'vs ref' = better than the reference row.")
