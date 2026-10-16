"""
scripts/compute_objective_J.py -- Play button, no arguments, takes seconds. Run after scripts/run_eval.py.
Shows the trade-off the agents were actually trained on. The reward is  -(change in waiting time) - lambda_s per phase change,
so the discounted return is about  -J  with
    J = mean waiting time per junction  +  lambda_s * (phase changes per junction per step) / (1 - gamma)      (lower is better)
(derivation in scripts/switch_cost_probe.py). A policy can lose on waiting time alone and still win on J by switching less.
Reads runs/<algo>_seed<N>/eval_<net>/summary.json and, if present, runs/realistic_fixed_time.json.
"""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGOS, SEEDS = ["dqn", "ppo"], [0, 1, 2]
NETS = [("4x4", 16), ("3x3", 9)]
LAMBDA_S, GAMMA, STEPS = 0.3, 0.99, 720
BASE = [("random", "Random"), ("always_switch", "Always-switch (= fixed 10)"), ("fixed_cycle", "Fixed-cycle (tuned)"), ("max_pressure", "Max-pressure")]


def terms(s, nj):
    w = s["mean_waiting_time"]["mean"] / nj
    ch = s["n_switch_votes"]["mean"] + s["n_forced_switches"]["mean"]
    sw = LAMBDA_S * (ch / nj / STEPS) / (1 - GAMMA)
    return w, ch / nj, sw, w + sw


rep = json.load(open(os.path.join(ROOT, "runs", "realistic_fixed_time.json"))) if os.path.exists(os.path.join(ROOT, "runs", "realistic_fixed_time.json")) else {}
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
        rows.append((label, [terms(s, s.get("n_agents", nj))]))
    rows += [(a.upper(), [terms(r["algo"], nj) for r in v]) for a, v in runs.items()]
    jf = next(r[1][0][3] for r in rows if r[0].startswith("Fixed-cycle")); wf = next(r[1][0][0] for r in rows if r[0].startswith("Fixed-cycle"))
    print(f"\n=== {net}  (lambda_s={LAMBDA_S}, gamma={GAMMA}; per junction; algorithms: mean over seeds)")
    print(f"{'policy':44s}{'waiting':>9s}{'changes':>9s}{'switch term':>12s}{'J':>8s}{'vs fixed: waiting':>19s}{'J':>8s}")
    for lab, tl in rows:
        m = np.mean(np.array(tl), axis=0)
        print(f"{lab:44s}{m[0]:9.2f}{m[1]:9.1f}{m[2]:12.2f}{m[3]:8.2f}{100 * (wf - m[0]) / wf:18.1f}%{100 * (jf - m[3]) / jf:7.1f}%")
print("\nwaiting = mean waiting per junction; changes = phase changes per junction per episode; positive 'vs fixed' = better than the tuned fixed-cycle.")
