"""
scripts/sweep_fixed_cycle.py   -- run ONCE per network (Play button), BEFORE scripts/run_eval.py.
Tunes the fixed-cycle baseline once and writes runs/fixed_cycle_choice.json, so DQN, PPO, AC and CA are all compared
against the SAME tuned baseline (and nobody re-sweeps for an hour per evaluation).

Candidate = "switch once the green has lasted >= c seconds". Decisions happen every 5 s and elapsed green time at a
decision is 1, 6, 11, 16, ..., so the real green is 11, 16, 21, 26, ... s (+4 s yellow). Candidates include 10 so the
optimum is not cut off at the edge of the search range. Tuned on SUMO seeds 2000-2002 (never used for evaluation).
RE-RUN this if the network or route files change (e.g. after editing routes3x3.rou.xml): run_eval.py checks a file hash.
"""
import hashlib, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import fixed_cycle_policy_fn, run_episode

CANDIDATES = [10, 15, 20, 25, 30, 45, 60]
SWEEP_SEEDS = [2000, 2001, 2002]
NETS = {"4x4": ("sumo_4x4_network", ["grid4x4.net.xml", "grid4x4_net.xml"], ["routes.rou.xml", "routes_rou.xml"]),
        "3x3": ("sumo_3x3_network", ["grid3x3.net.xml", "grid3x3_net.xml"], ["routes3x3.rou.xml", "routes3x3_rou.xml"])}
OUT = os.path.join(ROOT, "runs", "fixed_cycle_choice.json")


def find(folder, names):
    for n in names:
        p = os.path.join(ROOT, folder, n)
        if os.path.exists(p): return p
    sys.exit(f"ERROR: none of {names} found in {folder}/")


def files_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    result = {}
    for tag, (folder, nets, routes) in NETS.items():
        net, route = find(folder, nets), find(folder, routes)
        env = TrafficGridEnv(GridEnvConfig(net_file=net, route_file=route))
        means = {}
        for c in CANDIDATES:
            vals = [run_episode(env, fixed_cycle_policy_fn(c, env.cfg.max_green), s)["mean_waiting_time"] for s in SWEEP_SEEDS]
            means[c] = float(np.mean(vals)); print(f"[{tag}] cycle {c:>3d}s -> mean_waiting_time {means[c]:.1f}", flush=True)
        env.close()
        best = min(means, key=means.get)
        print(f"[{tag}] chosen fixed cycle: {best}s" + ("   <-- smallest candidate: optimum may be lower, report this" if best == min(CANDIDATES) else ""))
        result[tag] = {"chosen": best, "mean_waiting_time_by_cycle": {str(k): v for k, v in means.items()},
                       "seeds": SWEEP_SEEDS, "files_hash": files_hash(net, route)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")
