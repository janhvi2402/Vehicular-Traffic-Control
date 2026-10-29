"""
scripts/sweep_fixed_time_split.py -- Play button, no arguments, run ONCE before scripts/eval_fixed_time.py (about 40 min: 12 splits x 3 seeds x 2 grids).
Tunes the main/side green split of each fixed-time plan, per network, on SUMO seeds 2000-2002 (never used for evaluation). yellow_time=4 only.
Candidate = (main green threshold, side green threshold); the plan's cycle is main + side + 10 s. To disable tuning, leave one candidate per cycle:
(50, 30) and (30, 20). RE-RUN this if the network or route files change: eval_fixed_time.py checks a file hash.
Writes runs/fixed_time_choice.json.
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import run_episode
from scripts.eval_fixed_time import fixed_time_policy, find, files_hash, NETS

CANDIDATES = {90: [(50, 30), (40, 40), (30, 50)], 60: [(30, 20), (25, 25), (20, 30)]}
SWEEP_SEEDS = [2000, 2001, 2002]
OUT = os.path.join(ROOT, "runs", "fixed_time_choice.json")

if __name__ == "__main__":
    result, hashes = {}, {}
    for tag, (folder, nets, routes) in NETS.items():
        net_path, route_path = find(folder, nets), find(folder, routes)
        env = TrafficGridEnv(GridEnvConfig(net_file=net_path, route_file=route_path, yellow_time=4))
        result[tag] = {}
        for cyc, cands in CANDIDATES.items():
            means = {}
            for main_g, side_g in cands:
                policy = fixed_time_policy(main_g, side_g, env.cfg.max_green)
                vals = [run_episode(env, policy, sd)["mean_waiting_time"] for sd in SWEEP_SEEDS]
                means[f"{main_g}/{side_g}"] = float(np.mean(vals))
                print(f"[{tag}] {cyc} s cycle, split {main_g}/{side_g} -> mean_waiting_time {means[f'{main_g}/{side_g}']:.1f}", flush=True)
            best = min(means, key=means.get); main_g, side_g = (int(x) for x in best.split("/"))
            keys = list(means)
            if len(keys) > 1 and best in (keys[0], keys[-1]):
                print(f"WARNING [{tag}] {cyc} s: the best split {best} is at the edge of the candidates -- the optimum may lie outside the range")
            print(f"[{tag}] {cyc} s cycle: chosen split main {main_g} / side {side_g}", flush=True)
            result[tag][str(cyc)] = {"main": main_g, "side": side_g, "mean_waiting_time_by_split": means}
        env.close(); hashes[tag] = files_hash(net_path, route_path)
    result["seeds"] = SWEEP_SEEDS; result["files_hash"] = hashes
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")
