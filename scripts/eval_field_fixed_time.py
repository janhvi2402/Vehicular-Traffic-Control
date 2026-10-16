"""
scripts/eval_field_fixed_time.py -- Play button, no arguments (about 40-60 min). Touches NO trained model and NO existing code.
Evaluates a field-style fixed-time plan on the SAME evaluation seeds/episodes as the learned policies:
    main street (E-W, phase 0): MAIN_GREEN s green;  side street (N-S, phase 2): SIDE_GREEN s green;  YELLOW s clearance after each.
Decisions come every 5 s, so with the environment's 4 s yellow the elapsed green seen at a decision is 1, 6, 11, ... and the
plan is realised as 51 s / 31 s (cycle 51+4+31+4 = 90 s, same as 50+5+30+5). With a 5 s yellow the greens are exactly 50 / 30.
Variants run (edit YELLOWS):  4 = the SAME environment the agents were trained/evaluated in (the fair comparison);
                              5 = your field spec (all-red included); a slightly different environment, so report it as a sensitivity check.
All 16 (or 9) junctions start together with no offsets (no green wave): a limitation to state.
Writes runs/realistic_fixed_time.json (read by scripts/compute_objective_J.py).
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import run_policy_over_seeds, summarize

MAIN_GREEN, SIDE_GREEN = 50, 30
YELLOWS = [4, 5]
EPISODES, BASE_SEED = 10, 1000
NETS = {"4x4": ("sumo_4x4_network", ["grid4x4.net.xml", "grid4x4_net.xml"], ["routes.rou.xml", "routes_rou.xml"]),
        "3x3": ("sumo_3x3_network", ["grid3x3.net.xml", "grid3x3_net.xml"], ["routes3x3.rou.xml", "routes3x3_rou.xml"])}


def field_plan_policy(main_green, side_green, max_green):
    """Vote to switch when the current green (phase 0 = E+W main, phase 2 = N+S side) has lasted >= its planned green."""
    def policy(obs):
        f = obs["node_features"]
        phase = np.rint(f[:, 8] * 3).astype(int)                       # 0,2 = green phases; 1,3 = yellow
        elapsed = f[:, 10].astype(np.float64) * max_green
        need = np.where(phase == 0, main_green, side_green)
        return (((phase % 2) == 0) & (elapsed >= need - 1e-4)).astype(np.int64)
    return policy


def find(folder, names):
    for n in names:
        p = os.path.join(ROOT, folder, n)
        if os.path.exists(p): return p
    sys.exit(f"ERROR: none of {names} found in {folder}/")


if __name__ == "__main__":
    path = os.path.join(ROOT, "runs", "realistic_fixed_time.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    seeds = [BASE_SEED + i for i in range(EPISODES)]
    for tag, (folder, nets, routes) in NETS.items():
        out.setdefault(tag, {})
        for y in YELLOWS:
            label = f"Field plan {MAIN_GREEN}/{SIDE_GREEN} s green, {y} s yellow"
            try:
                env = TrafficGridEnv(GridEnvConfig(net_file=find(folder, nets), route_file=find(folder, routes), yellow_time=y))
                s = summarize(run_policy_over_seeds(env, field_plan_policy(MAIN_GREEN, SIDE_GREEN, env.cfg.max_green), seeds))
                s["n_agents"] = env.n_agents; env.close()
                out[tag][label] = s
                print(f"[{tag}] {label}: waiting {s['mean_waiting_time']['mean']:.1f}, per-vehicle wait {s['mean_vehicle_wait']['mean']:.1f}, "
                      f"queue {s['mean_queue_length']['mean']:.1f}, phase changes {s['n_switch_votes']['mean'] + s['n_forced_switches']['mean']:.0f}, "
                      f"teleports {s['n_teleports']['mean']:.1f}", flush=True)
            except Exception as e:                       # one failing variant must not lose the others
                print(f"[{tag}] {label}: FAILED ({e!r})", flush=True)
        json.dump(out, open(path, "w"), indent=2)
    print("saved", path)
