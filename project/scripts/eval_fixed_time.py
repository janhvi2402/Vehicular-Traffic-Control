"""
scripts/eval_fixed_time.py -- Play button, no arguments (about 40 min). Touches NO trained model and NO existing results.
Evaluates two fixed-time plans, Fixed-time 90 s and Fixed-time 60 s, on the SAME evaluation seeds (1000..1009) as the learned policies.
  * phase 0 = E-W green = main road (4 lanes); phase 2 = N-S green = side road (2 lanes); 4 s yellow/all-red after each green (yellow_time=4 only);
  * the main/side green split of each plan is tuned on sweep seeds 2000-2002 by scripts/sweep_fixed_time_split.py (runs/fixed_time_choice.json);
  * all junctions start together with no offsets (no green wave).
A plan's cycle = main green + side green + 10 s. Decisions come every 5 s and the elapsed green seen at a decision is 1, 6, 11, ... (the first green
of an episode starts at 0), so a real green is threshold + 1 s (threshold itself for the first green of each episode).
Writes runs/realistic_fixed_time.json (read by scripts/compute_objective_J.py), one entry "Fixed-time <cycle> s" per network, and prints PASS/FAIL
sanity checks after every evaluation (phase changes per junction, no forced switches, green-duration pattern).
"""
import hashlib, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import run_episode, run_policy_over_seeds, summarize

CYCLES = [90, 60]
EPISODES, BASE_SEED = 10, 1000
NETS = {"4x4": ("sumo_4x4_network", ["grid4x4.net.xml", "grid4x4_net.xml"], ["routes.rou.xml", "routes_rou.xml"]),
        "3x3": ("sumo_3x3_network", ["grid3x3.net.xml", "grid3x3_net.xml"], ["routes3x3.rou.xml", "routes3x3_rou.xml"])}
CHOICE = os.path.join(ROOT, "runs", "fixed_time_choice.json")


def plan_label(cyc):
    return f"Fixed-time {cyc} s"


def fixed_time_policy(main_green, side_green, max_green):
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


def files_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def sanity_checks(env, policy, s, cyc, main_g, side_g, changes):
    expected = 3600.0 / cyc * 2                                         # two phase changes per cycle
    print(f"   {'PASS' if abs(changes - expected) <= 0.05 * expected else 'FAIL'}  phase changes per junction per episode {changes:.1f} (expected about {expected:.0f}, +-5%)")
    share = s["share_greens_at_max_green"]["mean"]
    print(f"   {'PASS' if share == 0 else 'FAIL'}  share_greens_at_max_green = {share:.4f} (expected 0)")
    _, greens = run_episode(env, policy, BASE_SEED, return_events=True)
    per = {}
    for g in greens: per.setdefault(g["junction"], []).append(g)
    total = match = 0
    for gs in per.values():
        gs.sort(key=lambda g: g["start"])
        for i, g in enumerate(gs):                                      # even index = main (phase 0), odd = side (phase 2)
            exp = (main_g if i == 0 else main_g + 1) if i % 2 == 0 else side_g + 1
            total += 1; match += int(abs((g["end"] - g["start"]) - exp) < 0.5)
    frac = match / total if total else 0.0
    print(f"   {'PASS' if frac >= 0.99 else 'FAIL'}  green durations match the plan for {100 * frac:.1f}% of {total} greens (main {main_g + 1} s (first {main_g}), side {side_g + 1} s; need >= 99%)", flush=True)


if __name__ == "__main__":
    if not os.path.exists(CHOICE):
        sys.exit("ERROR: run scripts/sweep_fixed_time_split.py first (it tunes the main/side split of each plan).")
    choice = json.load(open(CHOICE))
    path = os.path.join(ROOT, "runs", "realistic_fixed_time.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    seeds, out = [BASE_SEED + i for i in range(EPISODES)], {}           # starts empty: no stale rows from earlier runs
    for tag, (folder, nets, routes) in NETS.items():
        net_path, route_path = find(folder, nets), find(folder, routes)
        if choice.get("files_hash", {}).get(tag) != files_hash(net_path, route_path):
            sys.exit(f"ERROR: the {tag} network/route files changed since the sweep -- re-run scripts/sweep_fixed_time_split.py.")
        env = TrafficGridEnv(GridEnvConfig(net_file=net_path, route_file=route_path, yellow_time=4))
        out[tag] = {}
        for cyc in CYCLES:
            main_g, side_g = choice[tag][str(cyc)]["main"], choice[tag][str(cyc)]["side"]
            policy = fixed_time_policy(main_g, side_g, env.cfg.max_green)
            s = summarize(run_policy_over_seeds(env, policy, seeds))
            s["n_agents"] = env.n_agents; s["cycle_seconds"] = cyc; s["split_main_green"] = main_g; s["split_side_green"] = side_g
            out[tag][plan_label(cyc)] = s
            changes = (s["n_switch_votes"]["mean"] + s["n_forced_switches"]["mean"]) / env.n_agents
            print(f"[{tag}] {plan_label(cyc)} (main {main_g}/side {side_g}): waiting {s['mean_waiting_time']['mean']:.1f}, "
                  f"phase changes per junction per episode {changes:.1f}, teleports {s['n_teleports']['mean']:.1f}", flush=True)
            sanity_checks(env, policy, s, cyc, main_g, side_g, changes)
        env.close()
        json.dump(out, open(path, "w"), indent=2)
    print("saved", path)
