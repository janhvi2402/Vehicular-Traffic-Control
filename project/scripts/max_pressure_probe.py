"""Candidate max_pressure baseline (queue-based, per junction) vs always_switch / fixed 15 s, seed 1000, full episode.
Play button (no arguments): runs every variant below once. Terminal: python scripts/max_pressure_probe.py max_m2 1000"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import run_episode, fixed_cycle_policy_fn, always_switch_policy_fn, max_pressure_policy_fn
NET = os.environ.get("TLS_NET_FILE", os.path.join(ROOT, "sumo_4x4_network", "grid4x4.net.xml"))
ROUTE = os.environ.get("TLS_ROUTE_FILE", os.path.join(ROOT, "sumo_4x4_network", "routes.rou.xml"))
VARIANTS = ["always_switch", "fixed15", "max_m0", "max_m2", "max_m5"]   # "max_m<margin>" = max-pressure with that margin

def make_policy(which, n_agents):
    if which == "always_switch": return always_switch_policy_fn(n_agents)
    if which == "fixed15": return fixed_cycle_policy_fn(15, 90)
    return max_pressure_policy_fn(margin=float(which.split("_m")[1]))

which_list = [sys.argv[1]] if len(sys.argv) > 1 else VARIANTS
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE))
for which in which_list:
    r = run_episode(env, make_policy(which, env.n_agents), seed)
    print(json.dumps({"policy": which, "seed": seed, **{k: round(float(r[k]), 2) for k in ("mean_waiting_time", "mean_vehicle_wait", "throughput", "mean_queue_length", "n_switch_votes", "n_forced_switches", "share_greens_at_max_green", "n_teleports")}}), flush=True)
env.close()
