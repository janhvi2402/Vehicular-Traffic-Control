"""Candidate max_pressure baseline (queue-based, per junction) vs always_switch / fixed 15 s, seed 1000, full episode."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import run_episode, fixed_cycle_policy_fn, always_switch_policy_fn, max_pressure_policy_fn
NET = os.environ.get("TLS_NET_FILE", "sumo_4x4_network/grid4x4.net.xml"); ROUTE = os.environ.get("TLS_ROUTE_FILE", "sumo_4x4_network/routes.rou.xml")


which = sys.argv[1]
env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE))
pol = {"always_switch": always_switch_policy_fn(16), "fixed15": fixed_cycle_policy_fn(15, 90)}.get(which) or max_pressure_policy_fn(margin=float(which.split("_m")[1]))
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
r = run_episode(env, pol, seed)
print(json.dumps({"policy": which, "seed": seed, **{k: round(float(r[k]), 2) for k in ("mean_waiting_time", "mean_vehicle_wait", "throughput", "mean_queue_length", "n_switch_votes", "n_forced_switches", "share_greens_at_max_green", "n_teleports")}}), flush=True)
