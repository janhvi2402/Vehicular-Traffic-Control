"""
Issue-2 probe: where does the reward's optimum sit among hand-coded timing policies?
Per episode (seed 1000, full 3600 s) for fixed cycles c and 'forced' (never vote, max_green forces):
  w_bar  = mean per-junction waiting time           (the waiting term, an average level in return units)
  s_term = lambda_s * (phase changes / junction / step) * 1/(1-gamma)   (discounted switch cost, stationary)
  J      = w_bar + s_term     (lower is better; approximates -(discounted return) up to a constant)
Also reports n_teleports and ended_early per run.
Play button (no arguments): runs every policy in POLICIES. Terminal: python scripts/switch_cost_probe.py 15
"""
import os, sys, json
import numpy as np
import traci
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import fixed_cycle_policy_fn
NET = os.environ.get("TLS_NET_FILE", os.path.join(ROOT, "sumo_4x4_network", "grid4x4.net.xml"))
ROUTE = os.environ.get("TLS_ROUTE_FILE", os.path.join(ROOT, "sumo_4x4_network", "routes.rou.xml"))
GAMMA, LAM = 0.99, 0.3
POLICIES = ["10", "15", "20", "30", "forced"]


def probe(env, cyc):
    n = env.n_agents
    obs, _ = env.reset(seed=1000)
    pol = (lambda o: np.zeros(n, int)) if cyc == "forced" else fixed_cycle_policy_fn(float(cyc), env.cfg.max_green)
    wsum = qsum = changes = steps = 0
    while True:
        obs, r, te, tr, info = env.step(pol(obs)); steps += 1
        wsum += info["total_waiting_time"]; qsum += info["total_queue_length"]
        changes += int(info["honored_switches"].sum() + info["forced_switches"].sum())
        if te or tr: break
    w_bar = wsum / steps / n; rate = changes / n / steps; s_term = LAM * rate / (1 - GAMMA)
    injected = open(ROUTE).read().count("<vehicle ")
    remaining = traci.simulation.getMinExpectedNumber()
    print(json.dumps(dict(policy=cyc, route=os.path.basename(ROUTE), injected=injected, w_bar=round(w_bar, 2), grid_wait=round(wsum / steps, 1),
          mean_queue=round(qsum / steps, 2), phase_changes=changes, switch_rate_per_junction_step=round(rate, 4), s_term=round(s_term, 2),
          J=round(w_bar + s_term, 2), throughput=info["throughput"], remaining_at_end=remaining, mean_vehicle_wait=round(info["mean_vehicle_wait"], 2),
          max_vehicle_wait=round(info["max_vehicle_wait"], 1), n_teleports=info["n_teleports"], ended_early=info["ended_early"], steps=steps)), flush=True)


if __name__ == "__main__":
    env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE))
    for c in ([sys.argv[1]] if len(sys.argv) > 1 else POLICIES):
        probe(env, c)
    env.close()
