"""Step 1 verification: forced-switch counting via independent paths. Read-only w.r.t. repo code."""
import os, sys, numpy as np, traci
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.grid_env import TrafficGridEnv, GridEnvConfig

NET = os.environ.get("TLS_NET_FILE", "sumo_4x4_network/grid4x4.net.xml")
ROUTE = os.environ.get("TLS_ROUTE_FILE", "sumo_4x4_network/routes.rou.xml")


class Probe(TrafficGridEnv):
    def __init__(s, cfg):
        super().__init__(cfg); s.run = None; s.greens_ge_max = 0
    def reset(s, **kw):
        out = super().reset(**kw)
        s.run = np.zeros(s.n_agents, int); s.greens_ge_max = 0
        return out
    def _advance_yellow_phases(s):                 # called every sim second
        for i, tid in enumerate(s.tl_ids):
            y = "y" in traci.trafficlight.getRedYellowGreenState(tid).lower()
            if y:
                if s.run[i] >= s.cfg.max_green - s.cfg.decision_interval + 1: s.greens_ge_max += 1
                s.run[i] = 0
            else:
                s.run[i] += 1
        super()._advance_yellow_phases()

def episode(policy, seed, name):
    cfg = GridEnvConfig(net_file=NET, route_file=ROUTE)
    env = Probe(cfg); obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    pathA = pathB = 0; prev_yellow = np.zeros(env.n_agents, bool)
    while True:
        votes = policy(obs, rng)
        # path B: independent re-derivation from pre-step state
        for i, tid in enumerate(env.tl_ids):
            y = "y" in traci.trafficlight.getRedYellowGreenState(tid).lower()
            if (not y) and env._time_in_phase[i] + cfg.decision_interval > cfg.max_green and votes[i] != 1:
                pathB += 1
        obs, r, te, tr, info = env.step(votes)
        pathA += int(info["forced_switches"].sum())     # what run_episode does
        if te or tr: break
    print(f"{name}: path A (sum info['forced_switches'])={pathA}  path B (pre-step re-derivation)={pathB}  "
          f"path C (green spells >=90s seen at 1s resolution)={env.greens_ge_max}  steps={env._episode_step}", flush=True)
    env.close()

episode(lambda o, rng: np.zeros(16, int), 1000, "hold-forever")
episode(lambda o, rng: ((rng.random(16) < 0.08) & (o["action_mask"] > 0.5)).astype(int), 1000, "sparse-vote(8%)")
