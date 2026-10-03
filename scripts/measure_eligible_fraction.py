"""Fraction of (step, junction) rows that are eligible to switch, per policy (1200 s episodes, seed 1000).
Needed for docs/LIMITATIONS.md: PPO's surrogate and entropy terms average over ALL rows, ineligible rows contribute 0."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import fixed_cycle_policy_fn
NET = os.environ.get("TLS_NET_FILE", "sumo_4x4_network/grid4x4.net.xml"); ROUTE = os.environ.get("TLS_ROUTE_FILE", "sumo_4x4_network/routes.rou.xml")
env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE, episode_seconds=1200)); res = {}
pols = {"always_switch(10s)": fixed_cycle_policy_fn(10, 90), "fixed_cycle_15s": fixed_cycle_policy_fn(15, 90),
        "fixed_cycle_30s": fixed_cycle_policy_fn(30, 90), "hold_forever": lambda o: np.zeros(16, int)}
if len(sys.argv) > 1:   # optional PPO checkpoint: stochastic (training-time) and greedy
    import torch
    from ppo.config import PPOConfig; from ppo.actor_critic import ActorCritic
    from environment.eligibility import eligibility_from_features
    c = PPOConfig(); net = ActorCritic(c.input_dim, c.hidden_sizes, c.n_actions); net.load_state_dict(torch.load(sys.argv[1])); net.eval()
    t = lambda o: (torch.as_tensor(o["node_features"], dtype=torch.float32), torch.as_tensor(eligibility_from_features(o["node_features"])))
    pols["ppo_ckpt_stochastic"] = lambda o: net.act(*t(o))[0].numpy()
    pols["ppo_ckpt_greedy"] = lambda o: net.act_greedy(*t(o)).numpy()
for name, pol in pols.items():
    obs, _ = env.reset(seed=1000); el = n = 0
    while True:
        el += int((obs["action_mask"] > 0.5).sum()); n += 16
        obs, _, te, tr, _ = env.step(pol(obs))
        if te or tr: break
    res[name] = round(el / n, 3); print(f"{name:22s} eligible fraction = {el / n:.3f}", flush=True)
env.close()
