"""Two identical tiny PPO runs (same --seed) must give bit-identical weights; a different seed must not."""
import os, sys, tempfile, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ppo.train as T
from ppo.config import PPOConfig
NET = os.environ.get("TLS_NET_FILE", "sumo_4x4_network/grid4x4.net.xml"); ROUTE = os.environ.get("TLS_ROUTE_FILE", "sumo_4x4_network/routes.rou.xml")
def run(seed):
    cfg = PPOConfig(); cfg.horizon = 60; cfg.total_updates = 2; cfg.num_epochs = 2; cfg.seed = seed
    with tempfile.TemporaryDirectory() as d:
        T.train(cfg, NET, ROUTE, d); return torch.load(os.path.join(d, "actor_critic_final.pt"))
a, b, c = run(0), run(0), run(1)
same = all(torch.equal(a[k], b[k]) for k in a); diff = any(not torch.equal(a[k], c[k]) for k in a)
print(f"seed0 vs seed0 identical: {same} | seed0 vs seed1 differ: {diff}")
assert same and diff
print("PPO reproducibility OK")
