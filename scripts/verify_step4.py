"""Step 4 verification: reward = (w(t-1)-w(t)) - lambda_s*[phase change executed] - lambda_w*[wasted vote]."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from environment.grid_env import TrafficGridEnv, GridEnvConfig

NET = os.environ.get("TLS_NET_FILE", os.path.join(_ROOT, "sumo_4x4_network", "grid4x4.net.xml"))
ROUTE = os.environ.get("TLS_ROUTE_FILE", os.path.join(_ROOT, "sumo_4x4_network", "routes.rou.xml"))
SECS = int(os.environ.get("VERIFY_SECS", 1200))


def run(policy, **kw):
    env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE, episode_seconds=SECS, **kw))
    obs, _ = env.reset(seed=1000); rng = np.random.default_rng(0)
    rows = []; prev_w = env._prev_wait.copy()
    while True:
        a = policy(obs, rng); obs, r, te, tr, info = env.step(a)
        rows.append(dict(ar=info["agent_rewards"].copy(), forced=info["forced_switches"].copy(),
                         honored=info["honored_switches"].copy(), q=info["total_queue_length"], w=info["total_waiting_time"],
                         wj=env._prev_wait.copy(), qj=env._per_junction_queue().copy(), prev_w=prev_w.copy(), r=r))
        prev_w = env._prev_wait.copy()
        if te or tr: break
    cfg = env.cfg; env.close(); return rows, cfg


hold = lambda o, rng: np.zeros(16, int)
sparse = lambda o, rng: ((rng.random(16) < 0.15) & (o["action_mask"] > 0.5)).astype(int)

print("default reward_mode:", GridEnvConfig(NET, ROUTE).reward_mode, "| lambda_s:", GridEnvConfig(NET, ROUTE).lambda_s,
      "| lambda_w:", GridEnvConfig(NET, ROUTE).lambda_w)
try:
    GridEnvConfig(NET, ROUTE, reward_mode="bogus"); print("FAIL: bogus mode accepted")
except ValueError as e:
    print("bad reward_mode rejected: OK")

for name, pol in [("hold-forever (all switches forced)", hold), ("sparse votes (mixed voted+forced)", sparse)]:
    A, c = run(pol, lambda_s=0.3); B, _ = run(pol, lambda_s=0.0)
    n_change = sum(int((x["forced"] | x["honored"].astype(bool)).sum()) for x in A)
    n_forced = sum(int(x["forced"].sum()) for x in A); n_hon = sum(int(x["honored"].sum()) for x in A)
    diff = sum(float(b["ar"].sum() - a["ar"].sum()) for a, b in zip(A, B))
    per_step = max(float(np.abs((b["ar"] - a["ar"]) - 0.3 * (a["forced"] | a["honored"].astype(bool))).max()) for a, b in zip(A, B))
    base = max(float(np.abs(b["ar"] - (b["prev_w"] - b["wj"])).max()) for b in B)
    print(f"\n[{name}] phase changes={n_change} (voted={n_hon}, forced={n_forced})")
    print(f"  total reward drop from lambda_s 0 -> 0.3 = {diff:.3f}  expected 0.3*{n_change} = {0.3 * n_change:.3f}")
    print(f"  max per-step/per-junction |penalty - 0.3*[phase change]| = {per_step:.2e}  (forced switches ARE penalized)")
    print(f"  lambda_s=0: max |r_i - (w_i(t-1)-w_i(t))| = {base:.2e}")
    assert abs(diff - 0.3 * n_change) < 1e-2 and per_step < 1e-3 and base < 1e-3  # float32 reward arrays

C, c = run(sparse, reward_mode="queue_wait_absolute")
err = max(float(np.abs(x["ar"] - (-0.01 * (0.5 * x["qj"] + 0.5 * x["wj"]) - 0.3 * (x["forced"] | x["honored"].astype(bool)))).max()) for x in C)
print(f"\n[queue_wait_absolute] max |r_i - formula| = {err:.2e}; queue still logged: mean total_queue_length = {np.mean([x['q'] for x in C]):.2f}")
D, _ = run(sparse)
print(f"[delta_wait default]   queue logged: mean total_queue_length = {np.mean([x['q'] for x in D]):.2f}")
assert err < 1e-3
print("\nALL STEP 4 CHECKS PASSED")
