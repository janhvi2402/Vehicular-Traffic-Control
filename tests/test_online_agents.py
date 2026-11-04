"""
tests/test_online_agents.py
Tests of the AC and CA agents and of the shared online trainer (needs torch and SUMO).

    python tests/test_online_agents.py        # plain runner, prints PASS/FAIL per test
    pytest -q tests/test_online_agents.py

Env files default to sumo_4x4_network/{grid4x4.net.xml,routes.rou.xml}; override with TLS_NET_FILE / TLS_ROUTE_FILE.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from environment.eligibility import eligibility_from_features  # noqa: E402
from environment.online_trainer import train_online, step_size  # noqa: E402
from ac.config import ACConfig  # noqa: E402
from ac.agent import ACAgent, assign_exponents  # noqa: E402
from ac.network import ActorNetwork  # noqa: E402
from ca.config import CAConfig  # noqa: E402
from ca.agent import CAAgent  # noqa: E402

NET = os.environ.get("TLS_NET_FILE", os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml"))
ROUTE = os.environ.get("TLS_ROUTE_FILE", os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml"))
MIN_G, MAX_G = GridEnvConfig.__dataclass_fields__["min_green"].default, GridEnvConfig.__dataclass_fields__["max_green"].default
FIXED = ["env_step", "episode", "episode_reward", "total_waiting_time", "throughput", "mean_waiting_time_ep",
         "mean_vehicle_wait_ep", "eligible_fraction"]


def tiny(cfg_cls, steps=60, **kw):
    cfg = cfg_cls(**kw)
    cfg.total_env_steps, cfg.episode_seconds, cfg.checkpoint_every, cfg.seed = steps, 100, 20, 2   # 20-step episodes
    return cfg


def synthetic_obs(n=16, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.random((n, 11)).astype(np.float32)
    f[:, 9] = (np.arange(n) % 4 == 0).astype(np.float32)                 # some junctions in yellow
    f[:, 10] = np.where(np.arange(n) % 3 == 0, 0.05, 0.5)               # some below min_green (10/90 = 0.11)
    return f, eligibility_from_features(f, MIN_G, MAX_G)


# 1. masked actor: ineligible rows hold, log-prob 0, entropy 0 (sampling and greedy)
def test_actor_masks_ineligible_rows():
    torch.manual_seed(0)
    actor = ActorNetwork(11, (128, 128), 2)
    with torch.no_grad():
        actor.net[-1].bias.copy_(torch.tensor([0.0, 30.0]))              # strongly prefers switch
    f, el = synthetic_obs()
    x, elt = torch.as_tensor(f), torch.as_tensor(el)
    assert (~el).sum() > 0 and el.sum() > 0
    for _ in range(50):
        a, lp, ent = actor.act(x, elt)
        assert (a[~elt] == 0).all() and (lp[~elt] == 0).all() and (ent[~elt] == 0).all()
        assert (ent[elt] >= 0).all()
    g = actor.act_greedy(x, elt)
    assert (g[~elt] == 0).all() and (g[elt] == 1).all()
    assert (torch.softmax(actor.masked_logits(x, elt), -1)[~elt, 1] == 0).all()


# 2. both agents return only executable actions and take finite, non-trivial learning steps
def test_agents_act_and_update_finite():
    for cls, cfg_cls in ((ACAgent, ACConfig), (CAAgent, CAConfig)):
        torch.manual_seed(0)
        agent = cls(cfg_cls(), "cpu")
        f, el = synthetic_obs()
        w0 = [p.clone() for p in agent.actor.parameters()]
        c0 = [p.clone() for p in agent.critic.parameters()]
        for step in range(1, 6):
            a = agent.act(f, el)
            assert a.dtype == np.int64 and (a[~el] == 0).all()
            m = agent.update(f, np.random.default_rng(step).normal(size=16).astype(np.float32), False, step)
            assert all(np.isfinite(v) for v in m.values()), (cls.__name__, m)
            assert set(m) == set(agent.metric_names)
        assert any(not torch.equal(p, q) for p, q in zip(agent.actor.parameters(), w0)), "actor did not change"
        assert any(not torch.equal(p, q) for p, q in zip(agent.critic.parameters(), c0)), "critic did not change"


# 3. timescale assignment: smaller exponent = faster recursion
def test_timescale_assignment():
    c, a = assign_exponents(ACConfig(variant="ac"))
    assert c == 0.55 and a == 0.75 and c < a                              # AC: critic fast
    c, a = assign_exponents(ACConfig(variant="ca"))
    assert c == 0.75 and a == 0.55 and a < c                              # discounted CA ablation: actor fast
    cfg = CAConfig()
    assert cfg.actor_exponent < cfg.critic_exponent                       # CA: actor (and L) fast, critic slow
    try:
        CAConfig(actor_exponent=0.6, critic_exponent=0.5); raise AssertionError("nu >= sigma must be refused")
    except ValueError:
        pass
    assert step_size(0, 3e-4, 0.5, 100) == 3e-4 and abs(step_size(300, 3e-4, 0.5, 100) - 3e-4 / 2.0) < 1e-12


# 4. AC TD error: r + gamma V(s') - V(s), bootstrap zeroed ONLY when terminated
def test_ac_td_error_and_termination():
    torch.manual_seed(0)
    cfg = ACConfig()
    f, el = synthetic_obs(); f2, _ = synthetic_obs(seed=1)
    r = np.random.default_rng(0).normal(size=16).astype(np.float32)
    for terminated in (False, True):
        torch.manual_seed(0)
        ag = ACAgent(cfg, "cpu"); ag.act(f, el)
        with torch.no_grad():
            v, vn = ag.critic(torch.as_tensor(f)), ag.critic(torch.as_tensor(f2))
        exp = torch.as_tensor(r) + cfg.gamma * (0 * vn if terminated else vn) - v
        m = ag.update(f2, r, terminated, 1)
        assert abs(m["abs_td_error"] - exp.abs().mean().item()) < 1e-5, (terminated, m["abs_td_error"], exp.abs().mean().item())


# 5. CA: TD error uses L_t (before its update); then L <- L + gamma_t (r - L), gamma_t = K * alpha_t; no discount
def test_ca_average_reward_recursion():
    torch.manual_seed(0)
    cfg = CAConfig()
    ag = CAAgent(cfg, "cpu")
    f, el = synthetic_obs(); f2, _ = synthetic_obs(seed=1)
    r = np.random.default_rng(0).normal(size=16).astype(np.float32)
    ag.act(f, el)
    L0 = np.linspace(-1, 1, 16).astype(np.float32); ag.avg_reward = L0.copy()
    with torch.no_grad():
        v, vn = ag.critic(torch.as_tensor(f)), ag.critic(torch.as_tensor(f2))
    exp_td = torch.as_tensor(r) - torch.as_tensor(L0) + vn - v
    m = ag.update(f2, r, False, 1)
    assert abs(m["abs_td_error"] - exp_td.abs().mean().item()) < 1e-5
    g = cfg.avg_reward_k * step_size(1, cfg.actor_lr0, cfg.actor_exponent, ag.decay_every)
    assert np.allclose(ag.avg_reward, L0 + g * (r - L0), atol=1e-7)
    assert not hasattr(cfg, "gamma")


# 6. real SUMO, 20-step episodes: seed scheme, log columns, executed == chosen (asserted by the trainer), files, resume
def test_trainer_runs_both_agents_with_seed_scheme_and_resume():
    for name, cls, cfg_cls in (("ac", ACAgent, ACConfig), ("ca", CAAgent, CAConfig)):
        seeds = []
        orig = TrafficGridEnv.reset

        def reset(self, *a, **kw):
            seeds.append(kw.get("seed")); return orig(self, *a, **kw)
        TrafficGridEnv.reset = reset
        try:
            with tempfile.TemporaryDirectory() as d:
                cfg = tiny(cfg_cls)
                train_online(cls, cfg, NET, ROUTE, d, algo=name, chunk_steps=40)
                train_online(cls, cfg, NET, ROUTE, d, algo=name, resume_from=os.path.join(d, "state_latest.pt"))
                rows = list(csv.reader(open(os.path.join(d, "train_log.csv"))))
                assert rows[0][:len(FIXED)] == FIXED and rows[0][len(FIXED):] == cls.metric_names, rows[0]
                assert [int(r[1]) for r in rows[1:]] == [0, 1, 2] and [int(r[0]) for r in rows[1:]] == [20, 40, 60]
                for need in ("config.json", "actor_final.pt", "critic_final.pt", "resume_state.json"):
                    assert os.path.exists(os.path.join(d, need)), need
                assert all(np.isfinite(float(x)) for r in rows[1:] for x in r)
                assert seeds == [20000, 20001, 20002], seeds
                # the saved actor loads into the evaluation policy and never switches an ineligible junction
                actor = ActorNetwork(11, (128, 128), 2); actor.load_state_dict(torch.load(os.path.join(d, "actor_final.pt")))
                f, el = synthetic_obs()
                assert (actor.act_greedy(torch.as_tensor(f), torch.as_tensor(el))[torch.as_tensor(~el)] == 0).all()
        finally:
            TrafficGridEnv.reset = orig


TESTS = [test_actor_masks_ineligible_rows, test_agents_act_and_update_finite, test_timescale_assignment,
         test_ac_td_error_and_termination, test_ca_average_reward_recursion,
         test_trainer_runs_both_agents_with_seed_scheme_and_resume]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL  {t.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
