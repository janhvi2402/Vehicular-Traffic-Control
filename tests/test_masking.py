"""
tests/test_masking.py
Eligibility-masking tests for DQN and PPO. Needs torch and SUMO (real TraCI episodes).

    python tests/test_masking.py          # plain runner, prints PASS/FAIL per test
    pytest -q tests/test_masking.py       # same tests under pytest

Env files default to sumo_4x4_network/{grid4x4.net.xml,routes.rou.xml}; override with
TLS_NET_FILE / TLS_ROUTE_FILE. Each test that trains uses a temp directory and a few dozen steps.
"""
from __future__ import annotations

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
import dqn.train as dqn_train  # noqa: E402
import dqn.q_network as qn  # noqa: E402
import ppo.train as ppo_train  # noqa: E402
import ppo.actor_critic as ac  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from ppo.config import PPOConfig  # noqa: E402

NET = os.environ.get("TLS_NET_FILE", os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml"))
ROUTE = os.environ.get("TLS_ROUTE_FILE", os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml"))
MIN_G, MAX_G = GridEnvConfig.__dataclass_fields__["min_green"].default, GridEnvConfig.__dataclass_fields__["max_green"].default


def make_env():
    return TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE))


def tiny_dqn_cfg(steps):
    cfg = DQNConfig()
    cfg.total_env_steps = steps
    cfg.epsilon_decay_fraction = 1.0       # epsilon anneals 1.0 -> 0.1 across the run: mostly exploration
    cfg.replay_start_size = 64
    cfg.batch_size = 32
    cfg.log_every = 10 ** 9
    cfg.checkpoint_every = 10 ** 9
    return cfg


# --------------------------------------------------------------------------- #
# 1. eligibility from features == action_mask, tolerance min_green - 1e-4
# --------------------------------------------------------------------------- #
def test_eligibility_matches_action_mask():
    # (a) synthetic: every integer elapsed second, yellow on/off, float32 exactly as the env builds it
    for yellow in (0.0, 1.0):
        for e in range(0, int(MAX_G) + 1):
            f = np.zeros((1, 11), dtype=np.float32)
            f[0, 9] = yellow
            f[0, 10] = np.clip(np.float32(e) / np.float32(MAX_G), 0.0, 1.0)
            expected = (yellow == 0.0) and (e >= MIN_G)
            assert bool(eligibility_from_features(f, MIN_G, MAX_G)[0]) == expected, (yellow, e)
    # (b) tolerance edge: 1e-4 below min_green passes, 1e-3 below does not
    f = np.zeros((2, 11), dtype=np.float32)
    f[:, 10] = np.array([(MIN_G - 0.5e-4) / MAX_G, (MIN_G - 1e-3) / MAX_G], dtype=np.float32)
    assert eligibility_from_features(f, MIN_G, MAX_G).tolist() == [True, False]
    # (c) real env: equals obs["action_mask"] at every step, every junction
    env = make_env(); rng = np.random.default_rng(0)
    try:
        obs, _ = env.reset(seed=1000)
        n_elig = n_inelig = 0
        for _ in range(120):
            el = eligibility_from_features(obs["node_features"], MIN_G, MAX_G)
            assert np.array_equal(el, obs["action_mask"] > 0.5)
            n_elig += int(el.sum()); n_inelig += int((~el).sum())
            obs, *_ = env.step(rng.integers(0, 2, env.n_agents))
        assert n_elig > 0 and n_inelig > 0, "test would be vacuous"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 2. random-weight networks, 20 steps: ineligible rows always executed 0 (incl. exploration)
# --------------------------------------------------------------------------- #
def test_ineligible_rows_always_hold_dqn_and_ppo():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    q = qn.QNetwork(11, (128, 128), 2)
    pol = ac.ActorCritic(11, (128, 128), 2)
    env = make_env()
    try:
        obs, _ = env.reset(seed=1000)
        n_inelig = n_exec_switch = 0
        for t in range(20):
            x = obs["node_features"]
            el = eligibility_from_features(x, MIN_G, MAX_G)
            xt, elt = torch.as_tensor(x), torch.as_tensor(el)
            for eps in (1.0, 0.5, 0.0):                       # pure exploration, mixed, greedy
                a = qn.select_actions_epsilon_greedy(q, x, eps, rng, "cpu", el)
                assert (a[~el] == 0).all(), f"DQN eps={eps} t={t}"
            a_dqn = qn.select_actions_epsilon_greedy(q, x, 1.0, rng, "cpu", el)
            a_ppo_s, _, _ = pol.act(xt, elt)
            a_ppo_g = pol.act_greedy(xt, elt)
            assert (a_ppo_s.numpy()[~el] == 0).all() and (a_ppo_g.numpy()[~el] == 0).all(), f"PPO t={t}"
            # the env must execute exactly what was chosen (alternate which policy drives the env)
            chosen = a_dqn if t % 2 == 0 else a_ppo_s.numpy()
            obs, _, _, _, info = env.step(chosen)
            assert np.array_equal(info["executed_actions"], chosen)
            n_inelig += int((~el).sum()); n_exec_switch += int(info["executed_actions"].sum())
        assert n_inelig > 0 and n_exec_switch > 0, "test would be vacuous (no ineligible rows or no switches)"
        # control: a raw all-ones VOTE on the first step is NOT what gets executed
        obs, _ = env.reset(seed=1000)
        _, _, _, _, info = env.step(np.ones(env.n_agents, dtype=int))
        assert info["executed_actions"].sum() == 0
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 3. replay stores the executed action; DQN train.py passes the right eligibility to the target
# --------------------------------------------------------------------------- #
def test_replay_stores_executed_action_and_target_mask_wired():
    steps = 60
    pushed, executed, target_calls = [], [], []
    orig_push, orig_step, orig_targets = qn.ReplayBuffer.push, TrafficGridEnv.step, dqn_train.dqn_targets

    def push(self, obs, action, reward, next_obs, done):
        pushed.append((np.array(obs), int(action))); return orig_push(self, obs, action, reward, next_obs, done)

    def step(self, a):
        out = orig_step(self, a); executed.append(out[4]["executed_actions"].copy()); return out

    def targets(target_net, rewards, next_obs, dones, gamma, next_eligible):
        target_calls.append((next_obs.numpy().copy(), next_eligible.numpy().copy()))
        return orig_targets(target_net, rewards, next_obs, dones, gamma, next_eligible)

    qn.ReplayBuffer.push, TrafficGridEnv.step, dqn_train.dqn_targets = push, step, targets
    try:
        with tempfile.TemporaryDirectory() as d:
            dqn_train.train(tiny_dqn_cfg(steps), NET, ROUTE, d)
    finally:
        qn.ReplayBuffer.push, TrafficGridEnv.step, dqn_train.dqn_targets = orig_push, orig_step, orig_targets
    n_agents = len(executed[0])
    assert len(pushed) == steps * n_agents == 16 * steps
    stored = np.array([a for _, a in pushed]).reshape(steps, n_agents)
    assert np.array_equal(stored, np.array(executed)), "replay holds something other than the executed action"
    el = eligibility_from_features(np.array([o for o, _ in pushed]), MIN_G, MAX_G)
    assert (~el).sum() > 0 and (np.array([a for _, a in pushed])[~el] == 0).all()
    assert stored.sum() > 0, "vacuous: no switch was ever executed"
    assert len(target_calls) > 0, "no gradient step ran"
    for nxt, ne in target_calls:
        assert np.array_equal(ne, eligibility_from_features(nxt, MIN_G, MAX_G))


# --------------------------------------------------------------------------- #
# 4. DQN target for an ineligible next-state uses only the hold Q-value
# --------------------------------------------------------------------------- #
def test_dqn_target_ineligible_uses_hold_only():
    torch.manual_seed(0)
    tgt = qn.QNetwork(11, (128, 128), 2)
    with torch.no_grad():
        tgt.net[-1].bias.copy_(torch.tensor([0.0, 100.0]))        # Q(switch) >> Q(hold) everywhere
    n = 16
    next_obs = torch.rand(n, 11)
    elig = torch.tensor([i % 2 == 0 for i in range(n)])
    r, d, gamma = torch.randn(n), torch.zeros(n), 0.99
    d[3] = 1.0                                                     # one terminal row
    nq = tgt(next_obs)
    out = qn.dqn_targets(tgt, r, next_obs, d, gamma, elig)
    assert (nq[~elig].max(dim=1).values != nq[~elig, 0]).all(), "vacuous: switch Q not larger"
    exp_hold = r + gamma * nq[:, 0] * (1 - d)
    exp_max = r + gamma * nq.max(dim=1).values * (1 - d)
    assert torch.allclose(out[~elig], exp_hold[~elig], atol=1e-6)   # hold Q only
    assert torch.allclose(out[elig], exp_max[elig], atol=1e-6)      # eligible: normal max
    assert torch.allclose(out[3], r[3])                             # done zeroes the bootstrap


# --------------------------------------------------------------------------- #
# 5. PPO: P(switch)==0 exactly when ineligible, in act() AND in the update-loop forward pass
# --------------------------------------------------------------------------- #
def test_ppo_switch_probability_zero_when_ineligible():
    torch.manual_seed(0)
    # --- act(): spy on the mask function act() calls, with a policy that strongly prefers switch
    pol = ac.ActorCritic(11, (128, 128), 2)
    with torch.no_grad():
        pol.policy_head.bias.copy_(torch.tensor([0.0, 30.0]))
    x = torch.rand(64, 11); el = torch.arange(64) % 2 == 0
    seen = []; orig = ac.mask_switch_logits
    ac.mask_switch_logits = lambda l, e: (seen.append(orig(l, e)) or seen[-1])
    try:
        for _ in range(50):
            a, lp, _v = pol.act(x, el)
            assert (a[~el] == 0).all() and (lp[~el] == 0).all()
        g = pol.act_greedy(x, el)
        assert (g[~el] == 0).all() and (g[el] == 1).all()
    finally:
        ac.mask_switch_logits = orig
    assert len(seen) == 51
    for m in seen:
        p = torch.softmax(m, dim=-1)
        assert (p[~el, 1] == 0.0).all() and (p[el, 1] > 0).all()

    # --- update loop: run a real (tiny) PPO update and spy on ppo.train's forward-pass masking
    cfg = PPOConfig(); cfg.horizon = 40; cfg.total_updates = 1; cfg.num_epochs = 1; cfg.minibatch_size = 64
    calls, stored = [], []
    orig_train_mask, orig_add = ppo_train.mask_switch_logits, ac.RolloutBuffer.add

    def spy_mask(logits, eligible):
        out = orig_train_mask(logits, eligible); calls.append((out.detach().clone(), eligible.clone())); return out

    def spy_add(self, obs, actions, log_probs, values, rewards, done, eligible):
        stored.append((np.array(obs), np.array(actions), np.array(eligible)))
        return orig_add(self, obs, actions, log_probs, values, rewards, done, eligible)

    ppo_train.mask_switch_logits, ac.RolloutBuffer.add = spy_mask, spy_add
    try:
        with tempfile.TemporaryDirectory() as d:
            ppo_train.train(cfg, NET, ROUTE, d)
    finally:
        ppo_train.mask_switch_logits, ac.RolloutBuffer.add = orig_train_mask, orig_add
    assert len(calls) >= 10, "update loop never called the mask"
    n_inelig = 0
    for out, e in calls:
        p = torch.softmax(out, dim=-1)
        assert (p[~e, 1] == 0.0).all()
        n_inelig += int((~e).sum())
    assert n_inelig > 0
    # rollout buffer: eligibility stored == derived from the stored obs; stored action is the executed one
    for o, a, e in stored:
        assert np.array_equal(e, eligibility_from_features(o, MIN_G, MAX_G)) and (a[~e] == 0).all()


# --------------------------------------------------------------------------- #
# 6. PPO is reproducible from cfg.seed (minibatch shuffling is seeded)
# --------------------------------------------------------------------------- #
def test_ppo_reproducible_from_seed():
    def run(seed):
        cfg = PPOConfig(); cfg.horizon = 60; cfg.total_updates = 2; cfg.num_epochs = 2; cfg.seed = seed
        with tempfile.TemporaryDirectory() as d:
            ppo_train.train(cfg, NET, ROUTE, d)
            return torch.load(os.path.join(d, "actor_critic_final.pt"))
    a, b, c = run(0), run(0), run(1)
    assert all(torch.equal(a[k], b[k]) for k in a), "same seed gave different weights"
    assert any(not torch.equal(a[k], c[k]) for k in a), "different seeds gave identical weights"


# --------------------------------------------------------------------------- #
# 7. training env seeds: scheme, disjointness, and actually used by both trainers; new log columns
# --------------------------------------------------------------------------- #
def _spy_resets_and_steps():
    seeds, waits, vwait = [], [], []
    orig_reset, orig_step = TrafficGridEnv.reset, TrafficGridEnv.step

    def reset(self, *a, **kw):
        seeds.append(kw.get("seed")); return orig_reset(self, *a, **kw)

    def step(self, act):
        out = orig_step(self, act); waits.append(out[4]["total_waiting_time"]); vwait.append(out[4]["mean_vehicle_wait"]); return out

    TrafficGridEnv.reset, TrafficGridEnv.step = reset, step
    return seeds, waits, vwait, (orig_reset, orig_step)


def test_training_seeds_disjoint():
    from environment.seeds import training_env_seed, RESERVED
    seen = {}
    for ts in range(5):
        for ep in range(500):
            v = training_env_seed(ts, ep)
            assert v not in seen, f"seed {v} shared by train seeds {seen[v]} and {ts}"
            assert not (RESERVED[0] <= v < RESERVED[1]), v        # eval 1000-1999, sweep 2000-2999
            seen[v] = ts
    assert len(seen) == 2500
    assert not (set(range(1000, 1010)) | set(range(2000, 2005))) & set(seen)
    try:
        training_env_seed(0, 1000); raise AssertionError("seed 0, episode 1000 should be refused")
    except ValueError:
        pass


def test_trainers_use_seed_scheme_and_log_new_columns():
    import csv
    OLD_DQN = ["env_step", "episode", "epsilon", "episode_reward", "total_waiting_time", "throughput", "loss"]
    OLD_PPO = ["update", "env_step", "episode", "episode_reward", "total_waiting_time", "throughput",
               "policy_loss", "value_loss", "entropy"]
    NEW = ["mean_waiting_time_ep", "mean_vehicle_wait_ep"]
    # ---- DQN: 20-step episodes (episode_seconds=100), log every episode
    cfg = tiny_dqn_cfg(60); cfg.episode_seconds = 100; cfg.log_every = 20; cfg.seed = 2
    seeds, waits, vwait, orig = _spy_resets_and_steps()
    try:
        with tempfile.TemporaryDirectory() as d:
            dqn_train.train(cfg, NET, ROUTE, d)
            rows = list(csv.reader(open(os.path.join(d, "train_log.csv"))))
    finally:
        TrafficGridEnv.reset, TrafficGridEnv.step = orig
    assert seeds == [20000 + k for k in range(len(seeds))] and len(seeds) >= 3, seeds
    assert rows[0] == OLD_DQN + NEW, rows[0]
    assert len(rows) == 4
    for k, r in enumerate(rows[1:]):
        assert abs(float(r[7]) - np.mean(waits[20 * k:20 * k + 20])) < 1e-6
        assert abs(float(r[8]) - vwait[20 * k + 19]) < 1e-6
    # ---- PPO: horizon == episode (20 steps)
    cfg = PPOConfig(); cfg.horizon = 20; cfg.episode_seconds = 100; cfg.total_updates = 3; cfg.num_epochs = 1; cfg.seed = 2
    seeds, waits, vwait, orig = _spy_resets_and_steps()
    try:
        with tempfile.TemporaryDirectory() as d:
            ppo_train.train(cfg, NET, ROUTE, d)
            rows = list(csv.reader(open(os.path.join(d, "train_log.csv"))))
    finally:
        TrafficGridEnv.reset, TrafficGridEnv.step = orig
    assert seeds == [20000 + k for k in range(len(seeds))] and len(seeds) >= 3, seeds
    assert rows[0] == OLD_PPO + NEW, rows[0]
    for k, r in enumerate(rows[1:]):
        assert abs(float(r[9]) - np.mean(waits[20 * k:20 * k + 20])) < 1e-6
        assert abs(float(r[10]) - vwait[20 * k + 19]) < 1e-6


def test_run_episode_health_and_green_metrics():
    from environment.eval_common import run_episode, fixed_cycle_policy_fn
    env = TrafficGridEnv(GridEnvConfig(net_file=NET, route_file=ROUTE, episode_seconds=400))
    try:
        row, greens = run_episode(env, fixed_cycle_policy_fn(10, env.cfg.max_green), 1000, return_events=True)
        for k in ("n_teleports", "ended_early", "n_greens", "share_greens_at_max_green"):
            assert k in row, k
        assert row["ended_early"] in (0.0, 1.0) and row["n_teleports"] >= 0
        assert row["n_greens"] == len(greens) == row["n_switch_votes"] + row["n_forced_switches"] > 0
        durs = np.array([g["end"] - g["start"] for g in greens])
        assert row["share_greens_at_max_green"] == sum(g["ended_by_forced"] for g in greens) / len(greens)
        assert durs.min() >= env.cfg.min_green and durs.max() <= env.cfg.max_green
    finally:
        env.close()


def test_max_pressure_policy_logic():
    from environment.eval_common import max_pressure_policy_fn
    f = np.zeros((4, 11), dtype=np.float32)
    f[:, 8] = [0.0, 0.0, 2 / 3, 2 / 3]                           # phase 0 (E+W green) x2, phase 2 (N+S green) x2
    f[0, 0:4] = np.array([5, 1, 5, 1]) / 20                      # N,E,S,W: waiting N+S=10 > served E+W=2  -> switch
    f[1, 0:4] = np.array([1, 5, 1, 5]) / 20                      # waiting 2 < served 10                   -> hold
    f[2, 0:4] = np.array([1, 5, 1, 5]) / 20                      # phase 2 serves N+S=2, waiting E+W=10    -> switch
    f[3, 0:4] = np.array([5, 1, 5, 1]) / 20                      # served N+S=10 > waiting 2               -> hold
    assert max_pressure_policy_fn(20.0, 0.0)({"node_features": f}).tolist() == [1, 0, 1, 0]
    assert max_pressure_policy_fn(20.0, 7.0)({"node_features": f}).tolist() == [1, 0, 1, 0]   # 10 > 2 + 7
    assert max_pressure_policy_fn(20.0, 8.0)({"node_features": f}).tolist() == [0, 0, 0, 0]   # 10 > 2 + 8 is false (strict)


TESTS = [test_eligibility_matches_action_mask, test_ineligible_rows_always_hold_dqn_and_ppo,
         test_replay_stores_executed_action_and_target_mask_wired, test_dqn_target_ineligible_uses_hold_only,
         test_ppo_switch_probability_zero_when_ineligible,
         test_ppo_reproducible_from_seed, test_training_seeds_disjoint,
         test_trainers_use_seed_scheme_and_log_new_columns, test_run_episode_health_and_green_metrics,
         test_max_pressure_policy_logic]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL  {t.__name__}"); traceback.print_exc()
    sys.exit(1 if failed else 0)
