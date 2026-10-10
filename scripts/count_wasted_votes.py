"""
scripts/count_wasted_votes.py
Runs ONE eval episode per policy and counts wasted votes = (votes the policy returned) - (votes the env
executed), where executed = info["executed_actions"]. For DQN/PPO (masked) this must be exactly 0;
random / fixed-cycle baselines emit raw votes and show how many votes were ignored.
Also prints eval_common.run_episode's n_switch_votes, which must equal the executed count.

  python scripts/count_wasted_votes.py --algo dqn --checkpoint runs/smoke_dqn/qnet_final.pt
"""
import argparse, os, sys
import numpy as np
import torch
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from environment.grid_env import TrafficGridEnv, GridEnvConfig
from environment.eval_common import random_policy_fn, fixed_cycle_policy_fn, run_episode


def count(env, policy, seed):
    obs, _ = env.reset(seed=seed); votes = executed = 0
    while True:
        a = np.asarray(policy(obs)); obs, _, te, tr, info = env.step(a)
        votes += int(a.sum()); executed += int(info["executed_actions"].sum())
        if te or tr: return votes, executed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn", choices=["dqn", "ppo"]); p.add_argument("--checkpoint", default=None)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--net_file", default=os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml"))
    p.add_argument("--route_file", default=os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml"))
    a, _ = p.parse_known_args()
    if a.checkpoint is None:   # Play button: no arguments -> seed-0 run of the chosen algo (edit --algo default above)
        a.checkpoint = os.path.join(PROJECT_ROOT, "runs", f"{a.algo}_seed0",
                                    "qnet_final.pt" if a.algo == "dqn" else "actor_critic_final.pt")
    if not os.path.exists(a.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {a.checkpoint}")
    if a.algo == "dqn":
        from dqn.config import DQNConfig; from dqn.q_network import QNetwork
        try:
            from dqn.evaluate import dqn_policy_fn; src = "dqn.evaluate.dqn_policy_fn"
        except Exception:
            from dqn.q_network import select_actions_epsilon_greedy
            dqn_policy_fn = lambda q, eps, dev, rng: (lambda o: select_actions_epsilon_greedy(q, o["node_features"], eps, rng, dev)); src = "inline copy"
        cfg = DQNConfig(); net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions)
        net.load_state_dict(torch.load(a.checkpoint, map_location="cpu")); net.eval()
        policy = dqn_policy_fn(net, 0.0, "cpu", np.random.default_rng(a.seed)); print(f"policy builder: {src} (eval_epsilon=0.0)")
    else:
        from ppo.config import PPOConfig; from ppo.actor_critic import ActorCritic
        try:
            from ppo.evaluate import ppo_policy_fn; src = "ppo.evaluate.ppo_policy_fn"
        except ImportError:
            from environment.eligibility import eligibility_from_features
            def ppo_policy_fn(net, device, greedy=True):
                return lambda o: net.act_greedy(torch.as_tensor(o["node_features"], dtype=torch.float32),
                                                torch.as_tensor(eligibility_from_features(o["node_features"]))).numpy()
            src = "inline copy"
        cfg = PPOConfig(); net = ActorCritic(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions)
        net.load_state_dict(torch.load(a.checkpoint, map_location="cpu")); net.eval()
        policy = ppo_policy_fn(net, "cpu", greedy=True); print(f"policy builder: {src} (greedy)")
    env = TrafficGridEnv(GridEnvConfig(net_file=a.net_file, route_file=a.route_file,
                                       episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval))
    print(f"{'policy':14s}{'votes':>8s}{'executed':>10s}{'wasted':>8s}   run_episode.n_switch_votes")
    # factories: every call gets a FRESH policy (the random policy's RNG must not carry over between the two runs)
    for name, make in [(a.algo, lambda: policy), ("fixed_cycle15", lambda: fixed_cycle_policy_fn(15, env.cfg.max_green)),
                       ("random", lambda: random_policy_fn(env.n_agents, np.random.default_rng(1)))]:
        v, e = count(env, make(), a.seed); r = run_episode(env, make(), a.seed)
        print(f"{name:14s}{v:8d}{e:10d}{v - e:8d}   {r['n_switch_votes']}")
        assert e == r["n_switch_votes"], f"{name}: executed votes != run_episode.n_switch_votes"
        if name == a.algo: assert v == e, "learned policy produced wasted votes"
    env.close(); print("OK: learned policy wasted votes == 0")


if __name__ == "__main__":
    main()
