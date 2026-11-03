"""
ca/evaluate.py
=================
Full evaluation of a trained CA actor (greedy = argmax of the masked logits, like PPO; DQN is evaluated with epsilon = 0), against
exactly the baselines and protocol of ppo/evaluate.py and dqn/evaluate.py (environment/eval_protocol.py).

    python -m ca.evaluate --checkpoint runs/ca_seed0/actor_final.pt --episodes 10 --out runs/ca_seed0/eval_4x4
    python -m ca.evaluate --checkpoint runs/ca_seed0/actor_final.pt --net_file sumo_3x3_network/grid3x3.net.xml \\
        --route_file sumo_3x3_network/routes3x3.rou.xml --episodes 10 --out runs/ca_seed0/eval_3x3
Or press Play (edit DEFAULT_SEED). scripts/run_eval.py calls this module for every (algo, seed, network).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.eligibility import eligibility_from_features  # noqa: E402
from environment.eval_protocol import add_standard_args, run_standard_evaluation  # noqa: E402
from ca.config import CAConfig  # noqa: E402
from ac.network import ActorNetwork  # noqa: E402

DEFAULT_SEED = 0
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "runs", f"ca_seed{DEFAULT_SEED}", "actor_final.pt")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "runs", f"ca_seed{DEFAULT_SEED}", "eval_4x4")


def ca_policy_fn(actor, device="cpu", greedy=True):
    def _policy(obs):
        x = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
        el = torch.as_tensor(eligibility_from_features(obs["node_features"]), device=device)
        if greedy:
            return actor.act_greedy(x, el).cpu().numpy()
        with torch.no_grad():
            return actor.act(x, el)[0].cpu().numpy()
    return _policy


def main():
    parser = add_standard_args(argparse.ArgumentParser(), DEFAULT_CHECKPOINT, DEFAULT_OUT)
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.checkpoint):
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    cfg = CAConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=device))
    actor.eval()
    run_standard_evaluation(ca_policy_fn(actor, device, greedy=not args.stochastic), algo_key="ca", algo_label="CA",
                            args=args, episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval)


if __name__ == "__main__":
    main()
