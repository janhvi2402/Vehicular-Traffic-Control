"""
ppo/train.py
============
Trains the shared per-junction PPO actor-critic on TrafficGridEnv.
Implements Algorithm 1 ("PPO, Actor-Critic Style") and the combined
clipped-surrogate + value-loss + entropy-bonus objective (Eq. 9) from
Schulman et al., 2017. See ppo/config.py for the full justification of
every hyperparameter against the paper's tables.

Adapted as follows from the paper's setting:
  - n_agents (16 on the 4x4 grid, 9 on the 3x3 generalization network)
    junctions play the role of the paper's N parallel actors, each
    contributing its own length-`horizon` trajectory every update
  - MLP (tanh activations, matching the paper's continuous-control MLP
    architecture) on an 11-dim hand-engineered feature vector, rather than
    the CNN used for Atari's raw pixels
  - like dqn/train.py, supports chunked/resumable runs (--chunk_steps) so
    a long training run can be split across multiple sessions without
    losing progress; only network weights persist across chunks (the
    optimizer's Adam moment estimates reset each chunk, the same
    simplification dqn/train.py makes for its optimizer)

Usage:
    python -m ppo.train --updates 196 --out runs/ppo_run1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from ppo.config import PPOConfig  # noqa: E402
from ppo.actor_critic import ActorCritic, RolloutBuffer  # noqa: E402


def train(cfg: PPOConfig, net_file: str, route_file: str, out_dir: str, use_gui: bool = False,
          resume_from: str | None = None, start_update: int = 0, start_episode: int = 0,
          chunk_updates: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    if chunk_updates is None:
        chunk_updates = cfg.total_updates - start_update

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    env_cfg = GridEnvConfig(
        net_file=net_file, route_file=route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
        seed=cfg.seed, use_gui=use_gui,
    )
    env = TrafficGridEnv(env_cfg)
    n_agents = env.n_agents

    net = ActorCritic(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    if resume_from:
        net.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed weights from {resume_from}")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

    buffer = RolloutBuffer(cfg.horizon, n_agents, cfg.input_dim)

    log_path = os.path.join(out_dir, "train_log.csv")
    write_header = not (resume_from and os.path.exists(log_path))
    log_file = open(log_path, "a" if resume_from else "w", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["update", "env_step", "episode", "episode_reward", "total_waiting_time",
             "throughput", "policy_loss", "value_loss", "entropy"]
        )

    obs, _ = env.reset(seed=cfg.seed + start_episode)
    episode = start_episode
    episode_reward = 0.0
    env_step = start_update * cfg.horizon

    end_update = min(start_update + chunk_updates, cfg.total_updates)
    for update in range(start_update + 1, end_update + 1):
        buffer.reset()
        for _ in range(cfg.horizon):
            feats = obs["node_features"]
            x = torch.as_tensor(feats, dtype=torch.float32, device=device)
            actions, log_probs, values = net.act(x)
            actions_np = actions.cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(actions_np)
            episode_done = terminated or truncated
            episode_reward += reward
            env_step += 1

            agent_rewards = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip)

            if truncated and not terminated:
                # This env only ever ends via the 3600s time limit --
                # terminated is always False, there is no true absorbing
                # state. compute_gae() zeroes the bootstrap wherever
                # buffer.dones[t]==1 (correctly, since self.values[t+1]
                # would otherwise belong to the NEXT, freshly-reset
                # episode -- an unrelated trajectory it must not bootstrap
                # from). But that means a plain truncation would silently
                # tell GAE this state's continuation is worth exactly 0,
                # which is false: the traffic network didn't end, the
                # clock just cut it off. Standard fix (CleanRL / Gymnasium
                # guidance for TimeLimit truncation in PPO): fold the TRUE
                # next-observation's bootstrap value directly into this
                # step's reward before GAE ever sees it, using next_obs as
                # it actually was BEFORE env.reset() overwrites `obs`.
                with torch.no_grad():
                    x_next = torch.as_tensor(next_obs["node_features"], dtype=torch.float32, device=device)
                    _, true_next_value = net.forward(x_next)
                agent_rewards = agent_rewards + cfg.gamma * true_next_value.cpu().numpy()

            buffer.add(feats, actions_np, log_probs.cpu().numpy(), values.cpu().numpy(),
                       agent_rewards, float(episode_done))

            obs = next_obs
            if episode_done:
                episode += 1
                obs, _ = env.reset(seed=cfg.seed + episode)
                episode_reward = 0.0

        # bootstrap value for the state right after the last stored step
        with torch.no_grad():
            x = torch.as_tensor(obs["node_features"], dtype=torch.float32, device=device)
            _, last_values = net.forward(x)
        advantages, returns = buffer.compute_gae(last_values.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

        b_obs, b_actions, b_log_probs, b_advantages, b_returns = buffer.get_flat(advantages, returns)
        # Standard PPO implementation detail (universal in reference code,
        # e.g. OpenAI Baselines), not explicitly tabulated in the paper:
        # normalize advantages per-update to stabilize the policy gradient
        # scale across updates.
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_obs_t = torch.as_tensor(b_obs, device=device)
        b_actions_t = torch.as_tensor(b_actions, device=device)
        b_log_probs_t = torch.as_tensor(b_log_probs, device=device)
        b_advantages_t = torch.as_tensor(b_advantages, device=device)
        b_returns_t = torch.as_tensor(b_returns, device=device)

        dataset_size = len(b_obs)
        last_policy_loss = last_value_loss = last_entropy = float("nan")
        for _epoch in range(cfg.num_epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, cfg.minibatch_size):
                mb_idx = idx[start:start + cfg.minibatch_size]
                logits, values_pred = net(b_obs_t[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions_t[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_log_probs_t[mb_idx])
                mb_adv = b_advantages_t[mb_idx]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()  # L^CLIP, Eq. 7 (negated for gradient descent)
                value_loss = (values_pred - b_returns_t[mb_idx]).pow(2).mean()  # L^VF, Eq. 9

                loss = policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

                last_policy_loss, last_value_loss, last_entropy = (
                    policy_loss.item(), value_loss.item(), entropy.item()
                )

        if update % cfg.log_every == 0:
            print(
                f"update {update:>5d} | env_step {env_step:>7d} | ep {episode:>4d} | "
                f"reward(ep so far) {episode_reward:8.1f} | wait {info['total_waiting_time']:8.1f} | "
                f"policy_loss {last_policy_loss:+.4f} | value_loss {last_value_loss:.4f} | entropy {last_entropy:.4f}"
            )
            log_writer.writerow([update, env_step, episode, episode_reward, info["total_waiting_time"],
                                  info["throughput"], last_policy_loss, last_value_loss, last_entropy])
            log_file.flush()

        if update % cfg.checkpoint_every == 0:
            torch.save(net.state_dict(), os.path.join(out_dir, f"actor_critic_update{update}.pt"))

    final_ckpt_name = "actor_critic_final.pt" if end_update >= cfg.total_updates else "actor_critic_latest.pt"
    torch.save(net.state_dict(), os.path.join(out_dir, final_ckpt_name))
    with open(os.path.join(out_dir, "resume_state.json"), "w") as f:
        import json
        json.dump({"update": end_update, "episode": episode, "done_training": end_update >= cfg.total_updates}, f)
    log_file.close()
    env.close()
    print(f"Chunk complete: ran updates {start_update + 1}..{end_update} of {cfg.total_updates}. "
          f"Checkpoint: {os.path.join(out_dir, final_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--out", default="runs/ppo_run1")
    parser.add_argument("--updates", type=int, default=None, help="override cfg.total_updates (grand total target)")
    parser.add_argument("--chunk_updates", type=int, default=None,
                         help="run only this many PPO updates in this invocation, then save and exit")
    parser.add_argument("--resume", default=None, help="path to a checkpoint (.pt) to resume weights from")
    parser.add_argument("--start_update", type=int, default=0)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.updates is not None:
        cfg.total_updates = args.updates

    train(
        cfg, args.net_file, args.route_file, args.out, use_gui=args.gui,
        resume_from=args.resume, start_update=args.start_update, start_episode=args.start_episode,
        chunk_updates=args.chunk_updates,
    )