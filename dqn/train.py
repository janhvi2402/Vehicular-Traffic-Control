"""
dqn/train.py
============
Trains the shared per-junction DQN on TrafficGridEnv. Implements
Algorithm 1 ("deep Q-learning with experience replay") from Mnih et al.,
2015, adapted as follows (each deviation is justified in config.py):
  - one shared Q-network applied independently to all 16 junctions each
    step (parameter-sharing / independent-DQN), instead of one Atari
    agent controlling one joystick
  - MLP on an 11-dim hand-engineered feature vector instead of a CNN on
    raw pixels (no frame stacking / preprocessing needed)
  - Adam instead of RMSProp (see config.py); Huber/smooth-L1 loss kept
    identical to the paper's error-clipping scheme
  - replay memory size, target-update frequency, replay-start size and
    exploration schedule all scaled down from the paper's Atari-scale
    values to match our far shorter total training budget

Usage:
    python -m dqn.train --steps 150000 --out runs/dqn_run1
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.q_network import QNetwork, ReplayBuffer, select_actions_epsilon_greedy  # noqa: E402


def linear_epsilon(step, total_steps, cfg: DQNConfig):
    anneal_steps = max(1, int(cfg.epsilon_decay_fraction * total_steps))
    if step >= anneal_steps:
        return cfg.epsilon_end
    frac = step / anneal_steps
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def make_optimizer(q_net: nn.Module, cfg: DQNConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            q_net.parameters(),
            lr=cfg.rmsprop_lr,
            momentum=cfg.rmsprop_momentum,
            alpha=cfg.rmsprop_alpha,
            eps=cfg.rmsprop_eps,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train(cfg: DQNConfig, net_file: str, route_file: str, out_dir: str, use_gui: bool = False,
          resume_from: str | None = None, start_step: int = 0, start_episode: int = 0,
          start_gradient_steps: int = 0, chunk_steps: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    if chunk_steps is None:
        chunk_steps = cfg.total_env_steps - start_step
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_cfg = GridEnvConfig(
        net_file=net_file,
        route_file=route_file,
        episode_seconds=cfg.episode_seconds,
        decision_interval=cfg.decision_interval,
        seed=cfg.seed,
        use_gui=use_gui,
    )
    env = TrafficGridEnv(env_cfg)

    q_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    target_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    if resume_from:
        q_net.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed weights from {resume_from}")
    target_net.load_state_dict(q_net.state_dict())  # "Initialize target action-value function Q^ with weights theta- = theta"
    target_net.eval()

    optimizer = make_optimizer(q_net, cfg)
    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_delta)  # paper's [-1,1] error clipping == Huber/smooth-L1
    buffer = ReplayBuffer(cfg.replay_capacity, seed=cfg.seed)
    buffer_path = os.path.join(out_dir, "replay_buffer.pkl")
    if resume_from and os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            buffer.buffer = pickle.load(f)
        print(f"Resumed replay buffer with {len(buffer)} transitions from {buffer_path}")

    log_path = os.path.join(out_dir, "train_log.csv")
    write_header = not (resume_from and os.path.exists(log_path))
    log_file = open(log_path, "a" if resume_from else "w", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["env_step", "episode", "epsilon", "episode_reward", "total_waiting_time", "throughput", "loss"]
        )

    obs, _ = env.reset(seed=cfg.seed + start_episode)
    episode = start_episode
    episode_reward = 0.0
    # NOTE: gradient_steps starts from start_gradient_steps (persisted in
    # resume_state.json across chunks), NOT from 0. It gates the target-
    # network sync below ("every C steps reset Q^ = Q"). If it reset to 0
    # on every --resume'd chunk instead, a run split across N chunks would
    # sync its target network far more often (once every
    # target_update_frequency gradient steps *within each chunk*) than the
    # same run executed in one sitting -- silently changing training
    # dynamics based on how the run happened to be split, not on how much
    # training actually occurred. Pass --start_gradient_steps (read from
    # the previous chunk's resume_state.json) whenever you --resume.
    gradient_steps = start_gradient_steps
    last_loss = float("nan")

    end_step = min(start_step + chunk_steps, cfg.total_env_steps)
    for step in range(start_step + 1, end_step + 1):
        epsilon = linear_epsilon(step, cfg.total_env_steps, cfg)
        node_feats = obs["node_features"]
        actions = select_actions_epsilon_greedy(q_net, node_feats, epsilon, rng, device)

        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        episode_reward += reward

        agent_rewards = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip)
        next_feats = next_obs["node_features"]
        for i in range(env.n_agents):
            buffer.push(node_feats[i], int(actions[i]), float(agent_rewards[i]), next_feats[i], done)

        obs = next_obs

        # --- learning update ("Sample random minibatch ... perform a
        # gradient descent step", every train_frequency env steps, matching
        # the paper's update_frequency=4) ---
        if len(buffer) >= cfg.replay_start_size and step % cfg.train_frequency == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(cfg.batch_size)
            b_obs_t = torch.as_tensor(b_obs, device=device)
            b_actions_t = torch.as_tensor(b_actions, device=device)
            b_rewards_t = torch.as_tensor(b_rewards, device=device)
            b_next_obs_t = torch.as_tensor(b_next_obs, device=device)
            b_dones_t = torch.as_tensor(b_dones, device=device)

            q_values = q_net(b_obs_t).gather(1, b_actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(b_next_obs_t).max(dim=1).values
                targets = b_rewards_t + cfg.gamma * next_q * (1.0 - b_dones_t)
            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gradient_steps += 1
            last_loss = loss.item()

            # "Every C steps reset Q^ = Q"
            if gradient_steps % cfg.target_update_frequency == 0:
                target_net.load_state_dict(q_net.state_dict())

        if step % cfg.log_every == 0:
            print(
                f"step {step:>7d} | ep {episode:>4d} | eps {epsilon:.3f} | "
                f"reward(ep so far) {episode_reward:8.1f} | wait {info['total_waiting_time']:8.1f} | "
                f"loss {last_loss:.4f}"
            )
            log_writer.writerow([step, episode, epsilon, episode_reward, info["total_waiting_time"], info["throughput"], last_loss])
            log_file.flush()

        if step % cfg.checkpoint_every == 0:
            torch.save(q_net.state_dict(), os.path.join(out_dir, f"qnet_step{step}.pt"))

        if done:
            episode += 1
            obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0

    final_ckpt_name = "qnet_final.pt" if end_step >= cfg.total_env_steps else "qnet_latest.pt"
    torch.save(q_net.state_dict(), os.path.join(out_dir, final_ckpt_name))
    with open(buffer_path, "wb") as f:
        pickle.dump(buffer.buffer, f)
    with open(os.path.join(out_dir, "resume_state.json"), "w") as f:
        import json
        json.dump({"step": end_step, "episode": episode, "gradient_steps": gradient_steps,
                   "done_training": end_step >= cfg.total_env_steps}, f)
    log_file.close()
    env.close()
    print(f"Chunk complete: ran steps {start_step + 1}..{end_step} of {cfg.total_env_steps}. "
          f"Checkpoint: {os.path.join(out_dir, final_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--out", default="runs/dqn_run1")
    parser.add_argument("--steps", type=int, default=None, help="override cfg.total_env_steps (grand total target)")
    parser.add_argument("--chunk_steps", type=int, default=None,
                         help="run only this many steps in this invocation, then save and exit "
                              "(for splitting a long run across multiple sessions/chunks)")
    parser.add_argument("--resume", default=None, help="path to a checkpoint (.pt) to resume weights from")
    parser.add_argument("--start_step", type=int, default=0, help="absolute step count already completed")
    parser.add_argument("--start_episode", type=int, default=0, help="episode count already completed")
    parser.add_argument("--start_gradient_steps", type=int, default=0,
                         help="gradient-step count already completed (read the 'gradient_steps' field "
                              "from the previous chunk's resume_state.json when using --resume, so the "
                              "target-network sync schedule stays correct across chunks)")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    cfg = DQNConfig()
    if args.steps is not None:
        cfg.total_env_steps = args.steps

    train(
        cfg, args.net_file, args.route_file, args.out, use_gui=args.gui,
        resume_from=args.resume, start_step=args.start_step, start_episode=args.start_episode,
        start_gradient_steps=args.start_gradient_steps, chunk_steps=args.chunk_steps,
    )