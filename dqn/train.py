"""
dqn/train.py
============
Trains the shared per-junction DQN on TrafficGridEnv (Mnih et al., 2015, Algorithm 1, adapted:
one parameter-shared Q-network applied to all junctions, MLP on the 11-dim feature vector,
Adam, Huber loss). See dqn/config.py for hyperparameters.

Action masking (identical protocol to ppo/train.py): a junction that is not eligible to switch
(yellow, or elapsed < min_green) is forced to hold, also while exploring. The EXECUTED action is
what goes into replay, and the bootstrap max in the TD target ignores the switch Q-value in
next-states that are ineligible.

Usage:
    python -m dqn.train --seed 0 --out runs/dqn_seed0
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from environment.eligibility import eligibility_from_features  # noqa: E402
from environment.seeds import training_env_seed  # noqa: E402
from dqn.config import DQNConfig  # noqa: E402
from dqn.q_network import QNetwork, ReplayBuffer, dqn_targets, select_actions_epsilon_greedy  # noqa: E402

DEFAULT_NET_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "grid4x4.net.xml")
DEFAULT_ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_4x4_network", "routes.rou.xml")
DEFAULT_SEED = 0   # Play button: EDIT THIS to train another seed. The output folder follows it: runs/dqn_seed<N>
DEFAULT_TOTAL_EPISODES = 500  # must equal PPO's total_updates (one PPO update = one episode)


def linear_epsilon(step, total_steps, cfg: DQNConfig):
    anneal_steps = max(1, int(cfg.epsilon_decay_fraction * total_steps))
    if step >= anneal_steps:
        return cfg.epsilon_end
    return cfg.epsilon_start + (step / anneal_steps) * (cfg.epsilon_end - cfg.epsilon_start)


def make_optimizer(q_net: nn.Module, cfg: DQNConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(q_net.parameters(), lr=cfg.rmsprop_lr, momentum=cfg.rmsprop_momentum,
                                   alpha=cfg.rmsprop_alpha, eps=cfg.rmsprop_eps)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train(cfg: DQNConfig, net_file: str, route_file: str, out_dir: str, use_gui: bool = False,
          resume_from: str | None = None, start_step: int = 0, start_episode: int = 0,
          start_gradient_steps: int = 0, chunk_steps: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as _cf:   # documents exactly what this run used
        json.dump(dataclasses.asdict(cfg), _cf, indent=2)
    if chunk_steps is None:
        chunk_steps = cfg.total_env_steps - start_step
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_cfg = GridEnvConfig(net_file=net_file, route_file=route_file, episode_seconds=cfg.episode_seconds,
                            decision_interval=cfg.decision_interval, seed=cfg.seed, use_gui=use_gui)
    env = TrafficGridEnv(env_cfg)
    min_green, max_green = env.cfg.min_green, env.cfg.max_green

    q_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    target_net = QNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    if resume_from:
        q_net.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"Resumed weights from {resume_from}")
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = make_optimizer(q_net, cfg)
    loss_fn = nn.SmoothL1Loss(beta=cfg.huber_delta)
    buffer = ReplayBuffer(cfg.replay_capacity, seed=cfg.seed)
    buffer_path = os.path.join(out_dir, "replay_buffer.pkl")
    if resume_from and os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            buffer.buffer = pickle.load(f)
        print(f"Resumed replay buffer with {len(buffer)} transitions from {buffer_path}")

    log_path = os.path.join(out_dir, "train_log.csv")
    write_header = not (resume_from and os.path.exists(log_path))
    # resuming a log written before the *_ep columns existed: keep its column count so rows stay aligned
    log_n_cols = None
    if not write_header:
        with open(log_path) as _f:
            log_n_cols = len(_f.readline().strip().split(","))
    log_file = open(log_path, "a" if resume_from else "w", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        # first 7 columns unchanged; the last two are episode aggregates (the earlier columns are
        # end-of-episode snapshots): mean over the episode's steps of the grid-wide waiting time, and
        # the per-vehicle mean wait over vehicles completed so far this episode (end-of-episode value)
        log_writer.writerow(["env_step", "episode", "epsilon", "episode_reward", "total_waiting_time", "throughput", "loss",
                             "mean_waiting_time_ep", "mean_vehicle_wait_ep"])

    obs, _ = env.reset(seed=training_env_seed(cfg.seed, start_episode))
    episode = start_episode
    episode_reward = 0.0
    ep_wait_sum, ep_steps = 0.0, 0   # running sums for mean_waiting_time_ep
    # persisted across --resume chunks so the target-sync schedule does not depend on how the run was split
    gradient_steps = start_gradient_steps
    last_loss = float("nan")
    t0 = time.time()

    end_step = min(start_step + chunk_steps, cfg.total_env_steps)
    for step in range(start_step + 1, end_step + 1):
        epsilon = linear_epsilon(step, cfg.total_env_steps, cfg)
        node_feats = obs["node_features"]
        eligible = eligibility_from_features(node_feats, min_green, max_green)
        actions = select_actions_epsilon_greedy(q_net, node_feats, epsilon, rng, device, eligible)  # executed actions

        next_obs, reward, terminated, truncated, info = env.step(actions)
        # the env must have executed exactly what we chose (ineligible rows are 0 by construction)
        assert np.array_equal(actions, info["executed_actions"]), "executed action mismatch"
        # only `truncated` ends an episode here; bootstrap through time-limit cutoffs
        # (Pardo et al., "Time Limits in RL"): done = terminated only.
        done = terminated
        episode_done = terminated or truncated
        episode_reward += reward
        ep_wait_sum += info["total_waiting_time"]; ep_steps += 1

        agent_rewards = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip)
        next_feats = next_obs["node_features"]
        for i in range(env.n_agents):
            buffer.push(node_feats[i], int(info["executed_actions"][i]), float(agent_rewards[i]), next_feats[i], done)

        obs = next_obs

        if len(buffer) >= cfg.replay_start_size and step % cfg.train_frequency == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = buffer.sample(cfg.batch_size)
            b_next_elig = torch.as_tensor(eligibility_from_features(b_next_obs, min_green, max_green), device=device)
            b_obs_t = torch.as_tensor(b_obs, device=device)
            b_actions_t = torch.as_tensor(b_actions, device=device)
            b_rewards_t = torch.as_tensor(b_rewards, device=device)
            b_next_obs_t = torch.as_tensor(b_next_obs, device=device)
            b_dones_t = torch.as_tensor(b_dones, device=device)

            q_values = q_net(b_obs_t).gather(1, b_actions_t.unsqueeze(1)).squeeze(1)
            # ineligible next-states can only hold, so dqn_targets bootstraps from Q(hold) alone there
            targets = dqn_targets(target_net, b_rewards_t, b_next_obs_t, b_dones_t, cfg.gamma, b_next_elig)
            loss = loss_fn(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gradient_steps += 1
            last_loss = loss.item()

            if gradient_steps % cfg.target_update_frequency == 0:
                target_net.load_state_dict(q_net.state_dict())

        if step % cfg.log_every == 0:
            _el = time.time() - t0
            _eta = _el / max(step - start_step, 1) * (end_step - step)
            print(f"step {step:>7d} | ep {episode:>4d} | eps {epsilon:.3f} | "
                  f"reward(ep so far) {episode_reward:8.1f} | wait {info['total_waiting_time']:8.1f} | loss {last_loss:.4f} | "
                  f"{_el / 60:6.1f} min elapsed, ETA {_eta / 3600:5.2f} h", flush=True)
            # equals the full-episode mean when log_every is a multiple of the episode length (default: 720)
            row = [step, episode, epsilon, episode_reward, info["total_waiting_time"], info["throughput"], last_loss,
                   ep_wait_sum / max(ep_steps, 1), info["mean_vehicle_wait"]]
            log_writer.writerow(row[:log_n_cols] if log_n_cols else row)
            log_file.flush()

        if step % cfg.checkpoint_every == 0:
            torch.save(q_net.state_dict(), os.path.join(out_dir, f"qnet_step{step}.pt"))

        if episode_done:
            episode += 1
            obs, _ = env.reset(seed=training_env_seed(cfg.seed, episode))
            episode_reward = 0.0
            ep_wait_sum, ep_steps = 0.0, 0

    final_ckpt_name = "qnet_final.pt" if end_step >= cfg.total_env_steps else "qnet_latest.pt"
    torch.save(q_net.state_dict(), os.path.join(out_dir, final_ckpt_name))
    if end_step < cfg.total_env_steps:   # the buffer is only needed to continue a chunked run; skipped once training is
        with open(buffer_path, "wb") as f:   # complete (saves ~200 MB of disk and a long pickle per seed)
            pickle.dump(buffer.buffer, f)
    with open(os.path.join(out_dir, "resume_state.json"), "w") as f:
        json.dump({"step": end_step, "episode": episode, "gradient_steps": gradient_steps,
                   "done_training": end_step >= cfg.total_env_steps}, f)
    log_file.close()
    env.close()
    print(f"Episodes completed: {episode} (expected {cfg.total_env_steps // (cfg.episode_seconds // cfg.decision_interval)} for a full run)")
    print(f"Chunk complete: ran steps {start_step + 1}..{end_step} of {cfg.total_env_steps}. "
          f"Checkpoint: {os.path.join(out_dir, final_ckpt_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_file", default=DEFAULT_NET_FILE)
    parser.add_argument("--route_file", default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--out", default=None, help="default: runs/dqn_seed<seed>")
    parser.add_argument("--episodes", type=int, default=DEFAULT_TOTAL_EPISODES)
    parser.add_argument("--steps", type=int, default=None, help="raw total step count (overrides --episodes)")
    parser.add_argument("--seed", type=int, default=None, help="training seed (default: DQNConfig.seed)")
    parser.add_argument("--chunk_steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint (.pt) to resume weights from")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--start_gradient_steps", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args, _unknown = parser.parse_known_args()

    cfg = DQNConfig()
    cfg.seed = args.seed if args.seed is not None else DEFAULT_SEED
    out_dir = args.out or os.path.join(PROJECT_ROOT, "runs", f"dqn_seed{cfg.seed}")
    # protect a finished run: a second Play-button click must not silently overwrite hours of training
    if not args.resume and os.path.exists(os.path.join(out_dir, "qnet_final.pt")):
        sys.exit(f"STOP: {out_dir} already contains a finished run (qnet_final.pt).\n"
                 f"Change DEFAULT_SEED (or pass --seed/--out), or delete that folder if you really want to retrain.")
    if args.steps is not None:
        cfg.total_env_steps = args.steps
    else:
        cfg.total_env_steps = args.episodes * (cfg.episode_seconds // cfg.decision_interval)
        print(f"Training for {args.episodes} episodes -> {cfg.total_env_steps} total steps")

    train(cfg, args.net_file, args.route_file, out_dir, use_gui=args.gui,
          resume_from=args.resume, start_step=args.start_step, start_episode=args.start_episode,
          start_gradient_steps=args.start_gradient_steps, chunk_steps=args.chunk_steps)
