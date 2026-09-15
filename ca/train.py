"""
ca_avg/train.py
================
Trains the average-reward two-timescale critic-actor algorithm (Algorithm
1 of Panda & Bhatnagar, 2025, AAAI-25) on TrafficGridEnv. See
ca_avg/config.py's module docstring for exactly how this differs from
ac/train.py's (discounted-cost) CA -- most importantly: three recursions
instead of two (an explicit running average-reward estimate L, per
junction), and a TD error with NO discount factor at all.

Per-decision-step update (one env.step() = one iterate t in the paper),
independently per junction i (parameter-shared network, same
independent-agent pattern as dqn/ppo/ac):
  1. actor samples an action per junction from the current softmax policy
  2. env.step() gives the next obs and the per-junction reward r_i
  3. average-reward estimate update (paper's L_{t+1} = L_t + gamma_t*(r_t
     - L_t)): L_i <- L_i + gamma_t*(r_i - L_i)
  4. TD error (paper's delta_t = r_t - L_t + phi(s_{t+1})^T v_t -
     phi(s_t)^T v_t, generalized from linear FA to a neural critic exactly
     as dqn/ac already generalize their own papers' tabular/linear
     settings): delta_i = r_i - L_i + V(next_i) - V(current_i)
     -- note there is NO gamma*V(next) term; this is the average-reward
     TD error, not the discounted one used in ac/train.py
  5. critic update: regress V(current) towards V(current)+beta_t*delta,
     i.e. gradient descent on delta^2 with the TIME-VARYING learning rate
     beta_t (paper's slow timescale)
  6. actor update: theta_{t+1} = theta_t + alpha_t*delta_t*grad(log pi) --
     already reward-maximizing as written in the paper (see
     ca_avg/config.py's docstring for why, unlike ac/train.py, NO sign
     flip is needed here), using the TIME-VARYING learning rate alpha_t
     (paper's fast timescale, shared with L's gamma_t = K*alpha_t)

Usage:
    python -m ca_avg.train --steps 50000 --out runs/ca_avg_run1
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
from ca_avg.config import CAAvgConfig  # noqa: E402
from ac.network import ActorNetwork, CriticNetwork  # noqa: E402 -- reused directly, see config.py docstring


def step_size(step: int, lr0: float, exponent: float, decay_every: int) -> float:
    """alpha(n)/beta(n)/gamma(n): lr0 / (floor(n / decay_every) + 1) **
    exponent -- same step-size family as ac/train.py's step_size(), and
    the paper's own (Section "Sample Complexity Results")."""
    k = step // decay_every
    return lr0 / ((k + 1) ** exponent)


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def train(cfg: CAAvgConfig, net_file: str, route_file: str, out_dir: str, use_gui: bool = False,
          resume_actor: str | None = None, resume_critic: str | None = None,
          start_step: int = 0, start_episode: int = 0, start_avg_reward: np.ndarray | None = None,
          chunk_steps: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    if chunk_steps is None:
        chunk_steps = cfg.total_env_steps - start_step
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    decay_every = max(1, int(cfg.lr_decay_every_fraction * cfg.total_env_steps))
    print(f"ca_avg: actor_exponent(nu)={cfg.actor_exponent} (fast, shared with avg-reward estimate), "
          f"critic_exponent(sigma)={cfg.critic_exponent} (slow) "
          f"[smaller exponent = faster timescale; paper requires nu < sigma]")
    print(f"total_env_steps={cfg.total_env_steps} -> lr_decay_every={decay_every} steps "
          f"({cfg.lr_decay_every_fraction:.1%} of total)")

    env_cfg = GridEnvConfig(
        net_file=net_file, route_file=route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
        seed=cfg.seed, use_gui=use_gui,
    )
    env = TrafficGridEnv(env_cfg)
    n_agents = env.n_agents

    actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    critic = CriticNetwork(cfg.input_dim, cfg.hidden_sizes).to(device)
    if resume_actor:
        actor.load_state_dict(torch.load(resume_actor, map_location=device))
        print(f"Resumed actor weights from {resume_actor}")
    if resume_critic:
        critic.load_state_dict(torch.load(resume_critic, map_location=device))
        print(f"Resumed critic weights from {resume_critic}")

    # Adam is the per-parameter step-taking mechanism; the actual step-size
    # MAGNITUDE at iterate t is fully controlled by set_lr() every step,
    # from alpha_t/beta_t below -- same rationale as ac/train.py's use of
    # Adam under a manually-scheduled learning rate.
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr0)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr0)

    # Average-reward estimate L, one per junction (paper's L_t is scalar
    # for a single-agent MDP; we track it per-junction, same independent-
    # agent-recursions pattern ac/train.py uses for V and theta).
    avg_reward = np.zeros(n_agents, dtype=np.float32) if start_avg_reward is None else start_avg_reward.copy()

    log_path = os.path.join(out_dir, "train_log.csv")
    write_header = not (resume_actor and os.path.exists(log_path))
    log_file = open(log_path, "a" if (resume_actor and os.path.exists(log_path)) else "w", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["env_step", "episode", "actor_lr", "critic_lr", "avg_reward_lr", "episode_reward",
             "total_waiting_time", "throughput", "critic_loss", "actor_loss", "entropy",
             "mean_avg_reward_estimate"]
        )

    obs, _ = env.reset(seed=cfg.seed + start_episode)
    episode = start_episode
    episode_reward = 0.0
    last_critic_loss = last_actor_loss = last_entropy = float("nan")

    end_step = min(start_step + chunk_steps, cfg.total_env_steps)
    for step in range(start_step + 1, end_step + 1):
        a_lr = step_size(step, cfg.actor_lr0, cfg.actor_exponent, decay_every)
        c_lr = step_size(step, cfg.critic_lr0, cfg.critic_exponent, decay_every)
        g_lr = cfg.avg_reward_k * a_lr  # paper's gamma_t = K*alpha_t
        set_lr(actor_optimizer, a_lr)
        set_lr(critic_optimizer, c_lr)

        feats = obs["node_features"]
        x = torch.as_tensor(feats, dtype=torch.float32, device=device)

        actions, log_probs, entropy = actor.act(x)
        actions_np = actions.cpu().numpy()

        v_current = critic(x)  # V_w(s), one value per junction

        next_obs, reward, terminated, truncated, info = env.step(actions_np)
        done = terminated or truncated
        episode_reward += reward

        agent_rewards = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip)

        # --- average-reward estimate update: L_i <- L_i + gamma_t*(r_i - L_i) ---
        avg_reward = avg_reward + g_lr * (agent_rewards - avg_reward)
        avg_reward_t = torch.as_tensor(avg_reward, dtype=torch.float32, device=device)
        rewards_t = torch.as_tensor(agent_rewards, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_x = torch.as_tensor(next_obs["node_features"], dtype=torch.float32, device=device)
            # NOTE: the paper's theory assumes a continuing, ergodic Markov
            # chain (Assumption 4.3/5.1) with no episode termination at
            # all -- the average-reward formulation isn't naturally
            # episodic. Our environment does have finite 3600s episodes,
            # so we zero the bootstrap at episode end as a pragmatic
            # adaptation (same convention ac/train.py uses for its own,
            # discounted, episode boundary), not something the paper
            # itself specifies.
            v_next = critic(next_x) if not done else torch.zeros_like(v_current)

        # Average-reward TD error -- NO discount factor, unlike
        # ac/train.py's td_error = r + gamma*v_next - v_current. This is
        # the defining difference of the average-reward setting: reward is
        # compared against the running average avg_reward, not discounted
        # against an infinite horizon.
        td_error = rewards_t - avg_reward_t + v_next - v_current

        # --- critic update (paper's slower recursion) ---
        critic_loss = td_error.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- actor update (paper's faster recursion, shared timescale
        # with the average-reward estimate above). Already reward-
        # maximizing as written -- see config.py docstring for why no
        # sign flip is needed here, unlike ac/train.py's cost-to-reward
        # conversion. td_error is detached: the actor must not backprop
        # through the critic's own value estimate. ---
        actor_loss = -(log_probs * td_error.detach()).mean() - cfg.entropy_coeff * entropy.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        last_critic_loss = critic_loss.item()
        last_actor_loss = actor_loss.item()
        last_entropy = entropy.mean().item()

        obs = next_obs

        if step % cfg.log_every == 0:
            print(
                f"step {step:>7d} | ep {episode:>4d} | a_lr {a_lr:.2e} | c_lr {c_lr:.2e} | "
                f"g_lr {g_lr:.2e} | reward(ep so far) {episode_reward:8.1f} | "
                f"wait {info['total_waiting_time']:8.1f} | critic_loss {last_critic_loss:.4f} | "
                f"actor_loss {last_actor_loss:+.4f} | entropy {last_entropy:.4f} | "
                f"avg_reward(mean over junctions) {float(avg_reward.mean()):+.4f}"
            )
            log_writer.writerow([step, episode, a_lr, c_lr, g_lr, episode_reward, info["total_waiting_time"],
                                  info["throughput"], last_critic_loss, last_actor_loss, last_entropy,
                                  float(avg_reward.mean())])
            log_file.flush()

        if step % cfg.checkpoint_every == 0:
            torch.save(actor.state_dict(), os.path.join(out_dir, f"actor_step{step}.pt"))
            torch.save(critic.state_dict(), os.path.join(out_dir, f"critic_step{step}.pt"))
            np.save(os.path.join(out_dir, f"avg_reward_step{step}.npy"), avg_reward)

        if done:
            episode += 1
            obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0

    final_suffix = "final" if end_step >= cfg.total_env_steps else "latest"
    actor_ckpt = os.path.join(out_dir, f"actor_{final_suffix}.pt")
    critic_ckpt = os.path.join(out_dir, f"critic_{final_suffix}.pt")
    avg_reward_ckpt = os.path.join(out_dir, f"avg_reward_{final_suffix}.npy")
    torch.save(actor.state_dict(), actor_ckpt)
    torch.save(critic.state_dict(), critic_ckpt)
    np.save(avg_reward_ckpt, avg_reward)
    with open(os.path.join(out_dir, "resume_state.json"), "w") as f:
        import json
        json.dump({"step": end_step, "episode": episode,
                   "done_training": end_step >= cfg.total_env_steps}, f)
    log_file.close()
    env.close()
    print(f"Chunk complete: ran steps {start_step + 1}..{end_step} of {cfg.total_env_steps}. "
          f"Checkpoints: {actor_ckpt}, {critic_ckpt}, {avg_reward_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--out", default="runs/ca_avg_run1")
    parser.add_argument("--steps", type=int, default=None, help="override cfg.total_env_steps")
    parser.add_argument("--chunk_steps", type=int, default=None,
                         help="run only this many steps in this invocation, then save and exit")
    parser.add_argument("--resume_actor", default=None, help="path to an actor checkpoint (.pt) to resume from")
    parser.add_argument("--resume_critic", default=None, help="path to a critic checkpoint (.pt) to resume from")
    parser.add_argument("--resume_avg_reward", default=None,
                         help="path to an avg_reward_*.npy checkpoint to resume the average-reward "
                              "estimate from (required alongside --resume_actor/--resume_critic to "
                              "resume correctly -- otherwise the estimate restarts from zero)")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    cfg = CAAvgConfig()
    if args.steps is not None:
        cfg.total_env_steps = args.steps

    start_avg_reward = np.load(args.resume_avg_reward) if args.resume_avg_reward else None

    train(
        cfg, args.net_file, args.route_file, args.out, use_gui=args.gui,
        resume_actor=args.resume_actor, resume_critic=args.resume_critic,
        start_step=args.start_step, start_episode=args.start_episode,
        start_avg_reward=start_avg_reward, chunk_steps=args.chunk_steps,
    )
