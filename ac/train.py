"""
ac/train.py
===========
Trains the shared per-junction actor and critic networks on TrafficGridEnv,
implementing the two-timescale recursions (7)-(8) of Bhatnagar, Borkar &
Guin, 2024 ("Actor-Critic or Critic-Actor? A Tale of Two Time Scales"),
generalized from a tabular V(i)/theta(i,a) to neural function approximation
-- the same style of generalization the paper itself demonstrates in its
Figs. 8-9 "neural network approximation" experiments (2-hidden-layer MLP,
tanh activations, policy-gradient actor + TD(0) critic).

Per-decision-step update (one env.step() = one iterate n in the paper):
  1. actor samples an action per junction from the current softmax policy
     (Eq. 6) -- this is phi_n(i), just above Eq. 7
  2. critic computes V_w(s) for the CURRENT obs (needed for both the TD
     target and, in the CA variant, as the actor's own baseline)
  3. env.step() gives the next obs and the per-junction reward
  4. TD error delta_n(i) = r + gamma * V_w(s') * (1-done) - V_w(s) is
     computed per junction -- this plays the role of BOTH
       (a) the critic's own update target (Eq. 7 collapses to a TD(0)
           update towards r + gamma*V(s') exactly when g and Vn are
           evaluated as delta implies), and
       (b) the actor's advantage estimate (Eq. 8's bracketed term
           [Vn(i) - g - gamma*Vn(next)] is exactly -delta_n(i))
  5. BOTH networks are updated on every step (matching the paper's
     recursions, which are both defined at every n) -- what makes this
     "two-timescale" is that the two optimizers use DIFFERENT, differently
     -decaying step-size sequences a(n) (critic) / b(n) (actor), not that
     one updates less often. See config.py's docstring for the exact
     variant -> (fast, slow) exponent assignment.

Usage:
    python -m ac.train --variant ac --steps 25000 --out runs/ac_run1
    python -m ac.train --variant ca --steps 25000 --out runs/ca_run1
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
from ac.config import ACConfig  # noqa: E402
from ac.network import ActorNetwork, CriticNetwork  # noqa: E402


def step_size(step: int, lr0: float, exponent: float, decay_every: int) -> float:
    """a(n) or b(n): lr0 / (floor(n / decay_every) + 1) ** exponent, the
    paper's own step-size family (Section V numerical results)."""
    k = step // decay_every
    return lr0 / ((k + 1) ** exponent)


def assign_exponents(cfg: ACConfig):
    """Returns (critic_exponent, actor_exponent) per config.py's docstring:
    the SMALLER exponent goes to whichever recursion is on the FASTER
    timescale for the chosen variant."""
    if cfg.variant == "ac":
        # standard actor-critic: critic (V) is faster (paper, Section I:
        # "faster time scale component for value function evaluation")
        return cfg.fast_exponent, cfg.slow_exponent   # (critic, actor)
    elif cfg.variant == "ca":
        # critic-actor: timescales reversed (paper, Section III:
        # "a(n) = o(b(n))" i.e. critic's step size shrinks faster -> critic
        # is the SLOW one here, actor is fast)
        return cfg.slow_exponent, cfg.fast_exponent   # (critic, actor)
    raise ValueError(f"Unknown variant: {cfg.variant} (expected 'ac' or 'ca')")


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def train(cfg: ACConfig, net_file: str, route_file: str, out_dir: str, use_gui: bool = False,
          resume_actor: str | None = None, resume_critic: str | None = None,
          start_step: int = 0, start_episode: int = 0, chunk_steps: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    if chunk_steps is None:
        chunk_steps = cfg.total_env_steps - start_step
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    critic_exponent, actor_exponent = assign_exponents(cfg)
    decay_every = max(1, int(cfg.lr_decay_every_fraction * cfg.total_env_steps))
    print(f"variant={cfg.variant} -> critic_exponent={critic_exponent} (a(n)), "
          f"actor_exponent={actor_exponent} (b(n)) "
          f"[smaller exponent = faster timescale]")
    print(f"total_env_steps={cfg.total_env_steps} -> lr_decay_every={decay_every} steps "
          f"({cfg.lr_decay_every_fraction:.1%} of total, so the schedule's shape matches "
          f"regardless of total_env_steps -- e.g. the same shape at --steps 50000 as at 25000)")

    env_cfg = GridEnvConfig(
        net_file=net_file, route_file=route_file,
        episode_seconds=cfg.episode_seconds, decision_interval=cfg.decision_interval,
        seed=cfg.seed, use_gui=use_gui,
    )
    env = TrafficGridEnv(env_cfg)

    actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
    critic = CriticNetwork(cfg.input_dim, cfg.hidden_sizes).to(device)
    if resume_actor:
        actor.load_state_dict(torch.load(resume_actor, map_location=device))
        print(f"Resumed actor weights from {resume_actor}")
    if resume_critic:
        critic.load_state_dict(torch.load(resume_critic, map_location=device))
        print(f"Resumed critic weights from {resume_critic}")

    # Adam is used purely as the per-parameter step-taking mechanism; the
    # actual step-size MAGNITUDE at iterate n is fully controlled by
    # set_lr() below every single step, from the paper's own a(n)/b(n)
    # schedule -- Adam's internal moment estimates still help condition
    # the update direction, same rationale as dqn/config.py's Adam choice.
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr0)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr0)

    log_path = os.path.join(out_dir, "train_log.csv")
    write_header = not (resume_actor and os.path.exists(log_path))
    log_file = open(log_path, "a" if (resume_actor and os.path.exists(log_path)) else "w", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(
            ["env_step", "episode", "critic_lr", "actor_lr", "episode_reward",
             "total_waiting_time", "throughput", "critic_loss", "actor_loss", "entropy"]
        )

    obs, _ = env.reset(seed=cfg.seed + start_episode)
    episode = start_episode
    episode_reward = 0.0
    last_critic_loss = last_actor_loss = last_entropy = float("nan")

    end_step = min(start_step + chunk_steps, cfg.total_env_steps)
    for step in range(start_step + 1, end_step + 1):
        c_lr = step_size(step, cfg.critic_lr0, critic_exponent, decay_every)
        a_lr = step_size(step, cfg.actor_lr0, actor_exponent, decay_every)
        set_lr(critic_optimizer, c_lr)
        set_lr(actor_optimizer, a_lr)

        feats = obs["node_features"]
        x = torch.as_tensor(feats, dtype=torch.float32, device=device)

        actions, log_probs, entropy = actor.act(x)
        actions_np = actions.cpu().numpy()

        v_current = critic(x)  # V_w(s), one value per junction

        next_obs, reward, terminated, truncated, info = env.step(actions_np)
        episode_done = terminated or truncated
        episode_reward += reward

        agent_rewards = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip)
        rewards_t = torch.as_tensor(agent_rewards, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_x = torch.as_tensor(next_obs["node_features"], dtype=torch.float32, device=device)
            v_next = critic(next_x) if not terminated else torch.zeros_like(v_current)   # ← was "done"

        # paper's Eq. 7 TD error, per junction (each junction is its own
        # independent (i, a) tuple being updated this step, matching
        # dqn/ppo's independent-per-junction / parameter-shared design)
        td_error = rewards_t + cfg.gamma * v_next * (0.0 if terminated else 1.0) - v_current  # ← was "done"

        # --- critic update: regress V_w(s) towards the TD target,
        # equivalent to Eq. 7's V_{n+1}(i) = V_n(i) + a(n)*delta_n(i) for a
        # linear/tabular V but generalized to gradient descent for a
        # neural V_w (same generalization dqn/train.py makes from tabular
        # Q-learning to a neural Q-network) ---
        critic_loss = td_error.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # --- actor update: paper's Eq. 8, theta_{n+1}(i,a) = theta_n(i,a)
        # + b(n)*[V(i) - g - gamma*V(next)]*1{Z_n=(i,a)}, i.e. a step in
        # the direction of +delta_n(i) on the LOG-probability of the
        # action actually taken -- exactly REINFORCE/policy-gradient with
        # the TD error as the advantage (detached: the actor must NOT
        # backprop through the critic's own value estimate, matching the
        # paper's treatment of V as quasi-static/quasi-equilibrated from
        # the actor's point of view in both variants) ---
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
                f"step {step:>7d} | ep {episode:>4d} | c_lr {c_lr:.2e} | a_lr {a_lr:.2e} | "
                f"reward(ep so far) {episode_reward:8.1f} | wait {info['total_waiting_time']:8.1f} | "
                f"critic_loss {last_critic_loss:.4f} | actor_loss {last_actor_loss:+.4f} | "
                f"entropy {last_entropy:.4f}"
            )
            log_writer.writerow([step, episode, c_lr, a_lr, episode_reward, info["total_waiting_time"],
                                  info["throughput"], last_critic_loss, last_actor_loss, last_entropy])
            log_file.flush()

        if step % cfg.checkpoint_every == 0:
            torch.save(actor.state_dict(), os.path.join(out_dir, f"actor_step{step}.pt"))
            torch.save(critic.state_dict(), os.path.join(out_dir, f"critic_step{step}.pt"))

        if episode_done:                                 # ← was "done"
            episode += 1
            obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0

    final_suffix = "final" if end_step >= cfg.total_env_steps else "latest"
    actor_ckpt = os.path.join(out_dir, f"actor_{final_suffix}.pt")
    critic_ckpt = os.path.join(out_dir, f"critic_{final_suffix}.pt")
    torch.save(actor.state_dict(), actor_ckpt)
    torch.save(critic.state_dict(), critic_ckpt)
    with open(os.path.join(out_dir, "resume_state.json"), "w") as f:
        import json
        json.dump({"step": end_step, "episode": episode, "variant": cfg.variant,
                   "done_training": end_step >= cfg.total_env_steps}, f)
    log_file.close()
    env.close()
    print(f"Chunk complete: ran steps {start_step + 1}..{end_step} of {cfg.total_env_steps}. "
          f"Checkpoints: {actor_ckpt}, {critic_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["ac", "ca"], default="ac",
                         help="'ac' = standard actor-critic (critic faster); "
                              "'ca' = critic-actor (actor faster, Bhatnagar/Borkar/Guin 2024)")
    parser.add_argument("--net_file", default="sumo_4x4_network/grid4x4.net.xml")
    parser.add_argument("--route_file", default="sumo_4x4_network/routes.rou.xml")
    parser.add_argument("--out", default="runs/ac_run1")
    parser.add_argument("--steps", type=int, default=None, help="override cfg.total_env_steps")
    parser.add_argument("--chunk_steps", type=int, default=None,
                         help="run only this many steps in this invocation, then save and exit")
    parser.add_argument("--resume_actor", default=None, help="path to an actor checkpoint (.pt) to resume from")
    parser.add_argument("--resume_critic", default=None, help="path to a critic checkpoint (.pt) to resume from")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    cfg = ACConfig(variant=args.variant)
    if args.steps is not None:
        cfg.total_env_steps = args.steps

    train(
        cfg, args.net_file, args.route_file, args.out, use_gui=args.gui,
        resume_actor=args.resume_actor, resume_critic=args.resume_critic,
        start_step=args.start_step, start_episode=args.start_episode,
        chunk_steps=args.chunk_steps,
    )
