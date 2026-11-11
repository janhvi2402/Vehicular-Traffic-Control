"""
environment/online_trainer.py
=============================
ONE training loop shared by the online (one-step, no replay buffer, no rollout) algorithms:
ac/ (two-timescale actor-critic) and ca/ (average-reward two-timescale critic-actor).
Everything that is not the learning rule itself lives here, so AC and CA can differ ONLY in their
update equations -- exactly the property a fair comparison needs, and the same protocol dqn/train.py and
ppo/train.py follow:

  * env seed of episode e of training seed s = environment.seeds.training_env_seed(s, e)   (never overlaps eval/sweep seeds)
  * the policy is masked with environment.eligibility.eligibility_from_features, and the EXECUTED action is what
    is learned from (asserted every step: actions == info["executed_actions"])
  * per-junction rewards info["agent_rewards"] clipped to +-cfg.reward_clip (same 50.0 as DQN / PPO)
  * only `terminated` cuts the bootstrap; the 3600 s time limit (`truncated`) bootstraps through, like DQN and PPO
  * budget cfg.total_env_steps (default 360,000 = 500 episodes), one train_log.csv row per finished episode with the
    same column names as dqn/ppo (env_step, episode, episode_reward, total_waiting_time, throughput,
    mean_waiting_time_ep, mean_vehicle_wait_ep) followed by the agent's own metrics
  * runs/<algo>_seed<N>/ with config.json, actor_final.pt, critic_final.pt; a finished run is never overwritten

Agent interface (see ac/agent.py, ca/agent.py):
    agent.metric_names                       list[str]   extra log columns (episode means)
    agent.act(feats, eligible) -> int64 (N,) masked sample; remembers what update() needs
    agent.update(next_feats, rewards, terminated, step) -> dict[str, float]   one learning step
    agent.state_dict() / load_state_dict(d)  full training state (nets, optimizers, estimates)
    agent.save_policy(out_dir, tag)          writes actor_<tag>.pt and critic_<tag>.pt
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.grid_env import TrafficGridEnv, GridEnvConfig  # noqa: E402
from environment.eligibility import eligibility_from_features  # noqa: E402
from environment.seeds import training_env_seed  # noqa: E402

FIXED_COLUMNS = ["env_step", "episode", "episode_reward", "total_waiting_time", "throughput",
                 "mean_waiting_time_ep", "mean_vehicle_wait_ep", "eligible_fraction"]


def step_size(step: int, lr0: float, exponent: float, decay_every: int) -> float:
    """lr0 / (floor(step / decay_every) + 1) ** exponent -- the step-size family of both papers
    (Bhatnagar-Borkar-Guin Sec. V; Panda-Bhatnagar Sec. 'Sample Complexity Results')."""
    return lr0 / ((step // decay_every + 1) ** exponent)


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


def train_online(make_agent, cfg, net_file: str, route_file: str, out_dir: str, *, algo: str,
                 use_gui: bool = False, resume_from: str | None = None, chunk_steps: int | None = None):
    """make_agent(cfg, device) -> agent. Returns the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:      # documents exactly what this run used
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    agent = make_agent(cfg, device)

    steps_per_episode = cfg.episode_seconds // cfg.decision_interval
    start_step, episode = 0, 0
    if resume_from:
        state = torch.load(resume_from, map_location=device, weights_only=False)   # our own resume file (holds optimizer state)
        agent.load_state_dict(state["agent"])
        start_step, episode = int(state["step"]), int(state["episode"])
        print(f"Resumed {algo} from {resume_from}: step {start_step}, episode {episode}")
    if chunk_steps is None:
        chunk_steps = cfg.total_env_steps - start_step
    elif chunk_steps % steps_per_episode:
        warnings.warn(f"chunk_steps={chunk_steps} is not a multiple of {steps_per_episode}: the chunk ends mid-episode and "
                      f"the next chunk restarts that episode (its steps still count towards the budget).")
    end_step = min(start_step + chunk_steps, cfg.total_env_steps)

    env = TrafficGridEnv(GridEnvConfig(net_file=net_file, route_file=route_file, episode_seconds=cfg.episode_seconds,
                                       decision_interval=cfg.decision_interval, seed=cfg.seed, use_gui=use_gui))
    min_green, max_green = env.cfg.min_green, env.cfg.max_green

    log_path = os.path.join(out_dir, "train_log.csv")
    resuming_log = bool(resume_from) and os.path.exists(log_path)
    log_file = open(log_path, "a" if resuming_log else "w", newline="")
    writer = csv.writer(log_file)
    if not resuming_log:
        writer.writerow(FIXED_COLUMNS + list(agent.metric_names))

    obs, _ = env.reset(seed=training_env_seed(cfg.seed, episode))
    ep_reward, ep_wait_sum, ep_elig_sum, ep_steps = 0.0, 0.0, 0.0, 0
    ep_metrics = {k: 0.0 for k in agent.metric_names}
    t0 = time.time()

    for step in range(start_step + 1, end_step + 1):
        feats = obs["node_features"]
        eligible = eligibility_from_features(feats, min_green, max_green)
        actions = agent.act(feats, eligible)                                   # the EXECUTED action (ineligible rows are 0)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        assert np.array_equal(actions, info["executed_actions"]), "executed action mismatch"
        rewards = np.clip(info["agent_rewards"], -cfg.reward_clip, cfg.reward_clip).astype(np.float32)
        metrics = agent.update(next_obs["node_features"], rewards, bool(terminated), step)

        ep_reward += reward
        ep_wait_sum += info["total_waiting_time"]
        ep_elig_sum += float(eligible.mean())
        ep_steps += 1
        for k in agent.metric_names:
            ep_metrics[k] += float(metrics[k])
        obs = next_obs

        if terminated or truncated:
            row = [step, episode, ep_reward, info["total_waiting_time"], info["throughput"],
                   ep_wait_sum / ep_steps, info["mean_vehicle_wait"], ep_elig_sum / ep_steps]
            row += [ep_metrics[k] / ep_steps for k in agent.metric_names]
            writer.writerow(row); log_file.flush()
            el = time.time() - t0
            eta = el / max(step - start_step, 1) * (end_step - step)
            print(f"step {step:>7d} | ep {episode:>4d} | reward {ep_reward:9.1f} | mean wait/step {ep_wait_sum / ep_steps:8.1f} | "
                  + " | ".join(f"{k} {ep_metrics[k] / ep_steps:.4g}" for k in agent.metric_names)
                  + f" | {el / 60:6.1f} min elapsed, ETA {eta / 3600:5.2f} h", flush=True)
            episode += 1
            ep_reward, ep_wait_sum, ep_elig_sum, ep_steps = 0.0, 0.0, 0.0, 0
            ep_metrics = {k: 0.0 for k in agent.metric_names}
            if step < end_step:
                obs, _ = env.reset(seed=training_env_seed(cfg.seed, episode))

        if step % cfg.checkpoint_every == 0 and step < cfg.total_env_steps:
            agent.save_policy(out_dir, f"step{step}")
            torch.save({"agent": agent.state_dict(), "step": step, "episode": episode},
                       os.path.join(out_dir, f"state_step{step}.pt"))

    finished = end_step >= cfg.total_env_steps
    tag = "final" if finished else "latest"
    agent.save_policy(out_dir, tag)
    torch.save({"agent": agent.state_dict(), "step": end_step, "episode": episode}, os.path.join(out_dir, f"state_{tag}.pt"))
    with open(os.path.join(out_dir, "resume_state.json"), "w") as f:
        json.dump({"step": end_step, "episode": episode, "done_training": finished}, f)
    log_file.close()
    env.close()
    print(f"Episodes completed: {episode} (expected {cfg.total_env_steps // steps_per_episode} for a full run)")
    print(f"Chunk complete: ran steps {start_step + 1}..{end_step} of {cfg.total_env_steps}. Checkpoints: "
          f"{os.path.join(out_dir, f'actor_{tag}.pt')}, {os.path.join(out_dir, f'critic_{tag}.pt')}")
    return out_dir


def cli_run(algo: str, config_cls, make_agent, project_root: str, default_seed: int = 0, default_episodes: int = 500,
            default_out_name: str | None = None, extra_args=None, cfg_kwargs=None):
    """Shared command line of ac/train.py and ca/train.py (same flags as dqn/train.py)."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--net_file", default=os.path.join(project_root, "sumo_4x4_network", "grid4x4.net.xml"))
    p.add_argument("--route_file", default=os.path.join(project_root, "sumo_4x4_network", "routes.rou.xml"))
    p.add_argument("--out", default=None, help=f"default: runs/{default_out_name or algo}_seed<seed>")
    p.add_argument("--episodes", type=int, default=default_episodes)
    p.add_argument("--steps", type=int, default=None, help="raw total step count (overrides --episodes)")
    p.add_argument("--seed", type=int, default=None, help="training seed")
    p.add_argument("--chunk_steps", type=int, default=None, help="run only this many steps, then save and exit (use multiples of 720)")
    p.add_argument("--resume", default=None, help="state_step<N>.pt / state_latest.pt to resume from")
    p.add_argument("--gui", action="store_true")
    if extra_args:
        extra_args(p)
    args, _ = p.parse_known_args()
    cfg = config_cls(**(cfg_kwargs(args) if cfg_kwargs else {}))
    cfg.seed = args.seed if args.seed is not None else default_seed
    out_dir = args.out or os.path.join(project_root, "runs", f"{default_out_name or algo}_seed{cfg.seed}")
    if not args.resume and os.path.exists(os.path.join(out_dir, "actor_final.pt")):
        sys.exit(f"STOP: {out_dir} already contains a finished run (actor_final.pt).\n"
                 f"Change DEFAULT_SEED (or pass --seed/--out), or delete that folder if you really want to retrain.")
    if args.steps is not None:
        cfg.total_env_steps = args.steps
    else:
        cfg.total_env_steps = args.episodes * (cfg.episode_seconds // cfg.decision_interval)
        print(f"Training for {args.episodes} episodes -> {cfg.total_env_steps} total steps")
    return train_online(make_agent, cfg, args.net_file, args.route_file, out_dir, algo=algo, use_gui=args.gui,
                        resume_from=args.resume, chunk_steps=args.chunk_steps)