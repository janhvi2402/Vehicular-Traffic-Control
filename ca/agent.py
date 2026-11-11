"""
ca/agent.py -- average-reward two-timescale critic-actor (Panda & Bhatnagar, Algorithm 1) with neural function approximation.
Per env step, for the 16 junction transitions (parameter-shared networks, one running average-reward estimate L_i per junction):
  delta_i = r_i - L_i + V(s'_i) * (1 - terminated) - V(s_i)      (uses L BEFORE its own update, as in the paper)
  L_i    <- L_i + gamma_t * (r_i - L_i)                           gamma_t = K * alpha_t
  critic: minimise mean(delta^2), step size beta_t (slow)         actor: ascend mean(log pi * delta) + entropy bonus, step alpha_t (fast)
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ca.config import CAConfig
from ac.network import ActorNetwork, CriticNetwork
from environment.online_trainer import set_lr, step_size


class CAAgent:
    metric_names = ["critic_loss", "actor_loss", "entropy", "abs_td_error", "mean_avg_reward", "critic_lr", "actor_lr"]

    def __init__(self, cfg: CAConfig, device: str, n_agents: int = 16):
        self.cfg, self.device = cfg, device
        self.actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
        self.critic = CriticNetwork(cfg.input_dim, cfg.hidden_sizes).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr0)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr0)
        self.decay_every = max(1, int(cfg.lr_decay_every_fraction * cfg.total_env_steps))
        self.avg_reward = None            # (n_agents,) float32, created on first act() from the observation shape
        print(f"ca: actor exponent nu={cfg.actor_exponent} (fast, also drives L), critic exponent sigma={cfg.critic_exponent} (slow); "
              f"step size decays every {self.decay_every} steps")
        self._x = self._logp = self._ent = self._v = None

    def act(self, feats: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        if self.avg_reward is None:
            self.avg_reward = np.zeros(feats.shape[0], dtype=np.float32)
        self._x = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        el = torch.as_tensor(eligible, dtype=torch.bool, device=self.device)
        actions, self._logp, self._ent = self.actor.act(self._x, el)
        self._v = self.critic(self._x)
        return actions.cpu().numpy().astype(np.int64)

    def update(self, next_feats: np.ndarray, rewards: np.ndarray, terminated: bool, step: int) -> dict:
        cfg = self.cfg
        a_lr = step_size(step, cfg.actor_lr0, cfg.actor_exponent, self.decay_every)
        c_lr = step_size(step, cfg.critic_lr0, cfg.critic_exponent, self.decay_every)
        g_lr = cfg.avg_reward_k * a_lr
        set_lr(self.actor_opt, a_lr)
        set_lr(self.critic_opt, c_lr)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        L = torch.as_tensor(self.avg_reward, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_next = torch.zeros_like(self._v) if terminated else \
                self.critic(torch.as_tensor(next_feats, dtype=torch.float32, device=self.device))
        td = r - L + v_next - self._v

        critic_loss = td.pow(2).mean()
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -(self._logp * td.detach()).mean() - cfg.entropy_coeff * self._ent.mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        self.avg_reward = self.avg_reward + g_lr * (rewards - self.avg_reward)       # L_{t+1}, after the TD error used L_t
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "entropy": self._ent.mean().item(),
                "abs_td_error": td.detach().abs().mean().item(), "mean_avg_reward": float(self.avg_reward.mean()),
                "critic_lr": c_lr, "actor_lr": a_lr}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "avg_reward": None if self.avg_reward is None else torch.as_tensor(self.avg_reward.copy())}

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
        self.actor_opt.load_state_dict(d["actor_opt"]); self.critic_opt.load_state_dict(d["critic_opt"])
        a = d["avg_reward"]
        self.avg_reward = None if a is None else np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float32)

    def save_policy(self, out_dir: str, tag: str):
        torch.save(self.actor.state_dict(), os.path.join(out_dir, f"actor_{tag}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(out_dir, f"critic_{tag}.pt"))
        if self.avg_reward is not None:
            np.save(os.path.join(out_dir, f"avg_reward_{tag}.npy"), self.avg_reward)