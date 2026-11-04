"""
ac/agent.py -- the two-timescale actor-critic learning rule (Bhatnagar, Borkar & Guin, Eqs. 7-8, neural function approximation).
One call of update() = one iterate n of the paper's recursions, applied to the 16 junction transitions of one env step.

  delta_i = r_i + gamma * V(s'_i) * (1 - terminated) - V(s_i)          (r_i = clipped per-junction reward)
  critic:  minimise mean(delta^2) with step size a(n)                  (Eq. 7)
  actor:   ascend  mean(log pi(a_i|s_i) * delta_i) + entropy bonus     (Eq. 8; delta detached)
Both networks are updated at every step; the two timescales come only from the step-size exponents:
variant "ac": critic exponent = fast (smaller), actor = slow; variant "ca": the other way round.
The sign convention: the paper minimises cost; here rewards are maximised, so the actor ASCENDS delta (reward TD error).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ac.config import ACConfig
from ac.network import ActorNetwork, CriticNetwork
from environment.online_trainer import set_lr, step_size


def assign_exponents(cfg: ACConfig):
    """(critic_exponent, actor_exponent): the SMALLER exponent goes to the faster recursion."""
    if cfg.variant == "ac":
        return cfg.fast_exponent, cfg.slow_exponent
    if cfg.variant == "ca":
        return cfg.slow_exponent, cfg.fast_exponent
    raise ValueError(f"Unknown variant {cfg.variant!r} (expected 'ac' or 'ca')")


class ACAgent:
    metric_names = ["critic_loss", "actor_loss", "entropy", "abs_td_error", "critic_lr", "actor_lr"]

    def __init__(self, cfg: ACConfig, device: str):
        self.cfg, self.device = cfg, device
        self.actor = ActorNetwork(cfg.input_dim, cfg.hidden_sizes, cfg.n_actions).to(device)
        self.critic = CriticNetwork(cfg.input_dim, cfg.hidden_sizes).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr0)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr0)
        self.critic_exp, self.actor_exp = assign_exponents(cfg)
        self.decay_every = max(1, int(cfg.lr_decay_every_fraction * cfg.total_env_steps))
        print(f"{cfg.variant}: critic exponent {self.critic_exp} (a(n)), actor exponent {self.actor_exp} (b(n)) "
              f"[smaller = faster]; step size decays every {self.decay_every} steps")
        self._x = self._logp = self._ent = self._v = None

    def act(self, feats: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        self._x = torch.as_tensor(feats, dtype=torch.float32, device=self.device)
        el = torch.as_tensor(eligible, dtype=torch.bool, device=self.device)
        actions, self._logp, self._ent = self.actor.act(self._x, el)
        self._v = self.critic(self._x)
        return actions.cpu().numpy().astype(np.int64)

    def update(self, next_feats: np.ndarray, rewards: np.ndarray, terminated: bool, step: int) -> dict:
        cfg = self.cfg
        c_lr = step_size(step, cfg.critic_lr0, self.critic_exp, self.decay_every)
        a_lr = step_size(step, cfg.actor_lr0, self.actor_exp, self.decay_every)
        set_lr(self.critic_opt, c_lr)
        set_lr(self.actor_opt, a_lr)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v_next = torch.zeros_like(self._v) if terminated else \
                self.critic(torch.as_tensor(next_feats, dtype=torch.float32, device=self.device))
        td = r + cfg.gamma * v_next - self._v

        critic_loss = td.pow(2).mean()
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -(self._logp * td.detach()).mean() - cfg.entropy_coeff * self._ent.mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(), "entropy": self._ent.mean().item(),
                "abs_td_error": td.detach().abs().mean().item(), "critic_lr": c_lr, "actor_lr": a_lr}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict()}

    def load_state_dict(self, d: dict):
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
        self.actor_opt.load_state_dict(d["actor_opt"]); self.critic_opt.load_state_dict(d["critic_opt"])

    def save_policy(self, out_dir: str, tag: str):
        torch.save(self.actor.state_dict(), os.path.join(out_dir, f"actor_{tag}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(out_dir, f"critic_{tag}.pt"))
