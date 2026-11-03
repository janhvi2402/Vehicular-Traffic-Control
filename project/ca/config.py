"""
ca/config.py
============
Hyperparameters of the AVERAGE-REWARD two-timescale CRITIC-ACTOR (CA), Panda & Bhatnagar (AAAI-25), "Two-Timescale
Critic-Actor for Average Reward MDPs with Function Approximation" (Algorithm 1). Replaces the earlier ca_avg/ folder.

Three recursions per iterate t, all driven by the per-junction rewards:
  TD error     delta_t = r_t - L_t + V(s_{t+1}) - V(s_t)            (no discount factor)
  avg reward   L_{t+1} = L_t + gamma_t * (r_t - L_t)                 (gamma_t = K * alpha_t, same timescale as the actor)
  actor        theta  += alpha_t * delta_t * grad log pi(a_t|s_t)    (FAST)
  critic       w      -= beta_t * grad (delta_t^2 / 2)               (SLOW)
Step sizes alpha_t = c / (1 + n)^nu, beta_t = c / (1 + n)^sigma with 0 < nu < sigma < 1 (paper: nu = 0.5, sigma = 0.51), n the
decay index (floor(step / decay_every)), as in ac/config.py. NOTE: with nu = 0.5 and sigma = 0.51 the two step sizes differ by
only ~5 % after 125 decays, so the timescale separation is weak at this training length. That is the paper's recommended pair
(best sample complexity), kept as the default. The paper's conditions are 0 < nu < sigma < 1, 2*sigma < 3*nu and 2*sigma - nu < 1,
so with nu = 0.5 any sigma in (0.5, 0.75) is allowed (e.g. 0.7 for a clearly separated variant); sigma = 0.75 or larger is NOT.

Same fair-comparison settings as ac/config.py, dqn/config.py, ppo/config.py: (128, 128) tanh MLPs on the 11-dim per-junction input,
parameter sharing over junctions, reward_clip 50, 360,000 env steps (500 episodes), switch votes masked by eligibility, truncation
at the 3600 s limit bootstraps (the average-reward setting is a continuing task). Deviations from the paper: neural instead of
linear critic, entropy bonus (not in the paper), a separate L per junction.
"""
from dataclasses import dataclass


@dataclass
class CAConfig:
    input_dim: int = 11
    hidden_sizes: tuple = (128, 128)
    n_actions: int = 2

    actor_lr0: float = 3e-4
    critic_lr0: float = 3e-4
    avg_reward_k: float = 1.0        # gamma_t = K * alpha_t
    actor_exponent: float = 0.50     # nu  (fast)
    critic_exponent: float = 0.51    # sigma (slow); must be > actor_exponent
    lr_decay_every_fraction: float = 0.008

    entropy_coeff: float = 0.01      # same as ppo / ac; NOT in the paper
    reward_clip: float = 50.0        # same as dqn / ppo / ac
    # (no gamma: the average-reward TD error has no discount factor)

    total_env_steps: int = 360_000
    episode_seconds: int = 3600
    decision_interval: int = 5
    seed: int = 0

    log_every: int = 720
    checkpoint_every: int = 18_000

    def __post_init__(self):
        nu, sigma = self.actor_exponent, self.critic_exponent
        if not (0 < nu < sigma < 1):
            raise ValueError("CA needs 0 < actor_exponent (nu) < critic_exponent (sigma) < 1")
        if not (2 * sigma < 3 * nu and 2 * sigma - nu < 1):          # Panda & Bhatnagar, Sample Complexity Results
            raise ValueError(f"CA needs 2*sigma < 3*nu and 2*sigma - nu < 1 (got nu={nu}, sigma={sigma})")
