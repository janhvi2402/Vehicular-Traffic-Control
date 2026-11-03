"""
ac/config.py
============
Hyperparameters of the two-timescale ACTOR-CRITIC (AC) agent, Bhatnagar, Borkar & Guin, "Actor-Critic or
Critic-Actor? A Tale of Two Time Scales" (arXiv:2210.04470): discounted setting, critic on the FAST timescale
(its step size decays more slowly), actor on the SLOW one. `variant="ca"` swaps the two timescales (the paper's
critic-actor, discounted). That variant is kept only as an ablation: the CA of this study is the average-reward
critic-actor of Panda & Bhatnagar (AAAI-25) in ca/.

Step sizes (paper Sec. V): lr0 / (floor(step / decay_every) + 1) ** exponent, with the SMALLER exponent on the
faster recursion. decay_every = lr_decay_every_fraction * total_env_steps, so the schedule shape does not depend on
the training length.

Fair-comparison settings (identical to dqn/config.py and ppo/config.py): 11-dim per-junction input, hidden (128, 128),
one parameter-shared network applied to every junction, gamma 0.99, reward_clip 50, decision_interval 5 s,
360,000 env steps = 500 episodes, checkpoint every 25 episodes, switch votes masked by environment.eligibility.
Known asymmetries (disclosed in docs/LIMITATIONS.md): AC/CA are one-step online learners (one gradient step on the
16 junction transitions per env step, no replay, no GAE, no advantage normalisation), tanh MLPs like PPO (DQN: ReLU),
Adam with the paper's decaying step sizes instead of a constant learning rate.
"""
from dataclasses import dataclass


@dataclass
class ACConfig:
    # ---- network (same as dqn/ppo) ----
    input_dim: int = 11
    hidden_sizes: tuple = (128, 128)
    n_actions: int = 2

    # ---- algorithm ----
    variant: str = "ac"            # "ac" = critic fast (this study); "ca" = actor fast (discounted ablation only)
    gamma: float = 0.99            # same as dqn/ppo
    entropy_coeff: float = 0.01    # same as ppo (policy entropy bonus; averaged over all junction rows like PPO)
    reward_clip: float = 50.0      # same as dqn/ppo

    # ---- two-timescale step sizes ----
    critic_lr0: float = 3e-4       # same base rate as ppo.learning_rate
    actor_lr0: float = 3e-4
    fast_exponent: float = 0.55
    slow_exponent: float = 0.75
    lr_decay_every_fraction: float = 0.008   # 360,000 steps -> decay every 2,880 steps (4 episodes), 125 decays

    # ---- budget / environment (matched to dqn/ppo) ----
    total_env_steps: int = 360_000          # 500 episodes of 720 decision steps
    episode_seconds: int = 3600
    decision_interval: int = 5
    seed: int = 0

    # ---- logging / checkpointing (episode aligned, like dqn/ppo) ----
    log_every: int = 720
    checkpoint_every: int = 18_000          # every 25 episodes
