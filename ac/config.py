"""
ac/config.py
============
Hyperparameters for the two-timescale Actor-Critic (AC) and Critic-Actor (CA)
algorithms, justified against Bhatnagar, Borkar & Guin, 2024, "Actor-Critic
or Critic-Actor? A Tale of Two Time Scales" (arXiv:2210.04470v6).

Core idea of the paper (Sections I-III): both AC and CA are two-timescale
stochastic approximations of the SAME two recursions -- a critic update for
V(i) governed by step-size sequence a(n), and an actor update for the policy
parameter theta(i,a) governed by step-size sequence b(n) (Eqs. 7-8). The
ONLY difference between the two algorithms is which recursion is on the
faster time scale:

  - Standard Actor-Critic (AC): critic (V) is faster, i.e. b(n) = o(a(n)).
    The critic "sees the actor as quasi-static" and tracks V_pi for the
    current (slowly-changing) policy -- this emulates POLICY ITERATION.
  - Critic-Actor (CA), the paper's proposed reversal: actor (theta) is
    faster, i.e. a(n) = o(b(n)). The critic now "sees the actor as
    quasi-static" from the OTHER side -- the actor greedily tracks the
    current (slowly-changing) V, which the paper proves emulates VALUE
    ITERATION instead (Section III/IV).

The paper's own step-size convention (Section V numerical results) is
   a(n) = lr0 / (floor(n / decay_every) + 1) ** exponent_a
   b(n) = lr0 / (floor(n / decay_every) + 1) ** exponent_b
with two-timescale separation achieved purely through the RELATIVE decay
exponents (a smaller exponent => step size shrinks more slowly => that
recursion effectively moves on the faster timescale, since its increments
stay non-negligible for longer relative to the other one). We copy this
scheme exactly (see train.py), rather than inventing a different
mechanism (e.g. differing update frequencies), so that AC and CA differ
ONLY in which exponent is smaller -- exactly the paper's own definition
of the two algorithms.

Paper's own reference step-size values, Fig. 1 (tabular): a(n) ~ n^-1,
b(n) ~ n^-0.55 (fast a / slow b => AC-like since here it's b(n)=o(a(n))
... note the paper's Fig. 1 uses THIS AS THE STANDARD AC EXAMPLE, and
Fig. 9 (neural-network function approximation, the closest precedent to
our setting) uses exponents alpha=0.55, beta=1 for AC and CA symmetrically
in their sweep -- we default to a comparable spread (0.55 / 0.75) that is
close to the paper's own function-approximation-regime choices while
being asymmetric enough for the timescale separation to be meaningful
over our much shorter training budget.
"""

from dataclasses import dataclass


@dataclass
class ACConfig:
    # ---------------------------------------------------------------- #
    # Network architecture
    # ---------------------------------------------------------------- #
    # Same input/action space as dqn/config.py and ppo/config.py, for a
    # fair three-way (DQN vs PPO vs AC/CA) algorithm comparison rather
    # than an architecture comparison. Unlike ppo/actor_critic.py's
    # ActorCritic (ONE shared-trunk network), AC/CA here use two SEPARATE
    # networks (see network.py) with their own optimizers -- this is
    # required by the algorithm itself: the whole point of AC vs CA is
    # that the critic and actor update on genuinely different timescales,
    # which is meaningless if they share parameters/one optimizer.
    input_dim: int = 11          # TrafficGridEnv.NUM_FEATURES, one junction's local obs
    hidden_sizes: tuple = (128, 128)
    n_actions: int = 2           # hold / switch, per junction

    # ---------------------------------------------------------------- #
    # Algorithm variant
    # ---------------------------------------------------------------- #
    # "ac": standard actor-critic (critic faster, Konda & Borkar's
    #       Algorithm 3, emulates policy iteration).
    # "ca": critic-actor (actor faster, this paper's proposed reversal,
    #       emulates value iteration; Eqs. 7-8, Section III).
    variant: str = "ac"

    # ---------------------------------------------------------------- #
    # Two-timescale step sizes (paper's a(n) / b(n), Section V)
    # ---------------------------------------------------------------- #
    # Base learning rates before decay (paper's neural-network experiments,
    # Fig. 9, use 0.01 as the base constant for both recursions -- we use
    # a more standard Adam-appropriate 3e-4 base for our much larger,
    # ReLU-MLP-on-11-dim-features network, matching ppo/config.py's own
    # learning_rate, and let the EXPONENTS (not the base rate) carry the
    # timescale separation, exactly as the paper's own scheme does).
    critic_lr0: float = 3e-4     # base rate for a(n), the V-update
    actor_lr0: float = 3e-4      # base rate for b(n), the theta-update

    # Decay exponents. The SMALLER exponent's step size shrinks more
    # slowly and so stays non-negligible longer -- that recursion is the
    # "fast" one that the other treats as quasi-static/quasi-equilibrated
    # (paper, Section I). train.py assigns these two values to
    # (critic_exponent, actor_exponent) according to `variant`:
    #   variant == "ac": critic gets `fast_exponent`, actor gets `slow_exponent`
    #   variant == "ca": actor gets `fast_exponent`,  critic gets `slow_exponent`
    fast_exponent: float = 0.55  # paper's Fig. 9 alpha=0.55 reference point
    slow_exponent: float = 0.75  # a wider gap than the paper's own 0.55/1.0
                                  # sweep would give at this scale, chosen
                                  # so the timescale separation is still
                                  # visible over our much shorter run
                                  # (~tens of thousands, not 1e8, steps)
    # Env steps between step-size decrements, expressed as a FRACTION of
    # total_env_steps rather than a fixed count -- same scaling technique
    # dqn/config.py uses for epsilon_decay_fraction. This matters
    # concretely for this project: the earlier 2-junction work trained for
    # 50,000 steps, this 4x4-grid run defaults to 25,000 -- a fixed
    # lr_decay_every=200 would give the 50k run 250 decay steps but the
    # 25k run only 125, silently changing how much the step sizes have
    # shrunk (and thus how separated the two timescales are) by the end of
    # training just because total_env_steps changed. Deriving decay_every
    # from the fraction below keeps the SAME number of decay steps (and
    # the same final step-size ratio between fast/slow) regardless of
    # total_env_steps, so --steps 50000 reproduces the earlier project's
    # schedule shape rather than a stretched-out version of the 25k one.
    lr_decay_every_fraction: float = 0.008  # -> decay_every=200 at 25k steps,
                                             #    decay_every=400 at 50k steps

    # ---------------------------------------------------------------- #
    # RL problem parameters
    # ---------------------------------------------------------------- #
    gamma: float = 0.99          # paper: discounted-cost formulation, Eq. 1
    entropy_coeff: float = 0.01  # NOT in the tabular paper (which uses an
                                  # exact softmax/Boltzmann policy with a
                                  # bounded parameter theta0, Eq. 6) -- for
                                  # neural function approximation we add a
                                  # small entropy bonus (same role/value as
                                  # ppo/config.py's entropy_coeff) purely to
                                  # prevent premature policy collapse, since
                                  # we have no theta0 projection step to
                                  # keep the softmax from saturating.
    reward_clip: float = 50.0    # same soft-clipping rationale as DQN/PPO configs

    # ---------------------------------------------------------------- #
    # Training length / episode
    # ---------------------------------------------------------------- #
    # Both AC and CA here are ONLINE, one-step algorithms (Eqs. 7-8 use a
    # single freshly-sampled transition per update, no replay buffer and
    # no rollout horizon) -- so "step" below means one env.step() call,
    # directly comparable to DQN's total_env_steps and unlike PPO's
    # total_updates*horizon.
    total_env_steps: int = 50_000  # matches DQN's 50,000 (dqn/config.py) and PPO's
                                    # ~50,048 (391*128, ppo/config.py) -- the same
                                    # step budget used in the earlier 2-junction
                                    # project. lr_decay_every_fraction above means
                                    # decay_every auto-scales to 400 here (was 200
                                    # at 25,000) with no other change needed.
    episode_seconds: int = 3600
    decision_interval: int = 5
    seed: int = 0

    # logging / checkpointing
    log_every: int = 200
    checkpoint_every: int = 5_000
