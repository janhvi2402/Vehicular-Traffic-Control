"""
ca_avg/config.py
=================
Hyperparameters for the AVERAGE-REWARD critic-actor (CA) algorithm of
Panda & Bhatnagar, 2025, "Two-Timescale Critic-Actor for Average Reward
MDPs with Function Approximation" (AAAI-25). This is a DIFFERENT
algorithm from the one in ac/config.py (Bhatnagar, Borkar & Guin, 2023,
"Actor-Critic or Critic-Actor? A Tale of Two Time Scales") -- do not
confuse the two "CA"s:

  ac/ (variant="ca")          ca_avg/ (this module)
  ---------------------       ----------------------
  Discounted-cost setting     Long-run AVERAGE-reward setting (no gamma)
  2 recursions: critic, actor 3 recursions: critic, actor, AND an average-
                               reward estimate L(theta) -- Algorithm 1's
                               L_{t+1} = L_t + gamma_t*(r_t - L_t)
  Either recursion can be     Critic is ALWAYS the slow recursion here;
  "fast" (--variant ac/ca     actor + average-reward estimate are ALWAYS
  toggles it)                 together on the fast timescale (paper,
                               Section I: "the average reward and actor
                               recursions together proceed on the same
                               timescale that is faster than the timescale
                               of the critic update") -- no ac/ca toggle,
                               this module only implements the CA
                               direction, which is the paper's proposal
  Actor update uses           Actor update uses the average-reward TD
  discounted TD error:        error: delta = r - L + V(next) - V(current)
  delta = r - g + gam*V(next) (paper's Eq. in Algorithm 1, no discount
  - V(current)                factor at all -- gamma has no role here)

The paper's own policy-gradient framing is already REWARD-MAXIMIZING
(Section II: "Our aim is to maximise the long-run average reward"), unlike
Bhatnagar/Borkar/Guin 2023's COST-minimizing framing (which is why
ac/train.py has to negate the sign of its actor loss to convert cost-min
to reward-max). Here no sign flip is needed -- Algorithm 1's
theta_{t+1} = theta_t + alpha_t*delta_t*grad(log pi) is already exactly
reward-maximizing gradient ascent as written.

Paper's step-size family (Section "Sample Complexity Results"):
    alpha_t = c_alpha / (1+t)^nu     (actor)
    beta_t  = c_beta  / (1+t)^sigma  (critic)
    gamma_t = c_gamma / (1+t)^nu     (average-reward estimate, SAME
                                       exponent nu as the actor -- the
                                       paper ties them together via
                                       gamma_t = K*alpha_t for some K>0,
                                       i.e. a constant ratio c_gamma/c_alpha,
                                       not an independently-decaying term)
with 0 < nu < sigma < 1 required (critic's exponent sigma is LARGER, so
beta_t shrinks faster / stays non-negligible for less time -- that's what
makes the critic the SLOW recursion here, same "smaller exponent = faster
timescale" convention used in ac/config.py's assign_exponents()).

The paper explicitly cites nu=0.5, sigma=0.51 as achieving its best
demonstrated sample complexity (~O(eps^-2.08)); we default to exactly
that pair.

Deviations from the paper, same spirit as every other config.py in this
project:
  - Paper's theoretical results use LINEAR function approximation for the
    critic (Assumption 4.1 etc. are stated in terms of a linear feature
    map phi). We use a small MLP instead (ActorNetwork/CriticNetwork,
    reused directly from ac/network.py -- same architecture requirement:
    map one junction's 11-dim obs to policy logits / a scalar value),
    the same neural generalization dqn/config.py and ac/config.py already
    make from their own papers' tabular/linear settings.
  - The paper has no entropy bonus (its policy is an exact Boltzmann/
    softmax parameterization with theoretical guarantees, Eq. 6). For a
    neural policy we add a small entropy bonus, same rationale as
    ac/config.py's entropy_coeff: prevent premature policy collapse with
    no theta0 projection step to guard against it.
  - n_agents (16 on 4x4 grid) independent per-junction recursions,
    parameter-shared, matching every other algorithm in this project --
    the paper's single-agent MDP framework is applied independently per
    junction the same way dqn/ppo/ac already do.
"""

from dataclasses import dataclass


@dataclass
class CAAvgConfig:
    # ---------------------------------------------------------------- #
    # Network architecture -- identical to ac/network.py's
    # ActorNetwork/CriticNetwork, reused directly (not redefined here)
    # ---------------------------------------------------------------- #
    input_dim: int = 11          # TrafficGridEnv.NUM_FEATURES, one junction's local obs
    hidden_sizes: tuple = (128, 128)
    n_actions: int = 2           # hold / switch, per junction

    # ---------------------------------------------------------------- #
    # Three-recursion step sizes (paper's alpha(n)/beta(n)/gamma(n))
    # ---------------------------------------------------------------- #
    actor_lr0: float = 3e-4      # base rate for alpha(n), the theta-update
    critic_lr0: float = 3e-4     # base rate for beta(n), the V-update
    avg_reward_k: float = 1.0    # gamma_t = avg_reward_k * alpha_t (paper's
                                  # gamma_t = K*alpha_t, Section "Two-Timescale
                                  # Critic-Actor Algorithm"); K=1 means the
                                  # average-reward estimate and the actor use
                                  # literally the same step size, satisfying
                                  # the paper's requirement that they share
                                  # one (fast) timescale exactly

    # Decay exponents: nu (actor + avg-reward, FAST) must be strictly
    # smaller than sigma (critic, SLOW) -- paper requires 0 < nu < sigma < 1,
    # 2*sigma < 3*nu, 2*sigma - nu < 1. Defaults are the paper's own
    # cited near-optimal choice (see module docstring).
    actor_exponent: float = 0.50   # nu
    critic_exponent: float = 0.51  # sigma

    # Env steps between step-size decrements, as a FRACTION of
    # total_env_steps -- same scaling technique as ac/config.py's
    # lr_decay_every_fraction, for the same reason: keeps the schedule's
    # shape (number of decay steps, final step-size ratio) invariant to
    # how long you choose to train, so --steps 100000 doesn't silently
    # give a differently-shaped schedule than --steps 50000.
    lr_decay_every_fraction: float = 0.008

    # ---------------------------------------------------------------- #
    # RL problem parameters
    # ---------------------------------------------------------------- #
    # NOTE: no `gamma` (discount factor) field -- this is deliberate. The
    # average-reward formulation has no discount factor at all; the TD
    # error in train.py is r - L + V(next) - V(current), never
    # r + gamma*V(next) - V(current). Adding a gamma field here would
    # invite silently reintroducing a discounted-cost TD error by mistake.
    entropy_coeff: float = 0.01  # NOT in the paper -- see module docstring
    reward_clip: float = 50.0    # same soft-clipping rationale as every
                                  # other config.py in this project

    # ---------------------------------------------------------------- #
    # Training length / episode
    # ---------------------------------------------------------------- #
    # This is an ONLINE, one-step algorithm (Algorithm 1 uses a single
    # freshly-sampled transition per update, no replay buffer, no rollout
    # horizon) -- directly comparable to DQN's total_env_steps and
    # ac/config.py's total_env_steps, unlike PPO's total_updates*horizon.
    total_env_steps: int = 50_000  # matches ac/config.py's default, itself
                                    # matched to DQN's 50,000 / PPO's ~50,048
                                    # -- same step budget across all four
                                    # algorithms for a fair comparison
    episode_seconds: int = 3600
    decision_interval: int = 5
    seed: int = 0

    # logging / checkpointing
    log_every: int = 200
    checkpoint_every: int = 5_000
