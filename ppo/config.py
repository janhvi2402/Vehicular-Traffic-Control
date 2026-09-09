"""
ppo/config.py
=============
All PPO hyperparameters in one place, justified against Schulman et al.,
2017 ("Proximal Policy Optimization Algorithms"). Since our action space
is discrete (2 choices per junction: hold/switch), the paper's ATARI
hyperparameters (Table 5) are the relevant reference, not the MuJoCo table
(Table 3, continuous Gaussian-policy control) or the Roboschool table
(Table 4, also continuous). Where a value is copied from Table 5 we say
so; where it's scaled or changed for our task we explain why.

Paper reference values, Atari (Table 5), for context:
    horizon (T)............. 128
    Adam stepsize............ 2.5e-4 x alpha (alpha linearly annealed 1->0)
    num. epochs.............. 3
    minibatch size........... 32 x 8 = 256
    discount (gamma)......... 0.99
    GAE parameter (lambda)... 0.95
    number of actors......... 8
    clipping parameter eps... 0.1 x alpha
    VF coeff. c1.............  1
    entropy coeff. c2........ 0.01

Also referenced: Table 1's clip-epsilon sweep on the MuJoCo benchmark,
which found epsilon=0.2 the best-performing fixed value overall (bolded
in the paper) -- we use this un-annealed value rather than Atari's
annealed 0.1, since our training run is far shorter than the paper's and
we don't want the clip range shrinking to near-zero before the policy has
had a chance to learn (see epsilon_end below for the one place annealing
is still used).
"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    # ---------------------------------------------------------------- #
    # Network architecture
    # ---------------------------------------------------------------- #
    # Same reasoning as dqn/config.py: our input is an 11-dim hand-engineered
    # feature vector, not raw pixels, so no CNN is needed. We use a shared
    # trunk (matching DQN's hidden_sizes exactly, for a fair side-by-side
    # comparison of algorithms rather than architectures) with two small
    # heads: a policy head (2 logits: hold/switch) and a value head (1
    # scalar). The paper explicitly allows -- and for compute-shared
    # settings recommends -- combining policy and value into one network
    # with a combined loss (Eq. 9); we do exactly that.
    input_dim: int = 11          # TrafficGridEnv.NUM_FEATURES, one junction's local obs
    hidden_sizes: tuple = (128, 128)
    n_actions: int = 2           # hold / switch, per junction

    # ---------------------------------------------------------------- #
    # Rollout / actors
    # ---------------------------------------------------------------- #
    # Paper's Algorithm 1: N (parallel) actors each collect T timesteps,
    # producing NT samples per update. We treat our 16 (or 9, on the 3x3
    # generalization network) junctions as the paper's N actors -- each
    # junction's local observation/reward stream is its own independent
    # trajectory under the shared policy, exactly matching the paper's
    # "parameter-shared parallel actors" structure, except our N actors
    # live inside ONE environment instance instead of N copies of it.
    # horizon T=128 is copied directly from the paper's Atari setting.
    horizon: int = 128           # env steps collected per actor before each PPO update
    # n_actors is NOT set here -- it equals env.n_agents at runtime (16 for
    # the 4x4 grid, 9 for the 3x3 generalization network), so the buffer
    # size (n_actors * horizon) adapts automatically to whichever network
    # train.py/evaluate.py is pointed at.

    # ---------------------------------------------------------------- #
    # PPO update
    # ---------------------------------------------------------------- #
    gamma: float = 0.99          # paper: unchanged across every table
    gae_lambda: float = 0.95     # paper: unchanged across every table
    clip_epsilon: float = 0.2    # paper: Table 1's best fixed-epsilon result (bolded, MuJoCo sweep)
    value_coeff: float = 1.0     # paper: Atari's c1 (Table 5), Eq. 9
    entropy_coeff: float = 0.01  # paper: Atari's c2 (Table 5) -- our action space is discrete
                                  # (categorical), like Atari, unlike the continuous MuJoCo/
                                  # Roboschool tasks which use a Gaussian's std for exploration
                                  # instead of an entropy bonus.
    max_grad_norm: float = 0.5   # standard PPO implementation detail (not tabulated in the
                                  # paper itself, but universal in reference implementations,
                                  # e.g. OpenAI Baselines) -- clips the global gradient norm to
                                  # prevent rare destructive updates; kept since our reward
                                  # scale (traffic waiting-time deltas) is far less bounded
                                  # than a clipped Atari game score.

    # Paper Atari: 3 epochs, minibatch 32x8=256 (out of NT=128x8=1024, i.e.
    # 4 minibatches/epoch). Our NT is n_agents*horizon (2048 on the 4x4
    # grid, 1152 on the 3x3 grid) -- larger than Atari's per-update batch
    # since we have more "actors". We use slightly more epochs (4) since
    # our per-junction observation is a compact hand-engineered vector
    # (far less prone to overfitting within an update than raw pixels),
    # and a similarly-sized minibatch (256) so the number of gradient
    # steps per update stays comparable to the paper's Atari setting.
    num_epochs: int = 4
    minibatch_size: int = 256

    # Paper Atari: Adam stepsize 2.5e-4, linearly annealed to 0 via alpha.
    # Paper MuJoCo (Table 3): fixed 3e-4, no annealing, on a similarly
    # small 2-hidden-layer-64-unit MLP -- much closer to our network size
    # than the Atari CNN. We use the MuJoCo table's fixed learning rate
    # since it's the paper's own prescription for a small-MLP setting,
    # rather than annealing Atari's CNN-tuned rate to zero over our much
    # shorter training run.
    learning_rate: float = 3e-4

    # ---------------------------------------------------------------- #
    # Reward handling
    # ---------------------------------------------------------------- #
    # Same reasoning as DQN: we train on a single task, not 49 heterogeneous
    # Atari games, so there's no cross-task motivation for hard reward
    # clipping. We keep the same soft clipping band as DQN for consistency
    # between the two algorithms' reported results.
    reward_clip: float = 50.0

    # ---------------------------------------------------------------- #
    # Training length / episode
    # ---------------------------------------------------------------- #
    # total_updates * horizon = total env steps actually simulated.
    # Chosen so total env steps roughly matches the DQN validation run
    # (25,000 env steps) for a like-for-like comparison at this
    # validation scale: 196 updates * 128 horizon ~= 25,088 env steps.
    total_updates: int = 196
    episode_seconds: int = 3600
    decision_interval: int = 5
    seed: int = 0

    # logging / checkpointing (in units of PPO updates, not env steps)
    log_every: int = 5
    checkpoint_every: int = 50
