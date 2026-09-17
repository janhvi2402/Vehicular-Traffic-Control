"""
dqn/config.py
=============
All DQN hyperparameters in one place, with an explicit justification for
every value against Mnih et al., 2015 ("Human-level control through deep
reinforcement learning", Nature 518). Where a value is copied from the
paper we say so; where it is scaled or changed we explain why, so this is
directly citable in the methods section of the report.

Paper reference values (Extended Data Table 1), for context:
    minibatch size................ 32
    replay memory size............ 1,000,000 frames
    agent history length.......... 4 (stacked frames)
    target network update freq.... 10,000 parameter updates
    discount factor (gamma)....... 0.99
    action repeat.................. 4
    update frequency.............. 4
    learning rate.................. 0.00025 (RMSProp)
    gradient / squared-grad momentum 0.95 / 0.95
    min squared gradient........... 0.01
    initial / final exploration.... 1.0 / 0.1
    final exploration frame........ 1,000,000
    replay start size.............. 50,000
    no-op max....................... 30
    total training................. 50,000,000 frames (~38 days of game time)
"""

from dataclasses import dataclass


@dataclass
class DQNConfig:
    # ---------------------------------------------------------------- #
    # Network architecture
    # ---------------------------------------------------------------- #
    # The paper uses a deep CNN because its input is raw 84x84x4 pixels.
    # Our state is already a compact, hand-computed 11-dim feature vector
    # per junction (queues, waits, phase, elapsed time) -- there is no
    # spatial image to convolve over, so a CNN would add parameters
    # without adding information. We keep the paper's other architectural
    # choices that ARE input-agnostic: ReLU ("rectifier nonlinearity")
    # hidden units, and a single forward pass producing Q-values for
    # every action simultaneously (paper's stated main advantage of its
    # architecture -- "the ability to compute Q-values for all possible
    # actions in a given state with only a single forward pass").
    input_dim: int = 11          # TrafficGridEnv.NUM_FEATURES, one junction's local obs
    hidden_sizes: tuple = (128, 128)
    n_actions: int = 2           # hold / switch, per junction

    # ---------------------------------------------------------------- #
    # Replay memory
    # ---------------------------------------------------------------- #
    # Paper: 1,000,000-frame replay memory, because a single Atari run is
    # 50,000,000 frames. Our environment produces far fewer transitions:
    # 720 decision steps/episode x 16 junctions (parameter-shared,
    # independent-DQN transitions, see USAGE.md) = 11,520 transitions per
    # episode. We size the buffer to comfortably hold several dozen
    # episodes' worth of experience (~35 episodes) rather than copying the
    # paper's absolute number, which was sized for a training run
    # thousands of times longer than ours.
    replay_capacity: int = 400_000
    replay_start_size: int = 5_000   # paper: 50,000; scaled down with buffer size above

    # ---------------------------------------------------------------- #
    # Optimization
    # ---------------------------------------------------------------- #
    # Paper: minibatch size 32. We use a larger minibatch (64) because
    # each of our environment steps yields 16 transitions at once (one per
    # junction) instead of Atari's 1, so more i.i.d.-ish samples are
    # available per unit of wall-clock time without over-correlating a
    # minibatch to a single decision step.
    batch_size: int = 64
    train_frequency: int = 4      # paper: update_frequency=4 (gradient step every 4 env steps)
    gamma: float = 0.99            # paper: discount factor, unchanged -- long-horizon credit
                                    # assignment (queue build-up/dissipation) is exactly the
                                    # regime this discount was designed for.

    # Optimizer: the paper uses RMSProp(lr=0.00025, momentum=0.95,
    # squared-grad momentum=0.95, min squared grad=0.01), tuned for a deep
    # CNN trained on raw pixels across 49 different games with one shared
    # hyperparameter setting. For a 2-hidden-layer MLP on an 11-dim
    # hand-engineered feature vector, Adam is the standard modern
    # replacement for RMSProp in this regime (per-parameter adaptive
    # learning rates without needing RMSProp's separate momentum terms)
    # and converges more reliably for small MLPs; this is a common,
    # citable substitution in DQN follow-up work. We keep RMSProp
    # available for an ablation/ablation-style comparison if the report
    # wants to isolate the optimizer's effect.
    optimizer: str = "adam"        # "adam" or "rmsprop"
    learning_rate: float = 1e-3    # Adam default-scale LR appropriate for this net size
    rmsprop_lr: float = 0.00025    # used only if optimizer == "rmsprop" (paper value)
    rmsprop_momentum: float = 0.95
    rmsprop_alpha: float = 0.95    # torch's RMSprop "alpha" == squared-grad momentum
    rmsprop_eps: float = 0.01      # paper: "min squared gradient"

    # Paper: clips the TD error to [-1, 1] via an absolute-value loss
    # outside that range -- equivalent to Huber/smooth-L1 loss with
    # delta=1. We use exactly that (nn.SmoothL1Loss), unchanged.
    huber_delta: float = 1.0

    # ---------------------------------------------------------------- #
    # Target network
    # ---------------------------------------------------------------- #
    # Paper: clone Q -> target Q every C=10,000 parameter updates. Our
    # total training budget (see train.py) is far smaller than the
    # paper's 50M-frame runs, so C is scaled down proportionally to still
    # give several hundred target refreshes over a full training run.
    target_update_frequency: int = 500   # gradient steps between target syncs

    # ---------------------------------------------------------------- #
    # Exploration (epsilon-greedy)
    # ---------------------------------------------------------------- #
    # Paper: anneal epsilon linearly 1.0 -> 0.1 over the first 1,000,000
    # of 50,000,000 frames (2% of training), then hold at 0.1. We anneal
    # over the same *fraction* (roughly the first 10% of training decision
    # steps, a little more generous than the paper's 2% since our total
    # training budget is far shorter in absolute steps and under-exploring
    # is the costlier failure mode here), then hold at the paper's final
    # value of 0.1.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1              # paper: final exploration value
    epsilon_decay_fraction: float = 0.10  # fraction of total_env_steps to anneal over

    # Paper's evaluation protocol uses a fixed, lower epsilon=0.05 at test
    # time (still epsilon-greedy, not fully greedy) "to minimize the
    # possibility of overfitting during evaluation". We use exactly this
    # value in test.py / evaluate.py.
    eval_epsilon: float = 0.05

    # ---------------------------------------------------------------- #
    # Reward handling
    # ---------------------------------------------------------------- #
    # Paper clips all rewards to {-1, 0, +1} because it trains ONE
    # architecture across 49 Atari games whose raw score scales differ by
    # orders of magnitude -- clipping was a cross-task normalization
    # trick, not a claim that clipping helps any single task. We only ever
    # train on this one traffic task, so hard [-1,1] clipping would
    # destroy real magnitude information (e.g. the difference between a
    # 2-second and a 200-second waiting-time improvement). Instead we
    # apply soft clipping to a wide band, purely to guard against rare
    # outlier transitions (e.g. a teleport event) destabilizing a
    # minibatch, while preserving relative magnitude for everything else.
    reward_clip: float = 50.0   # clip agent_rewards to [-reward_clip, +reward_clip]

    # ---------------------------------------------------------------- #
    # Training length / episode
    # ---------------------------------------------------------------- #
    total_env_steps: int = 150_000   # total decision steps (across all episodes)
    episode_seconds: int = 3600      # matches the network's route file demand horizon
    decision_interval: int = 5       # seconds of sim time per decision (env default)
    seed: int = 0

    # logging / checkpointing
    log_every: int = 200             # env steps between console/CSV logging
    checkpoint_every: int = 25_000   # env steps between checkpoint saves (raised from 5,000:
                                      # at 5,000 a 1000-episode/720,000-step research run would
                                      # write 144 checkpoint files; at 25,000 it writes ~29,
                                      # keeping disk usage manageable on a student account)