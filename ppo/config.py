"""
ppo/config.py
=============
Central hyperparameter config for the PPO traffic-signal-control agent.
Every field here is read somewhere in train.py / evaluate.py / test.py /
actor_critic.py -- see the inline comments for exactly where and why.

This is a plain dataclass, not an argparse target, so importing it never
requires any command-line input. That means you can just hit "Run" /
F5 in VS Code on train.py (or test.py / evaluate.py) directly: every
argparse flag in those scripts already has a default, so the whole chain
runs with zero required setup once this file exists.

If you want to change a hyperparameter, edit the value below -- there is
nothing else to wire up.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    # ---------------------------------------------------------------- #
    # Network architecture (actor_critic.py: ActorCritic(input_dim,
    # hidden_sizes, n_actions))
    # ---------------------------------------------------------------- #
    # 11-dim hand-engineered per-junction feature vector (see
    # actor_critic.py's module docstring).
    input_dim: int = 11
    # FAIRNESS FIX: was (64, 64), matching the paper's continuous-control
    # MLP convention (Schulman et al., 2017, App. C). Changed to match
    # DQN's hidden_sizes=(128, 128) instead -- with the paper's smaller
    # network, DQN's net had ~4x the hidden-layer parameters on the exact
    # same 11-dim input / 2-action output, so any performance gap could
    # have just been "bigger network", not "better algorithm". This
    # deviates from the paper's own convention specifically so this
    # project's DQN-vs-PPO comparison isolates the algorithm, not the
    # architecture.
    hidden_sizes: tuple = (128, 128)
    # Binary per-junction decision: "switch" or "hold" (see actor_critic.py:
    # ActorCritic docstring, "policy logits (2 actions)").
    n_actions: int = 2

    # ---------------------------------------------------------------- #
    # Environment (train.py / evaluate.py / test.py: GridEnvConfig(...))
    # ---------------------------------------------------------------- #
    # Each episode runs for a fixed 3600 simulated seconds (see train.py's
    # comment: "This env only ever ends via the 3600s time limit").
    episode_seconds: int = 3600
    # Seconds of simulated time between successive agent decisions.
    #
    # FAIRNESS FIX: was 10s. Matched to DQN's decision_interval=5s here.
    # A different decision_interval isn't just a step-count difference --
    # it means one algorithm literally gets to observe and act on traffic
    # twice as often as the other. Finer-grained control is a real
    # advantage independent of which learning algorithm is used: an agent
    # that reacts every 5s can catch a building queue faster than one that
    # only gets to look every 10s. Left mismatched, a DQN-vs-PPO
    # performance gap could just mean "checking twice as often helps",
    # not "DQN is the better algorithm" -- so both must observe/act at the
    # same temporal granularity for the comparison to isolate the
    # algorithm itself.
    decision_interval: int = 5

    # ---------------------------------------------------------------- #
    # Rollout / PPO update sizing (train.py: RolloutBuffer(horizon, ...),
    # end_update loop, train_log.csv header)
    # ---------------------------------------------------------------- #
    # Env steps collected per PPO update, per agent (Algorithm 1's
    # "run policy for T timesteps"). Set to episode_seconds /
    # decision_interval (3600 / 5 = 720), so each update's rollout lines
    # up EXACTLY with one episode boundary (unlike the original
    # horizon=128, which straddled episodes and relied on the
    # truncation-bootstrap fix in train.py). This makes train_log.csv's
    # per-update "episode_reward" and "total_waiting_time" columns
    # directly one-row-per-episode, which is what you want for clean
    # training-curve plots in a report.
    #
    # NOTE: this was 360 when decision_interval was 10s. Now that
    # decision_interval matches DQN's 5s (see above), each episode takes
    # twice as many decision-steps, so horizon doubles to keep the
    # one-horizon-equals-one-episode property intact. total_updates below
    # is halved in turn, to keep the total env-step BUDGET the same.
    horizon: int = 720
    # Grand total number of PPO updates for a full run (train.py:
    # cfg.total_updates, overridable with --updates).
    #
    # 500 updates x horizon=720 = 360,000 env steps = 500 episodes --
    # 500 is the floor for a defensible research-grade run (300-600
    # episodes is the typical range needed to show clear convergence on a
    # 16-junction grid), and this MUST equal DQN's own matched-comparison
    # training budget episode-for-episode (see dqn/config.py and the
    # matched-run instructions in run_matched_eval.sh) -- mismatched
    # training budgets are a much bigger fairness problem than any single
    # hyperparameter, since an undertrained checkpoint will lose to an
    # overtrained one regardless of which algorithm is "better". This is
    # a budget, not a guarantee -- SUMO step cost is the real bottleneck
    # and depends on your server, not on this number. See the timing note
    # at the bottom of this file.
    total_updates: int = 500

    # ---------------------------------------------------------------- #
    # PPO objective (Eq. 7 / Eq. 9, Schulman et al., 2017)
    # ---------------------------------------------------------------- #
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # GAE lambda (Eq. 11-12)
    clip_epsilon: float = 0.2    # L^CLIP clipping range (Eq. 7)
    value_coeff: float = 0.5     # c1: value-loss weight in Eq. 9
    entropy_coeff: float = 0.01  # c2: entropy-bonus weight in Eq. 9
    max_grad_norm: float = 0.5   # gradient-clipping norm

    # ---------------------------------------------------------------- #
    # Optimization (train.py: Adam optimizer, epoch/minibatch loop)
    # ---------------------------------------------------------------- #
    learning_rate: float = 3e-4
    num_epochs: int = 4          # PPO epochs per update over the rollout
    minibatch_size: int = 64

    # ---------------------------------------------------------------- #
    # Reward shaping (train.py: np.clip(info["agent_rewards"], ...))
    # ---------------------------------------------------------------- #
    # FAIRNESS FIX: was 10.0. Matched to DQN's reward_clip=50.0 -- the two
    # networks must learn from identically-SCALED reward signals, or this
    # becomes a second confound alongside network size: a tighter clip
    # changes gradient magnitudes and silently redefines what counts as
    # an "outlier" transition getting clipped away, independent of which
    # algorithm is doing the learning.
    reward_clip: float = 50.0

    # ---------------------------------------------------------------- #
    # Misc / bookkeeping (train.py: torch.manual_seed, env reset seed,
    # log/checkpoint cadence)
    # ---------------------------------------------------------------- #
    # FAIRNESS FIX: was 42. Matched to DQNConfig.seed=0 -- with different
    # defaults and no CLI override on either side, the two "matched"
    # 500-episode runs would have trained under different SUMO/torch
    # randomness by default, the same class of confound the rest of this
    # file's fixes were written to remove. A single matched seed doesn't
    # solve everything (see the multi-seed disclaimer already in both
    # config files -- one seed can't distinguish a real algorithmic
    # difference from seed variance), but it removes this as an
    # UNCONTROLLED variable for whatever single-seed run you do make.
    seed: int = 0
    log_every: int = 1           # log a train_log.csv row every N updates
    # Checkpoint fairly often (every 25 updates = every 25 episodes here,
    # ~45 min-ish depending on your server). train.py's final save only
    # happens AFTER the whole loop finishes -- if your 15-hour job gets
    # killed or disconnected partway through (common on shared college
    # servers), only these periodic actor_critic_updateN.pt files survive,
    # and you resume with:
    #   --resume runs/ppo_run_matched500/actor_critic_update<N>.pt --start_update <N>
    checkpoint_every: int = 25


# --------------------------------------------------------------------- #
# TIMING NOTE -- read this before trusting the 500-update budget above
# --------------------------------------------------------------------- #
# 500 updates is a *target*, sized purely from episode counts. Whether it
# actually fits your available compute depends on your server's SUMO
# throughput, which this file can't know. Do a short timing probe first:
#
#   1. Temporarily run with --updates 10 (or edit total_updates to 10)
#      and note the wall-clock time for those 10 updates to finish.
#   2. seconds_per_update = elapsed_seconds / 10
#      updates_that_fit_in_your_window = window_seconds / seconds_per_update
#   3. If that comes out under 500, do NOT quietly ship a smaller PPO run
#      against a full 500-episode DQN run (or vice versa) -- retrain
#      WHICHEVER side is smaller so both land on the same episode count.
#      A mismatched budget silently reintroduces the exact fairness
#      problem this file's other fixes were meant to close.
#
# Don't shrink horizon back down to squeeze in more nominal "updates" --
# keep horizon=720 (clean episode alignment, matched to DQN's decision
# granularity) and adjust total_updates on BOTH algorithms together if
# you must reduce the budget.

# --------------------------------------------------------------------- #
# KNOWN, NOT FIXED -- gradient-update budget per episode still differs
# --------------------------------------------------------------------- #
# Even with decision_interval, hidden_sizes, and reward_clip now matched
# to DQN above, structural asymmetries remain and are being disclosed
# here rather than patched:
#
# Gradient steps per episode. PPO takes num_epochs=4 full passes each
# update over a rollout of horizon x n_agents = 720 x 16 = 11,520
# transitions (one horizon-length rollout across all 16 junctions, since
# they share one policy but each contributes its own row of data every
# step). At minibatch_size=64, that's 11,520 / 64 = 180 minibatches x 4
# epochs = 720 gradient steps per UPDATE, and one update is exactly one
# 720-step episode here -- so PPO takes ~720 gradient steps per episode.
# DQN takes one gradient step every train_frequency=4 ENV steps, and one
# episode is 720 env steps, so DQN takes 720 / 4 = 180 gradient steps per
# episode. The real ratio is 4:1 (NOT the ~2x a step-count-only glance
# might suggest, and NOT the ~45-per-episode this comment previously and
# incorrectly stated by forgetting to multiply by n_agents). In terms of
# raw sample-presentations per episode: PPO re-uses its 11,520-transition
# on-policy rollout across 4 epochs, presenting ~46,080 sample-instances
# per episode; DQN draws 180 gradient steps x batch_size=64 = 11,520
# samples per episode WITH REPLACEMENT from a much larger, older,
# off-policy replay buffer -- the same raw count as one PPO epoch's
# dataset, but a different statistical character (i.i.d.-ish draws from
# history vs. 4 correlated passes over one fresh rollout).
#
# Activation function. DQN uses ReLU, PPO uses Tanh -- each is that
# algorithm's own paper-standard choice (Mnih et al. for DQN; Schulman et
# al.'s continuous-control MLP convention for PPO), but it is still
# another architectural difference beyond "the algorithm" itself.
#
# Learning rate. DQN uses 1e-3 (Adam), PPO uses 3e-4 -- neither has been
# tuned for this specific task; both are the common default scale for
# their respective optimizer/architecture combination in the literature.
#
# None of these three are forced to match here: doing so would mean
# overriding num_epochs/train_frequency away from each algorithm's own
# standard on-policy vs. off-policy design, or replacing each paper's own
# standard activation/LR choice with an arbitrary shared one -- a
# stranger and less defensible change than just naming the asymmetries.
# State these plainly as limitations in the report's methodology /
# limitations section rather than silently treating "same episode
# budget, same hidden size, same reward clip" as "identical training
# regime".