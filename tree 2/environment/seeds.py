"""
environment/seeds.py
SUMO/env seed scheme shared by dqn/train.py and ppo/train.py (and the evaluation scripts' seed ranges).

  training  : env seed = train_seed * 10000 + episode        (train seed 0: 0..9999, seed 1: 10000..19999, ...)
  evaluation: 1000..1999   (evaluate.py base_seed 1000 + episode index)
  fixed-cycle sweep: 2000..2999 (e.g. 2000-2004)

Training seeds therefore never share an episode seed with each other, and never with the evaluation or
sweep ranges. training_env_seed() refuses to return a value inside the reserved range (this only happens
for train seed 0 beyond 1000 episodes; use seeds >= 1 or fewer episodes).
"""
TRAIN_SEED_STRIDE = 10_000
EVAL_SEED_RANGE = (1000, 2000)     # [lo, hi)
SWEEP_SEED_RANGE = (2000, 3000)    # [lo, hi)
RESERVED = (EVAL_SEED_RANGE[0], SWEEP_SEED_RANGE[1])


def training_env_seed(train_seed: int, episode: int) -> int:
    if not (0 <= episode < TRAIN_SEED_STRIDE):
        raise ValueError(f"episode {episode} outside [0, {TRAIN_SEED_STRIDE})")
    s = train_seed * TRAIN_SEED_STRIDE + episode
    if RESERVED[0] <= s < RESERVED[1]:
        raise ValueError(f"training env seed {s} (train_seed={train_seed}, episode={episode}) falls in the reserved "
                         f"evaluation/sweep range {RESERVED}; use a train seed >= 1 or fewer than {RESERVED[0]} episodes")
    return s
