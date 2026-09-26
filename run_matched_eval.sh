#!/bin/bash
# Run all four evaluations needed for training_curves.png's companion
# dqn_ppo_comparison.png. Edit the checkpoint paths if yours differ.
#
# IMPORTANT: DQN_CKPT below must come from a run trained for the SAME
# episode count as PPO_CKPT (both default to 500 episodes / 360,000 env
# steps as of dqn/config.py and ppo/config.py's current defaults -- see
# the fairness notes in those files). The previous DQN checkpoint this
# script pointed at (runs/dqn_run_25k/qnet_final.pt) strongly suggests a
# ~25,000-step run (~35 episodes) -- roughly 14x LESS trained than a
# 500-episode PPO run. If that's what it actually was, every comparison
# made from it is meaningless regardless of any config fix, since an
# undertrained checkpoint will lose to an overtrained one no matter which
# algorithm is "better". Retrain DQN with the matched budget before using
# this script:
#   python -m dqn.train --episodes 500 --out runs/dqn_run_matched500
set -e

DQN_CKPT=runs/dqn_run_matched500/qnet_final.pt
PPO_CKPT=runs/ppo_run_matched500/actor_critic_final.pt
CYCLE=15   # keep this the SAME across all four calls -- see note below

python -m dqn.evaluate --checkpoint "$DQN_CKPT" \
    --net_file sumo_4x4_network/grid4x4.net.xml \
    --route_file sumo_4x4_network/routes.rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/dqn_run_matched500/eval_4x4

python -m dqn.evaluate --checkpoint "$DQN_CKPT" \
    --net_file sumo_3x3_network/grid3x3_net.xml \
    --route_file sumo_3x3_network/routes3x3_rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/dqn_run_matched500/eval_3x3

python -m ppo.evaluate --checkpoint "$PPO_CKPT" \
    --net_file sumo_4x4_network/grid4x4.net.xml \
    --route_file sumo_4x4_network/routes.rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/ppo_run_matched500/eval_4x4

python -m ppo.evaluate --checkpoint "$PPO_CKPT" \
    --net_file sumo_3x3_network/grid3x3_net.xml \
    --route_file sumo_3x3_network/routes3x3_rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/ppo_run_matched500/eval_3x3

echo "All four evaluations complete."