#!/bin/bash
# Run all four evaluations needed for training_curves.png's companion
# dqn_ppo_comparison.png. Edit the checkpoint paths if yours differ.
set -e

DQN_CKPT=runs/dqn_run_25k/qnet_final.pt
PPO_CKPT=runs/ppo_run1/actor_critic_final.pt
CYCLE=15   # keep this the SAME across all four calls -- see note below

python -m dqn.evaluate --checkpoint "$DQN_CKPT" \
    --net_file sumo_4x4_network/grid4x4.net.xml \
    --route_file sumo_4x4_network/routes.rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/dqn_run_25k/eval_4x4

python -m dqn.evaluate --checkpoint "$DQN_CKPT" \
    --net_file sumo_3x3_network/grid3x3.net.xml \
    --route_file sumo_3x3_network/routes3x3_rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/dqn_run_25k/eval_3x3

python -m ppo.evaluate --checkpoint "$PPO_CKPT" \
    --net_file sumo_4x4_network/grid4x4.net.xml \
    --route_file sumo_4x4_network/routes.rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/ppo_run1/eval_4x4

python -m ppo.evaluate --checkpoint "$PPO_CKPT" \
    --net_file sumo_3x3_network/grid3x3.net.xml \
    --route_file sumo_3x3_network/routes3x3_rou.xml \
    --fixed_cycle_seconds $CYCLE --episodes 10 --out runs/ppo_run1/eval_3x3

echo "All four evaluations complete."
