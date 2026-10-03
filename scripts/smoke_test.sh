#!/bin/bash
# 2-episode smoke run of both trainers + wasted-vote check. Run from the project root.
set -e
python tests/test_masking.py
python -m dqn.train --episodes 2 --seed 0 --out runs/smoke_dqn
python -m ppo.train --updates 2 --seed 0 --out runs/smoke_ppo
for f in runs/smoke_dqn/train_log.csv runs/smoke_ppo/train_log.csv; do
  echo "== $f"; cat "$f"; test "$(wc -l < "$f")" -ge 2
done
python scripts/count_wasted_votes.py --algo dqn --checkpoint runs/smoke_dqn/qnet_final.pt
python scripts/count_wasted_votes.py --algo ppo --checkpoint runs/smoke_ppo/actor_critic_final.pt
