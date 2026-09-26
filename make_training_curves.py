"""
make_training_curves.py
========================
Regenerates the 4-panel PPO training-curves figure (entropy, value loss,
waiting time, episode reward) from ppo/train.py's train_log.csv.

No command-line args -- edit the two paths below, then run:
    python make_training_curves.py
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# EDIT THESE
# ------------------------------------------------------------------ #
TRAIN_LOG_CSV = "runs/ppo_run_matched500/train_log.csv"
OUT_PATH = "runs/ppo_run_matched500/training_curves.png"
TOTAL_UPDATES_LABEL = "500-update (~360k env step)"  # just for the title text
# ------------------------------------------------------------------ #

updates, entropy, value_loss, wait, reward = [], [], [], [], []
with open(TRAIN_LOG_CSV) as f:
    for row in csv.DictReader(f):
        updates.append(int(row["update"]))
        entropy.append(float(row["entropy"]))
        value_loss.append(float(row["value_loss"]))
        wait.append(float(row["total_waiting_time"]))
        reward.append(float(row["episode_reward"]))

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

axes[0].plot(updates, entropy, color="#6A5ACD")
axes[0].set_title("Policy entropy")
axes[0].set_xlabel("PPO update")

axes[1].plot(updates, value_loss, color="#993556")
axes[1].set_yscale("log")
axes[1].set_title("Value loss")
axes[1].set_xlabel("PPO update")

axes[2].plot(updates, wait, color="#0F6E56")
axes[2].set_title("Waiting time (end of episode)")
axes[2].set_xlabel("PPO update")

axes[3].plot(updates, reward, color="#854F0B")
# was "Episode reward (so far)" -- stale wording from before the
# episode_reward-logged-as-zero fix in ppo/train.py (see
# last_completed_episode_reward). Each point is now the COMPLETED
# episode's total reward, not a live running sum, so "so far" no
# longer describes what's plotted.
axes[3].set_title("Episode reward (completed episode)")
axes[3].set_xlabel("PPO update")

fig.suptitle(f"PPO training curves \u2014 {TOTAL_UPDATES_LABEL} validation run")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=150)
plt.close(fig)
print(f"Saved: {OUT_PATH}")