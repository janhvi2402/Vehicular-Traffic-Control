"""
diagnostics/plot_green_durations.py  -- PNG only; numbers printed to the console.
Green-duration histograms of the trained DQN and PPO side by side (same bins, same seed, same env files),
built from the env's completed-green events. Unlike the other diagnostics this takes BOTH checkpoints.
Footer prints both checkpoint paths, the seed and the env-file hash.

  python -m diagnostics.plot_green_durations --checkpoint_dqn runs/dqn_seed0/qnet_final.pt \
         --checkpoint_ppo runs/ppo_seed0/actor_critic_final.pt --seed 1000
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.diag_common import (DEFAULT_NET, DEFAULT_ROUTE, file_hash, load_policy, make_env,
                                     run_logged_episode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dqn", required=True); p.add_argument("--checkpoint_ppo", required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--net_file", default=DEFAULT_NET); p.add_argument("--route_file", default=DEFAULT_ROUTE)
    p.add_argument("--out", default=None)
    a, _ = p.parse_known_args()

    results = {}
    for algo, ckpt in (("dqn", a.checkpoint_dqn), ("ppo", a.checkpoint_ppo)):
        policy, cfg = load_policy(algo, ckpt)
        env = make_env(a, cfg)
        log = run_logged_episode(env, policy, a.seed)
        results[algo] = (log, env.cfg, list(env.tl_ids)); env.close()

    ecfg = results["dqn"][1]
    bins = np.arange(ecfg.min_green, ecfg.max_green + 5.1, 5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    print(f"{'algo':5s}{'greens':>8s}{'voted':>8s}{'forced':>8s}{'at_max':>8s}{'share@max':>11s}{'mean_s':>8s}{'p10':>6s}{'p50':>6s}{'p90':>6s}"
          f"{'votes/junction':>16s}")
    for ax, (algo, color) in zip(axes, (("dqn", "#0F6E56"), ("ppo", "#2B5FA8"))):
        log, ecfg, tl_ids = results[algo]
        g = log["greens"]
        dur = np.array([e["end"] - e["start"] for e in g]); forced = np.array([e["ended_by_forced"] for e in g], dtype=bool)
        ax.hist(dur, bins=bins, color=color)
        ax.set_title(f"{algo.upper()}: {len(g)} greens, {int((dur >= ecfg.max_green).sum())} at max_green, {int(forced.sum())} forced")
        ax.set_xlabel("green duration (s)"); ax.set_xlim(ecfg.min_green - 1, ecfg.max_green + 6)
        n_at_max = int((dur >= ecfg.max_green).sum())
        per_j = [sum(1 for e in g if e["junction"] == t and not e["ended_by_forced"]) for t in tl_ids]
        print(f"{algo:5s}{len(g):8d}{len(g) - int(forced.sum()):8d}{int(forced.sum()):8d}{n_at_max:8d}{n_at_max / max(len(g), 1):11.3f}"
              f"{dur.mean():8.1f}{np.percentile(dur, 10):6.0f}{np.percentile(dur, 50):6.0f}{np.percentile(dur, 90):6.0f}"
              f"{np.mean(per_j):10.1f} (min {min(per_j)}, max {max(per_j)})")
    axes[0].set_ylabel("count")
    fig.suptitle(f"Green-duration distribution, seed {a.seed} (allowed {ecfg.min_green}-{ecfg.max_green} s)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(0.005, 0.004, f"dqn={os.path.abspath(a.checkpoint_dqn)}  ppo={os.path.abspath(a.checkpoint_ppo)}  seed={a.seed}  "
             f"env_hash(net+routes)={file_hash(a.net_file, a.route_file)}", fontsize=6, color="0.35")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint_dqn)), "figures", "green_durations_dqn_vs_ppo.png")
    os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
