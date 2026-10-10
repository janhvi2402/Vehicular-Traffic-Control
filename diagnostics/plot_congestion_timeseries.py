"""
diagnostics/plot_congestion_timeseries.py  -- PNG only.
Grid-wide total queue length and total waiting time at every decision step of one episode
(uncapped values from info["total_queue_length"] / info["total_waiting_time"]).

  python -m diagnostics.plot_congestion_timeseries --algo ppo --checkpoint runs/ppo_seed0/actor_critic_final.pt --seed 1000
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # Play button: find project root
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.diag_common import (add_common_args, default_out, footer, load_policy,
                                     make_env, run_logged_episode)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args, _ = p.parse_known_args()
    policy, cfg = load_policy(args.algo, args.checkpoint)
    env = make_env(args, cfg)
    log = run_logged_episode(env, policy, args.seed)
    dt = env.cfg.decision_interval
    env.close()

    q, w = np.asarray(log["queue"]), np.asarray(log["wait"])
    t = (np.arange(len(q)) + 1) * dt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, q, color="#0F6E56"); axes[0].set_ylabel("total queue length (vehicles)")
    axes[1].plot(t, w, color="#993556"); axes[1].set_ylabel("total waiting time (s)"); axes[1].set_xlabel("sim time (s)")
    fig.suptitle(f"Congestion over one episode: {args.algo.upper()}")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97)); footer(fig, args)
    out = default_out(args, "congestion_timeseries.png"); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"saved: {out}")
    print(f"n_teleports={log['n_teleports']}  ended_early={log['ended_early']}  (teleports hide gridlock; ended_early shortens the episode)")
    print(f"queue: mean={q.mean():.2f} max={q.max():.0f} | waiting: mean={w.mean():.2f} max={w.max():.0f} | steps={len(q)}")


if __name__ == "__main__":
    main()
