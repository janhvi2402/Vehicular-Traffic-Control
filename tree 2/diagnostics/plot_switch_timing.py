"""
diagnostics/plot_switch_timing.py  -- PNG only.
Green-duration histogram, phase timeline (green + yellow), forced switches per junction.
Built from the env's completed-green events (info["completed_greens"]), NOT from phase
sampled at 5 s decision boundaries. Yellow for each green is [end, end + yellow_time].

  python -m diagnostics.plot_switch_timing --algo dqn --checkpoint runs/dqn_seed0/qnet_final.pt --seed 1000
"""
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
    p.add_argument("--t0", type=float, default=0.0, help="timeline window start (s)")
    p.add_argument("--t1", type=float, default=600.0, help="timeline window end (s)")
    args, _ = p.parse_known_args()

    policy, cfg = load_policy(args.algo, args.checkpoint)
    env = make_env(args, cfg)
    log = run_logged_episode(env, policy, args.seed)
    ycfg, tl_ids = env.cfg, list(env.tl_ids)
    env.close()

    g = log["greens"]
    dur = np.array([e["end"] - e["start"] for e in g])
    forced = np.array([e["ended_by_forced"] for e in g], dtype=bool)
    n_greens, n_forced_ev = len(g), int(forced.sum())
    n_honored_ev = n_greens - n_forced_ev

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax_h, ax_f, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    bins = np.arange(ycfg.min_green, ycfg.max_green + 5.1, 5)
    ax_h.hist(dur[~forced], bins=bins, color="#0F6E56", label="ended by vote")
    ax_h.hist(dur[forced], bins=bins, color="#993556", label="ended by forced (max_green)", bottom=np.histogram(dur[~forced], bins=bins)[0])
    ax_h.set_xlabel("green duration (s)"); ax_h.set_ylabel("count"); ax_h.set_title(f"Green durations (n={n_greens})"); ax_h.legend()

    per_forced = np.array([sum(1 for e in g if e["junction"] == t and e["ended_by_forced"]) for t in tl_ids])
    per_voted = np.array([sum(1 for e in g if e["junction"] == t and not e["ended_by_forced"]) for t in tl_ids])
    x = np.arange(len(tl_ids))
    ax_f.bar(x, per_voted, color="#0F6E56", label="executed by vote")
    ax_f.bar(x, per_forced, bottom=per_voted, color="#993556", label="forced (max_green)")
    ax_f.set_xticks(x); ax_f.set_xticklabels(tl_ids, rotation=90, fontsize=7); ax_f.legend(fontsize=8)
    ax_f.set_ylabel("phase changes per episode")
    ax_f.set_title(f"Phase changes per junction (voted {n_honored_ev}, forced {n_forced_ev}; "
                   f"forced share {n_forced_ev / max(n_greens, 1):.1%})")

    for row, t in enumerate(tl_ids):
        ev = sorted((e for e in g if e["junction"] == t), key=lambda e: e["start"])
        greens = [(e["start"], e["end"] - e["start"]) for e in ev]
        yellows = [(e["end"], ycfg.yellow_time) for e in ev]
        ax_t.broken_barh(greens, (row - 0.4, 0.8), facecolors="#2E9E5B")
        ax_t.broken_barh(yellows, (row - 0.4, 0.8), facecolors="#F2C200")
    ax_t.set_xlim(args.t0, args.t1); ax_t.set_ylim(-0.6, len(tl_ids) - 0.4)
    ax_t.set_yticks(range(len(tl_ids))); ax_t.set_yticklabels(tl_ids, fontsize=7)
    ax_t.set_xlabel("sim time (s)"); ax_t.set_title("Phase timeline (green = green phase, yellow = yellow clearance)")

    fig.suptitle(f"Switch timing: {args.algo.upper()}"); fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    footer(fig, args)
    out = default_out(args, "switch_timing.png"); fig.savefig(out, dpi=150); plt.close(fig)

    print(f"saved: {out}")
    print(f"greens={n_greens}  honored(vote)={n_honored_ev}  forced={n_forced_ev}")
    print(f"info totals: honored={log['n_honored_info']}  forced={log['n_forced_info']}")
    print(f"forced share of all phase changes: {n_forced_ev / max(n_greens, 1):.1%}   "
          f"(near 100% with greens piled at {ycfg.max_green}s = 'never vote, let max_green switch' shortcut)")
    print(f"voted switches per junction per episode: min={per_voted.min()} mean={per_voted.mean():.1f} max={per_voted.max()}")
    print(f"greens ended at max_green: {int((dur >= ycfg.max_green).sum())} of {n_greens}")
    print(f"green duration  min={dur.min():.0f}  mean={dur.mean():.1f}  max={dur.max():.0f}  "
          f"(allowed {ycfg.min_green}-{ycfg.max_green})")
    print("CHECK greens == honored+forced :", n_greens == log["n_honored_info"] + log["n_forced_info"])
    print("CHECK durations in range       :", bool(dur.min() >= ycfg.min_green and dur.max() <= ycfg.max_green))


if __name__ == "__main__":
    main()
