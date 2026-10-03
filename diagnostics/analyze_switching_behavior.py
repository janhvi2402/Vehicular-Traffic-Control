"""
diagnostics/analyze_switching_behavior.py  -- PNG only; correlations printed to console.
Is the learned switching congestion-driven or timer-driven? One deterministic episode;
only ELIGIBLE decision points are used (action_mask==1: not yellow and elapsed >= min_green).
Phase convention (verified against grid4x4.net.xml for all 16 junctions): phase 0 serves E+W,
phase 2 serves N+S; phases 1 and 3 are yellow and never eligible.
Note: queue/wait come from node_features, which are clipped at QUEUE_NORM=20 / WAIT_NORM=300
per approach, so very congested approaches saturate.

  python -m diagnostics.analyze_switching_behavior --algo dqn --checkpoint runs/dqn_seed0/qnet_final.pt --seed 1000
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostics.diag_common import (add_common_args, default_out, footer, load_policy,
                                     make_env, run_logged_episode)

APPROACH_ORDER = ["N", "E", "S", "W"]  # must match TrafficGridEnv.APPROACH_ORDER
# upper edge exclusive, except the last bin which is the max_green cap itself (forced switch point)
BINS = [(10, 15), (15, 20), (20, 30), (30, 50), (50, 90), (90, 91)]


def bin_label(lo, hi):
    return "90 (cap)" if lo == 90 else f"{lo}-{hi}s"


def collect(log, env):
    rows = []
    for feats, mask, act in zip(log["feats"], log["mask"], log["actions"]):
        for i in range(env.n_agents):
            if mask[i] < 0.5:
                continue                                   # eligible decision points only
            phase_idx = int(round(float(feats[i, 8]) * 3))
            served = ["E", "W"] if phase_idx == 0 else ["N", "S"] if phase_idx == 2 else None
            if served is None:
                continue
            s_idx = [APPROACH_ORDER.index(a) for a in served]
            w_idx = [j for j in range(4) if j not in s_idx]
            q, w = feats[i, 0:4] * env.QUEUE_NORM, feats[i, 4:8] * env.WAIT_NORM
            rows.append((np.rint(feats[i, 10] * env.cfg.max_green), q[s_idx].sum(), q[w_idx].sum(),
                         w[s_idx].sum(), w[w_idx].sum(), int(act[i])))
    return np.array(rows, dtype=np.float64)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args, _ = p.parse_known_args()
    policy, cfg = load_policy(args.algo, args.checkpoint)
    env = make_env(args, cfg)
    log = run_logged_episode(env, policy, args.seed)
    R = collect(log, env)
    env.close()

    elapsed, sq, wq, sw, ww, act = R.T
    imb = ww - sw
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    print(f"eligible decision points: {len(R)}   overall switch rate: {act.mean():.3f}")
    print("\nswitch rate by elapsed bin")
    rates, labels = [], []
    for lo, hi in BINS:
        m = (elapsed >= lo) & (elapsed < hi)
        labels.append(bin_label(lo, hi)); rates.append(act[m].mean() if m.any() else 0.0)
        print(f"  {bin_label(lo, hi):>9s}: n={int(m.sum()):5d}  switch_rate={rates[-1]:.3f}")
    print("\ncorrelation of the switch decision with:")
    for name, v in [("elapsed", elapsed), ("served_queue", sq), ("waiting_queue", wq),
                    ("served_wait", sw), ("waiting_wait", ww), ("imbalance(wait-side minus served-side wait)", imb)]:
        print(f"  {name:45s} {corr(act, v):+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(labels, rates, color="#0F6E56"); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("elapsed green time"); axes[0].set_ylabel("P(switch | eligible)")
    axes[0].set_title("Switch rate vs elapsed green time")
    m10 = (elapsed >= 10) & (elapsed < 15)
    if m10.sum() >= 3:
        q1, q3 = np.percentile(imb[m10], [33, 67])
        groups = [("Low\n(waiting<<served)", imb[m10] < q1), ("Mid", (imb[m10] >= q1) & (imb[m10] < q3)),
                  ("High\n(waiting>>served)", imb[m10] >= q3)]
        axes[1].bar([g[0] for g in groups], [act[m10][g[1]].mean() if g[1].any() else 0 for g in groups], color="#993556")
        print("\nswitch rate by imbalance tercile at elapsed 10-15s:",
              [round(float(act[m10][g[1]].mean()), 3) if g[1].any() else None for g in groups])
    axes[1].set_ylim(0, 1); axes[1].set_xlabel("waiting-side minus served-side wait")
    axes[1].set_ylabel("P(switch | eligible)"); axes[1].set_title("Switch rate vs congestion imbalance\n(fixed elapsed = 10-15 s)")
    fig.suptitle(f"Congestion-driven or timer-driven? {args.algo.upper()}")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); footer(fig, args)
    out = default_out(args, "switching_behavior.png"); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
