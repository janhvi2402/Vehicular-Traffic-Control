"""
scripts/plot_training_curves.py  (replaces make_training_curves.py in the project root -- delete that one)
Training curves for every algorithm in ALGOS, from runs/<algo>_seed<N>/train_log.csv. Play button, no arguments.
Writes runs/aggregate/training_curves_<algo>.png (one figure per algorithm, with its own diagnostics: DQN loss/epsilon,
PPO entropy/value-loss/policy-loss) and training_curves_compare.png (all algorithms on one pair of axes).

x-axis = environment steps (identical meaning for every algorithm; 720 steps = 1 episode).
Lines = moving average over SMOOTH episodes of the mean over seeds; band = std over seeds (>= 2 seeds).
Training curves include each algorithm's exploration (DQN epsilon-greedy, PPO sampling), so they show convergence,
NOT the final comparison -- that comes from the greedy evaluation (scripts/run_eval.py).
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------ #
ALGOS = ["dqn", "ppo"]        # EDIT (add "ac", "ca" once they exist)
SEEDS = [0, 1, 2]                   # EDIT
SMOOTH = 10                   # episodes
# ------------------------------------------------------------------ #
STEPS_PER_EPISODE = 720
COLORS = {"dqn": "#0F6E56", "ppo": "#2B5FA8", "ac": "#B5651D", "ca": "#7B3FA0"}
LABELS = {"dqn": "DQN", "ppo": "PPO", "ac": "AC", "ca": "CA"}
EXTRA = {"dqn": [("loss", "DQN Huber loss", True), ("epsilon", "epsilon (exploration)", False)],
         "ppo": [("entropy", "policy entropy", False), ("value_loss", "value loss", True), ("policy_loss", "policy loss", False)]}


def load(algo, seed):
    p = os.path.join(ROOT, "runs", f"{algo}_seed{seed}", "train_log.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    def col(n):
        try: return np.array([float(r[n]) if r[n] not in ("", "nan") else np.nan for r in rows])
        except (KeyError, ValueError): return None
    d = {n: col(n) for n in rows[0].keys()}
    d = {k: v for k, v in d.items() if v is not None}
    if "env_step" in d: d["_x"] = d["env_step"]
    elif "update" in d: d["_x"] = d["update"] * STEPS_PER_EPISODE
    else: d["_x"] = (np.arange(len(rows)) + 1) * STEPS_PER_EPISODE
    return d


def smooth(y, w):
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1): i + 1]; out[i] = np.nanmean(seg) if np.isfinite(seg).any() else np.nan
    return out


def stack(runs, name):
    """-> (x, mean, std) over seeds, truncated to the shortest run; None if the column is missing."""
    if any(name not in r for r in runs): return None
    n = min(len(r[name]) for r in runs)
    Y = np.stack([r[name][:n] for r in runs]); return runs[0]["_x"][:n], np.nanmean(Y, 0), (np.nanstd(Y, 0, ddof=1) if len(runs) > 1 else np.zeros(n))


def draw(ax, algo, runs, name, title, log=False):
    s = stack(runs, name)
    if s is None: ax.set_visible(False); return False
    x, mu, sd = s; sm = smooth(mu, SMOOTH)
    ax.plot(x, mu, color=COLORS[algo], alpha=0.25, lw=0.8); ax.plot(x, sm, color=COLORS[algo], lw=2, label=LABELS[algo])
    if len(runs) > 1: ax.fill_between(x, smooth(mu - sd, SMOOTH), smooth(mu + sd, SMOOTH), color=COLORS[algo], alpha=0.15)
    if log: ax.set_yscale("log")
    ax.set_title(title, fontsize=10); ax.set_xlabel("environment steps (720 = 1 episode)"); return True


data = {}
for a in ALGOS:
    runs = [r for r in (load(a, s) for s in SEEDS) if r is not None]
    if runs: data[a] = runs
    else: print(f"SKIP {a}: no train_log.csv for seeds {SEEDS}")
if not data: sys.exit("No training logs found.")
os.makedirs(os.path.join(ROOT, "runs", "aggregate"), exist_ok=True)
wait_col = lambda runs: "mean_waiting_time_ep" if all("mean_waiting_time_ep" in r for r in runs) else "total_waiting_time"

for a, runs in data.items():
    panels = [(wait_col(runs), "Waiting time per episode" + (" (episode mean)" if wait_col(runs) != "total_waiting_time" else " (end-of-episode snapshot, noisy)"), False),
              ("episode_reward", "Episode reward (training; dominated by switch penalty)", False)] + EXTRA.get(a, [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    for ax, (name, title, log) in zip(np.atleast_1d(axes), panels): draw(ax, a, runs, name, title, log)
    fig.suptitle(f"{LABELS[a]} training curves  (seeds {SEEDS[:len(runs)] if len(runs) < len(SEEDS) else SEEDS}, moving average {SMOOTH} episodes)")
    fig.tight_layout(); out = os.path.join(ROOT, "runs", "aggregate", f"training_curves_{a}.png"); fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)

if len(data) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, runs in data.items():
        draw(axes[0], a, runs, wait_col(runs), "Waiting time per episode (training)")
        draw(axes[1], a, runs, "episode_reward", "Episode reward (training)")
    for ax in axes: ax.legend()
    fig.suptitle("Training convergence (includes each algorithm's own exploration; the final comparison is the greedy evaluation)", fontsize=10)
    fig.tight_layout(); out = os.path.join(ROOT, "runs", "aggregate", "training_curves_compare.png"); fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)
