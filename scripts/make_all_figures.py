"""
scripts/make_all_figures.py -- Play button, no arguments. Builds every report figure from FINISHED training + evaluation:
  training curves (per algorithm + combined), behaviour diagnostics per algorithm (switching, timing, congestion),
  green-duration histograms. (Baseline-dependent figures come from scripts/make_fixed_time_figures.py.)
Each step runs in its own process; a failing step is reported at the end and does not stop the others.
Run scripts/run_eval.py first. Diagnostics support "dqn" and "ppo" only (they load those checkpoints).
"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------ #
ALGOS = ["dqn", "ppo"]        # EDIT
SEEDS = [0, 1, 2]                   # EDIT: seeds used by the aggregate figures
BEHAVIOUR_SEED = 0            # which trained seed the behaviour diagnostics use
EVAL_SEED = "1000"            # SUMO seed of the single diagnostic episode (an evaluation seed)
# ------------------------------------------------------------------ #
CKPT = {"dqn": "qnet_final.pt", "ppo": "actor_critic_final.pt"}
results = []


def run(label, *args):
    print(f"\n=== {label}", flush=True)
    rc = subprocess.run([sys.executable, *args], cwd=ROOT).returncode
    results.append((label, rc))


def ckpt(algo): return os.path.join("runs", f"{algo}_seed{BEHAVIOUR_SEED}", CKPT[algo])


run("training curves", "scripts/plot_training_curves.py")
for a in [a for a in ALGOS if a in CKPT]:
    if not os.path.exists(os.path.join(ROOT, ckpt(a))):
        print(f"SKIP behaviour diagnostics for {a}: {ckpt(a)} missing"); continue
    for d in ("analyze_switching_behavior", "plot_switch_timing", "plot_congestion_timeseries"):
        run(f"{a}: {d}", f"diagnostics/{d}.py", "--algo", a, "--checkpoint", ckpt(a), "--seed", EVAL_SEED)
if all(a in ALGOS and os.path.exists(os.path.join(ROOT, ckpt(a))) for a in ("dqn", "ppo")):
    run("green durations dqn vs ppo", "diagnostics/plot_green_durations.py", "--checkpoint_dqn", ckpt("dqn"), "--checkpoint_ppo", ckpt("ppo"), "--seed", EVAL_SEED)
print("baseline-dependent figures: run scripts/make_fixed_time_figures.py")

print("\n" + "=" * 60)
for label, rc in results: print(f"{'ok  ' if rc == 0 else 'FAIL'} {label}")
print("Figures are in runs/aggregate/ and runs/<algo>_seed<N>/figures/")
