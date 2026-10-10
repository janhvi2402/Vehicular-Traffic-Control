"""
scripts/check_matched_budget.py  -- Play button, no arguments. Run it after training and BEFORE evaluating.
Confirms every algorithm / training seed saw the same training budget (default: 500 episodes = 360,000 env steps),
so the comparison is matched. Reads runs/<algo>_seed<N>/train_log.csv. Exits with an error if anything differs.
"""
import csv, glob, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALGOS = ["dqn", "ppo"]          # add "ac", "ca" once they are trained
EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_EPISODES = 500
STEPS_PER_EPISODE = 720          # 3600 s / 5 s decision interval
FINAL_CKPT = {"dqn": "qnet_final.pt", "ppo": "actor_critic_final.pt"}   # ac / ca: any *final*.pt

problems, seeds_by_algo = [], {}
print(f"{'run':14s}{'log rows':>10s}{'last step':>12s}{'episodes':>10s}  final checkpoint")
for algo in ALGOS:
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", f"{algo}_seed*"))):
        m = re.search(r"_seed(\d+)$", d)
        if not m or not os.path.isdir(d): continue
        seed = int(m.group(1)); seeds_by_algo.setdefault(algo, []).append(seed)
        name = os.path.basename(d); log = os.path.join(d, "train_log.csv")
        if not os.path.exists(log):
            problems.append(f"{name}: no train_log.csv"); print(f"{name:14s}  (no train_log.csv)"); continue
        rows = list(csv.DictReader(open(log)))
        cols = rows[0].keys() if rows else []
        if "env_step" in cols:   last_step = int(float(rows[-1]["env_step"]))
        elif "update" in cols:   last_step = int(float(rows[-1]["update"])) * STEPS_PER_EPISODE   # PPO: one update = one episode
        else:                    last_step = -1
        episodes = last_step // STEPS_PER_EPISODE if last_step >= 0 else -1
        ck = glob.glob(os.path.join(d, FINAL_CKPT.get(algo, "*final*.pt")))
        print(f"{name:14s}{len(rows):>10d}{last_step:>12d}{episodes:>10d}  {'yes' if ck else 'MISSING'}")
        if episodes != EXPECTED_EPISODES:
            problems.append(f"{name}: trained {episodes} episodes (expected {EXPECTED_EPISODES}); check its config/--episodes")
        if not ck:
            problems.append(f"{name}: final checkpoint missing (training did not finish?)")
ref = None
for algo, seeds in seeds_by_algo.items():
    if ref is None: ref = (algo, sorted(seeds))
    elif sorted(seeds) != ref[1]:
        problems.append(f"{algo} was trained with seeds {sorted(seeds)} but {ref[0]} with {ref[1]} -- use the same seeds for every algorithm")
for algo, seeds in seeds_by_algo.items():
    if sorted(seeds) != EXPECTED_SEEDS:
        problems.append(f"{algo}: seeds found {sorted(seeds)}, expected {EXPECTED_SEEDS}")
    cfgs = {}
    for sd in seeds:
        cf = os.path.join(ROOT, "runs", f"{algo}_seed{sd}", "config.json")
        if os.path.exists(cf):
            c = json.load(open(cf)); c.pop("seed", None); cfgs[sd] = c
        else:
            problems.append(f"{algo}_seed{sd}: no config.json (trained with an older train.py?)")
    if cfgs and any(c != next(iter(cfgs.values())) for c in cfgs.values()):
        problems.append(f"{algo}: hyperparameters differ between seeds {sorted(cfgs)} -- every seed must use identical code/config")
missing = [a for a in ALGOS if a not in seeds_by_algo]
if missing: problems.append(f"no runs found yet for: {missing}")
print()
if problems:
    print("NOT MATCHED:"); [print("  -", p) for p in problems]; sys.exit(1)
print(f"OK: all algorithms/seeds trained for {EXPECTED_EPISODES} episodes ({EXPECTED_EPISODES * STEPS_PER_EPISODE} env steps), same seeds.")
