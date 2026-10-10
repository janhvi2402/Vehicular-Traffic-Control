"""
scripts/run_eval.py  -- replaces run_matched_eval.sh so it runs from the VS Code Play button.
Evaluates every (algo, train seed) in RUNS on the trained 4x4 grid and the unseen 3x3 grid. EDIT the block below.
Output: runs/<algo>_seed<N>/eval_4x4/ and eval_3x3/  (read by diagnostics/compare_algorithms.py).
"""
import glob, hashlib, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------- EDIT ------------------------------- #
ALGOS = ["dqn", "ppo"]
SEEDS = [0, 1, 2]                      # training seeds you have trained, e.g. [0, 1, 2]
EPISODES = 10                    # evaluation episodes (seeds 1000..)
CYCLE_4X4 = "auto"               # "auto" = read the tuned value from runs/fixed_cycle_choice.json (run scripts/sweep_fixed_cycle.py first)
CYCLE_3X3 = "auto"               # or put a number of seconds here to force it
# --------------------------------------------------------------------- #
CKPT = {"dqn": "qnet_final.pt", "ppo": "actor_critic_final.pt"}

def first_existing(folder, patterns):
    for pat in patterns:
        hits = glob.glob(os.path.join(ROOT, folder, pat))
        if hits: return hits[0]
    sys.exit(f"ERROR: none of {patterns} found in {folder}/ -- check the real file names on disk.")

def _files_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def _resolve_cycle(tag, setting, net, route):
    if setting != "auto":
        return float(setting)
    choice_file = os.path.join(ROOT, "runs", "fixed_cycle_choice.json")
    if not os.path.exists(choice_file):
        sys.exit("ERROR: run scripts/sweep_fixed_cycle.py first (it tunes the fixed-cycle baseline once for all algorithms).")
    c = json.load(open(choice_file))[tag]
    if c["files_hash"] != _files_hash(net, route):
        sys.exit(f"ERROR: the {tag} network/route files changed since the sweep -- re-run scripts/sweep_fixed_cycle.py.")
    return float(c["chosen"])


NETS = {
    "4x4": (first_existing("sumo_4x4_network", ["grid4x4.net.xml", "grid4x4_net.xml"]),
            first_existing("sumo_4x4_network", ["routes.rou.xml", "routes_rou.xml"]), CYCLE_4X4),
    "3x3": (first_existing("sumo_3x3_network", ["grid3x3.net.xml", "grid3x3_net.xml"]),
            first_existing("sumo_3x3_network", ["routes3x3.rou.xml", "routes3x3_rou.xml"]), CYCLE_3X3),
}

NETS = {tag: (net, route, _resolve_cycle(tag, cyc, net, route)) for tag, (net, route, cyc) in NETS.items()}
print("fixed-cycle baseline used:", {t: v[2] for t, v in NETS.items()})

for algo in ALGOS:
    for seed in SEEDS:
        run_dir = os.path.join("runs", f"{algo}_seed{seed}")
        ckpt = os.path.join(ROOT, run_dir, CKPT[algo])
        if not os.path.exists(ckpt):
            print(f"SKIP {algo} seed {seed}: no checkpoint at {ckpt}"); continue
        for name, (net, route, cycle) in NETS.items():
            cmd = [sys.executable, "-m", f"{algo}.evaluate", "--checkpoint", ckpt, "--net_file", net, "--route_file", route,
                   "--episodes", str(EPISODES), "--out", os.path.join(ROOT, run_dir, f"eval_{name}")]
            if cycle is not None: cmd += ["--fixed_cycle_seconds", str(cycle)]
            print("\n$", " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
print("\nAll evaluations complete.")
