"""
scripts/preflight_check.py -- Play button, no arguments. Run it BEFORE starting any 12-hour training run.
Confirms the files on disk are the final versions and that DQN and PPO are configured to be a fair, matched comparison.
Prints PASS/FAIL per check and exits non-zero if anything fails.
"""
import ast, glob, os, re, symtable, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
fails = []


def check(ok, msg, detail=""):
    print(("PASS  " if ok else "FAIL  ") + msg + (f"   [{detail}]" if detail else ""))
    if not ok: fails.append(msg)


def text(rel):
    p = os.path.join(ROOT, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


# ---- files are the final versions -------------------------------------------------
for rel, needles in {
    "dqn/train.py": ["DEFAULT_SEED", "config.json", "Episodes completed", "min elapsed"],
    "ppo/train.py": ["DEFAULT_SEED", "config.json", "ret_scale", "Episodes completed", "min elapsed"],
    "ppo/config.py": ["scale_value_loss"],
    "environment/eval_common.py": ["- 1e-4"],
    "diagnostics/diag_common.py": ["DEFAULT_CHECKPOINTS"],
    "scripts/run_eval.py": ["fixed_cycle_choice.json"],
}.items():
    t = text(rel)
    check(t is not None and all(n in t for n in needles), f"{rel} is the final version", "missing file" if t is None else "")
for rel in ("dqn/train.py", "ppo/train.py"):
    t = text(rel)
    if t:
        top = symtable.symtable(t, rel, "exec"); mod = {s.get_name() for s in top.get_symbols() if s.is_imported()}
        bad = [(f.get_name(), s.get_name()) for f in top.get_children() if f.get_type() == "function"
               for s in f.get_symbols() if s.is_local() and s.get_name() in mod]
        check(not bad, f"{rel}: no function shadows a module-level import (would crash with UnboundLocalError)", str(bad))

# ---- DQN and PPO configured identically where it matters ---------------------------
try:
    from dqn.config import DQNConfig
    from ppo.config import PPOConfig
    d, p = DQNConfig(), PPOConfig()
    check(d.total_env_steps == p.total_updates * p.horizon == 360_000, "training budget 500 episodes = 360,000 env steps for both",
          f"dqn {d.total_env_steps}, ppo {p.total_updates}x{p.horizon}")
    check(p.horizon == d.episode_seconds // d.decision_interval == 720, "PPO horizon = one 720-step episode")
    for name in ("gamma", "reward_clip", "decision_interval", "episode_seconds"):
        check(getattr(d, name) == getattr(p, name), f"same {name}", f"{getattr(d, name)} vs {getattr(p, name)}")
    check(tuple(d.hidden_sizes) == tuple(p.hidden_sizes), "same hidden sizes", f"{d.hidden_sizes} vs {p.hidden_sizes}")
    check(p.scale_value_loss is True, "PPO scale_value_loss is enabled (decided from the gradient probe)", f"{p.scale_value_loss}")
except Exception as e:
    check(False, "could not import dqn.config / ppo.config", repr(e))

# ---- environment limits --------------------------------------------------------------
try:
    from environment.grid_env import GridEnvConfig
    f = GridEnvConfig.__dataclass_fields__
    check((f["min_green"].default, f["max_green"].default, f["yellow_time"].default) == (10, 90, 4), "env min_green/max_green/yellow = 10/90/4")
except Exception as e:
    check(False, "could not read GridEnvConfig", repr(e))

# ---- 3x3 vehicles use the same vehicle type as the 4x4 ----------------------------------
for tag, folder, names in (("4x4", "sumo_4x4_network", ["routes.rou.xml"]), ("3x3", "sumo_3x3_network", ["routes3x3.rou.xml"])):
    t = next((text(f"{folder}/{n}") for n in names if text(f"{folder}/{n}")), None)
    if t is None: check(False, f"{tag} route file found"); continue
    n_veh, n_car = len(re.findall(r"<vehicle ", t)), len(re.findall(r'<vehicle [^>]*type="car"', t))
    check(n_veh > 0 and n_veh == n_car and 'maxSpeed="16.67"' in t, f"{tag} routes: all {n_veh} vehicles use vType car (maxSpeed 16.67)", f"{n_car} of {n_veh}")

print("\n" + ("ALL CHECKS PASSED -- safe to start training." if not fails else f"{len(fails)} CHECK(S) FAILED -- fix before training."))
sys.exit(1 if fails else 0)
