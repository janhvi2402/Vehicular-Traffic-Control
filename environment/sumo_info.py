"""
environment/sumo_info.py
Records WHICH SUMO an evaluation ran under, so results from different simulator installations cannot be mixed silently.
grid_env.py launches the executable called "sumo" (traci.start([... "sumo" ...])), i.e. the first one on PATH, so that is what is recorded:
its path, its version line, the python interpreter and the traci module path.
"""
import json
import shutil
import subprocess
import sys


def sumo_info() -> dict:
    exe = shutil.which("sumo")
    try:
        out = subprocess.run(["sumo", "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()
        version = out[0].strip() if out else "unknown"
    except Exception as e:                      # never let bookkeeping break an evaluation
        version = f"unavailable: {e!r}"
    try:
        import traci
        traci_file = getattr(traci, "__file__", None)
    except Exception:
        traci_file = None
    return {"binary": exe, "version": version, "python": sys.executable, "traci": traci_file}


def stamp_summary(path: str) -> dict:
    """Adds/overwrites the "sumo" field of an existing summary.json (the evaluation that produced it must have just run)."""
    with open(path) as f:
        s = json.load(f)
    s["sumo"] = sumo_info()
    with open(path, "w") as f:
        json.dump(s, f, indent=2)
    return s["sumo"]
