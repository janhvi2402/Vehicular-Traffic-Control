"""
scripts/smoke_test.py  -- replaces smoke_test.sh so it runs from the VS Code Play button (no bash needed).
Masking tests, then a 2-episode run of both trainers, then the wasted-vote check. Takes a few minutes.
"""
import os, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
for _d in ("runs/smoke_dqn", "runs/smoke_ppo"):   # dqn.train refuses to overwrite a finished run
    shutil.rmtree(os.path.join(ROOT, _d), ignore_errors=True)

def run(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([PY, *cmd], cwd=ROOT, check=True)

run("tests/test_masking.py")
run("-m", "dqn.train", "--episodes", "2", "--seed", "0", "--out", "runs/smoke_dqn")
run("-m", "ppo.train", "--updates", "2", "--seed", "0", "--out", "runs/smoke_ppo")
for f in ("runs/smoke_dqn/train_log.csv", "runs/smoke_ppo/train_log.csv"):
    lines = open(os.path.join(ROOT, f)).read().strip().splitlines()
    print(f"== {f}: {len(lines)} lines"); print("\n".join(lines))
    assert len(lines) >= 2, f"{f} has no data rows"
run("scripts/count_wasted_votes.py", "--algo", "dqn", "--checkpoint", "runs/smoke_dqn/qnet_final.pt")
run("scripts/count_wasted_votes.py", "--algo", "ppo", "--checkpoint", "runs/smoke_ppo/actor_critic_final.pt")
print("\nSMOKE TEST PASSED")
