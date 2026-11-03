"""
tests/test_fixed_time_policy.py -- Play button, no SUMO simulation, takes a second. Checks the fixed-time plans WITHOUT running SUMO:
  * the policy only votes on green phases, main (phase 0) and side (phase 2) use their own thresholds;
  * on a toy timeline with a 4 s yellow the realised greens are main+1 / side+1 (the very first green of an episode is `main`);
  * every candidate split in scripts/sweep_fixed_time_split.py has cycle = main + side + 10 = 90 s or 60 s.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_fixed_time import fixed_time_policy
from sweep_fixed_time_split import CANDIDATES


def obs(phase, elapsed, max_green=90):
    f = np.zeros((1, 11)); f[0, 8] = phase / 3; f[0, 10] = elapsed / max_green
    return {"node_features": f}


def test_votes_only_on_green_with_own_threshold():
    pol = fixed_time_policy(50, 30, 90)
    assert pol(obs(0, 46))[0] == 0 and pol(obs(0, 51))[0] == 1       # main needs 50
    assert pol(obs(2, 26))[0] == 0 and pol(obs(2, 31))[0] == 1       # side needs 30
    assert pol(obs(1, 80))[0] == 0 and pol(obs(3, 80))[0] == 0       # never during yellow


def test_realised_greens_on_toy_timeline():
    for main_g, side_g in ((50, 30), (30, 20), (40, 40), (25, 25)):
        pol = fixed_time_policy(main_g, side_g, 90)
        phase, elapsed, greens = 0, 0, []
        while len(greens) < 6:
            if pol(obs(phase, elapsed))[0] == 1:
                greens.append((phase, elapsed)); phase = 2 if phase == 0 else 0; elapsed = 1   # 4 s yellow leaves 1 s of the 5 s step
            else:
                elapsed += 5
        assert greens[0] == (0, main_g)
        assert all(g == (0, main_g + 1) for g in greens[2::2]) and all(g == (2, side_g + 1) for g in greens[1::2]), greens


def test_candidate_cycles():
    for cyc, cands in CANDIDATES.items():
        assert all(m + s + 10 == cyc for m, s in cands), (cyc, cands)


if __name__ == "__main__":
    for t in (test_votes_only_on_green_with_own_threshold, test_realised_greens_on_toy_timeline, test_candidate_cycles):
        t(); print("PASS", t.__name__)
