"""
environment/eligibility.py
Single definition of "this junction may execute a switch vote right now", shared by DQN and PPO:
    eligible = (not yellow) and (elapsed_green_seconds >= min_green)
derived from TrafficGridEnv node_features: column 9 = yellow flag, column 10 = elapsed / max_green.
This equals obs["action_mask"] exactly (checked by tests/test_masking.py). A 1e-4 s tolerance guards
against float32 error in feats[:,10] * max_green (observed error is ~4e-6 s).
"""
import numpy as np

YELLOW_COL, ELAPSED_COL = 9, 10


def _env_default(name):
    from environment.grid_env import GridEnvConfig  # single source of truth for min/max green
    return GridEnvConfig.__dataclass_fields__[name].default


def eligibility_from_features(node_features, min_green=None, max_green=None):
    """node_features: (..., 11) array -> bool array (...,)."""
    min_green = _env_default("min_green") if min_green is None else min_green
    max_green = _env_default("max_green") if max_green is None else max_green
    f = np.asarray(node_features)
    not_yellow = f[..., YELLOW_COL] < 0.5
    elapsed = f[..., ELAPSED_COL].astype(np.float64) * max_green
    return not_yellow & (elapsed >= min_green - 1e-4)
