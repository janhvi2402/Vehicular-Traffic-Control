"""
compare_all_algorithms.py
==========================
Combines FOUR already-computed evaluate.py summary.json files (Q-Learning,
DQN, PPO, PPO_Routing) into the single master comparison table + plot used
in the report (Table 8.6-style):

    Method               Avg. Waiting Time (s)   Std. Dev. (s)   Improvement (%)
    Fixed-Time Baseline           ...                 ...              -
    Random                        ...                 ...             ...
    Q-Learning                    ...                 ...             ...
    DQN                           ...                 ...             ...
    PPO                           ...                 ...             ...
    PPO_Routing                   ...                 ...             ...

IMPORTANT: only ONE of these summary.json files supplies the "Fixed-Time
Baseline" and "Random" rows (BASELINE_SUMMARY below) -- see
environment.eval_common.build_multi_algo_table's docstring for why using a
different baseline per algorithm would make the Improvement (%) column
incomparable across rows. All four evaluate.py runs must also have used the
SAME net_file/route_file/seeds, or this table compares different conditions
rather than different algorithms under the same one.

No command-line args -- everything is hardcoded below, same as
make_topology_plot.py. Just edit the five paths to match your own folders,
then run:

    python compare_all_algorithms.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from environment.eval_common import (  # noqa: E402
    build_multi_algo_table, print_results_table, save_results_table_csv,
    plot_multi_algo_comparison,
)

# ------------------------------------------------------------------ #
# EDIT THESE PATHS
# ------------------------------------------------------------------ #
# FIX: previously pointed at .../eval/summary.json (plain "eval"), which
# nothing actually produces -- run_matched_eval.sh only ever writes to
# eval_4x4/ and eval_3x3/ subfolders, and running dqn.evaluate/ppo.evaluate
# a THIRD time with no --out override just to get a plain "eval" folder
# would be redundant work for data you already have. Pointing DQN and PPO
# at their existing eval_4x4/ output (the standard trained-network
# evaluation, not the 3x3 generalization test) reuses what
# run_matched_eval.sh already produced -- no extra runs needed for those
# two. Q-Learning and PPO_Routing don't have an evaluate.py in this
# project yet (see environment/eval_common.py's own docstring, which
# anticipates them); once you build one, have it write to the same
# eval_4x4/ naming convention so it drops in here without another edit.
BASELINE_SUMMARY = os.path.join(PROJECT_ROOT, "runs", "dqn_run_matched500", "eval_4x4", "summary.json")
ALGO_SUMMARIES = {
    "Q-Learning":  os.path.join(PROJECT_ROOT, "runs", "qlearning_run_matched500", "eval_4x4", "summary.json"),
    "DQN":         os.path.join(PROJECT_ROOT, "runs", "dqn_run_matched500",       "eval_4x4", "summary.json"),
    "PPO":         os.path.join(PROJECT_ROOT, "runs", "ppo_run_matched500",       "eval_4x4", "summary.json"),
    "PPO_Routing": os.path.join(PROJECT_ROOT, "runs", "ppo_routing_run_matched500", "eval_4x4", "summary.json"),
}
OUT_DIR = os.path.join(PROJECT_ROOT, "runs", "overall_comparison")
# "mean_waiting_time" = grid-wide per-step average (what the earlier 2-junction
# report called "Avg. Waiting Time"). Use "mean_vehicle_wait" instead for the
# per-vehicle whole-trip average -- these are NOT the same number, see
# grid_env.py's _update_vehicle_wait_tracking docstring.
METRIC = "mean_waiting_time"
# ------------------------------------------------------------------ #

os.makedirs(OUT_DIR, exist_ok=True)

_all_paths = {"BASELINE_SUMMARY": BASELINE_SUMMARY, **ALGO_SUMMARIES}
for label, path in _all_paths.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected a summary.json for {label} at:\n  {path}\n"
            f"Run that algorithm's evaluate.py first (with the SAME "
            f"net_file/route_file/seeds as the other three), or edit the "
            f"path near the top of this file."
        )

with open(BASELINE_SUMMARY) as f:
    baseline_summary = json.load(f)

algo_summaries = {}
for label, path in ALGO_SUMMARIES.items():
    with open(path) as f:
        algo_summaries[label] = json.load(f)

rows = build_multi_algo_table(baseline_summary, algo_summaries, metric=METRIC)

metric_label = "Avg. Waiting Time (s)" if METRIC == "mean_waiting_time" else "Avg. Wait/Vehicle (s)"
print_results_table(
    rows, metric_label=metric_label,
    title="Overall Performance Comparison Against Fixed-Time Baseline",
)
save_results_table_csv(rows, os.path.join(OUT_DIR, "overall_comparison.csv"))
plot_multi_algo_comparison(
    rows, os.path.join(OUT_DIR, "overall_comparison.png"), metric_label=metric_label
)