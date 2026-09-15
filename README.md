# DQN for TrafficGridEnv — README

This delivers `train.py`, `test.py`, `evaluate.py`, and `switch_timing_plot.py`
for training and analyzing a DQN agent on your 4x4 SUMO grid, plus the shared
`environment/grid_env.py` (unchanged from before). Every file here has been
run end-to-end in a sandbox with SUMO installed — not just written and hoped
to work. `sample_results/` contains real output from a validation training
run (25,000 steps / 33 episodes) so you can see what correct output looks
like before running your own full-scale training.

## File layout (drop into your repo as-is)

```
vehicular-traffic-control/
├── environment/
│   ├── __init__.py
│   └── grid_env.py
├── dqn/
│   ├── __init__.py
│   ├── config.py              <- all hyperparameters + justification
│   ├── q_network.py           <- Q-network + replay buffer (Algorithm 1)
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── switch_timing_plot.py
├── sumo_4x4_network/
│   ├── grid4x4.net.xml
│   └── routes.rou.xml
└── runs/                       <- created automatically, checkpoints + logs go here
```

## Setup

```bash
pip install torch gymnasium traci sumolib matplotlib pandas
# SUMO_HOME must point at your SUMO install, and `sumo` must be on PATH
export SUMO_HOME=/usr/share/sumo   # adjust to your system
```

## Hyperparameters — what's from the paper, what's adapted

Every value in `dqn/config.py` has an inline comment justifying it against
Mnih et al., 2015. Summary for the methods section:

| Setting | Paper (Atari) | This implementation | Why |
|---|---|---|---|
| Network | Deep CNN | 2-layer MLP (128, 128) | Input is an 11-dim hand-engineered vector, not pixels — no spatial structure to convolve |
| Optimizer | RMSProp, lr=0.00025 | Adam, lr=1e-3 | Standard modern substitute for RMSProp on small MLPs; RMSProp still available via `cfg.optimizer="rmsprop"` for an ablation |
| Loss | Huber / [-1,1] error clip | Identical (`nn.SmoothL1Loss`) | Kept exactly as specified |
| Discount γ | 0.99 | 0.99 | Unchanged — same long-horizon credit assignment regime |
| Replay capacity | 1,000,000 | 400,000 | Scaled to our much shorter total training budget |
| Replay start size | 50,000 | 5,000 | Scaled proportionally with replay capacity |
| Batch size | 32 | 64 | Each env step yields 16 transitions (one per junction) instead of Atari's 1, so more data is available per unit time |
| Target update freq | every 10,000 param updates | every 500 | Scaled down for a far shorter training run |
| Epsilon schedule | 1.0→0.1 over first 2% of training | 1.0→0.1 over first 10% | More generous anneal since our absolute training budget is much smaller; final value unchanged |
| Eval epsilon | 0.05 | 0.05 | Unchanged — paper's exact evaluation protocol |
| Reward clipping | hard clip to {-1,0,+1} | soft clip to [-50,50] | Paper's clipping was for cross-task normalization across 49 different games; we only train on one task, so hard clipping would destroy real magnitude information |

**Cite this table's reasoning directly if asked why any hyperparameter
differs from the paper — this was a deliberate, documented adaptation, not
an oversight.**

## Usage

### Train
```bash
python -m dqn.train --steps 150000 --out runs/dqn_run1
```
For a run split across multiple sessions (or to fit a time-limited
environment), use `--chunk_steps` + `--resume`:
```bash
python -m dqn.train --steps 150000 --chunk_steps 20000 --out runs/dqn_run1
# ... then continue from where it left off (see runs/dqn_run1/resume_state.json):
python -m dqn.train --steps 150000 --chunk_steps 20000 --out runs/dqn_run1 \
    --resume runs/dqn_run1/qnet_latest.pt --start_step 20000 --start_episode <N>
```
This resumes both the network weights AND the replay buffer (saved to
`replay_buffer.pkl`), so no training time is wasted re-warming the buffer.

### Quick sanity check
```bash
python -m dqn.test --checkpoint runs/dqn_run1/qnet_final.pt
```

### Full evaluation vs. baselines
```bash
python -m dqn.evaluate --checkpoint runs/dqn_run1/qnet_final.pt --episodes 30 --out runs/dqn_run1/eval
```
Compares DQN against two baselines, mirroring the paper's Fig. 3
normalized-performance methodology:
- **Random** = paper's 0% reference point (uniform random switch votes)
- **Fixed-cycle** = stand-in for the paper's human reference (a
  traffic-engineering-standard fixed-time controller). **Important**: by
  default this now *sweeps* several cycle lengths (15/20/25/30/45/60s) on a
  probe episode and picks the best-performing one as the reported baseline,
  because we found fixed-cycle performance is highly sensitive to cycle
  length on this network (15s cycles roughly halved waiting time versus
  30s). Comparing against an untuned, arbitrary cycle length would not be a
  defensible baseline for the paper. Override with `--fixed_cycle_seconds`
  if you want a specific value instead.

**Caveat you should carry into the paper**: unlike Atari (where a human
always beats random play), there's no guarantee a fixed-time controller
beats random switching — if it doesn't, the normalized percentage can come
out negative or behave unintuitively even when DQN wins on every raw
metric. `evaluate.py` prints an explicit warning when this happens. Lead
with the raw metrics (`mean_waiting_time`, `mean_queue_length`,
`throughput`) as your primary reported numbers; treat the normalized
percentage as a secondary, paper-methodology-matching statistic.

### Switch-timing visualization
```bash
python -m dqn.switch_timing_plot --policy dqn --checkpoint runs/dqn_run1/qnet_final.pt --out runs/dqn_run1/timing
python -m dqn.switch_timing_plot --policy fixed_cycle --fixed_cycle_seconds 15 --out runs/timing_fixed
python -m dqn.switch_timing_plot --policy random --out runs/timing_random
```
Produces 4 figures: phase timeline (Gantt chart, one row per junction),
green-duration histogram (with min/max_green reference lines), grid-wide
congestion time series, and forced-switch counts per junction. Useful
figures for a "policy behavior" section of the paper, and directly
comparable across policies by running all three.

## Validated results (sample_results/, 25,000-step training run)

This is a **shortened validation run**, not a paper-scale result — it exists
to prove the pipeline is correct, not to be cited as your final numbers.
Run the full 150,000+ step training on your own hardware for real results.

- `training_curves.png` — loss drops from ~17 to ~1.5-2 over training;
  waiting time spikes during the high-epsilon exploration phase then drops
  sharply once epsilon anneals to 0.1, exactly the expected learning curve.
- `baseline_comparison.png` / `summary.json` — DQN beat both baselines on
  every raw metric (mean waiting time, mean queue length, throughput,
  total reward) after only 25,000 steps; normalized performance vs. the
  swept fixed-cycle baseline (15s cycle selected) came out to 86.8%.
- `green_duration_histogram.png` — the trained policy mostly switches
  around 10-15s (proactively, not waiting for the 90s cap) with only 3
  hard-cap-forced switches across the entire 3600s episode — a sign it
  learned a real switching policy rather than defaulting to the cap.
- `phase_timeline.png`, `congestion_timeseries.png` — full-episode signal
  and congestion behavior for a qualitative "how does it behave" figure.

## A note on reproducibility

Every number above came from an actual run in a working SUMO installation,
not a hand-estimate. If you rerun with the same seeds you should get very
close to (not necessarily bit-identical to, since TraCI/SUMO's own internal
RNG isn't fully pinned by our seeding alone) these numbers.
