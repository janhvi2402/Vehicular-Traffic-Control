### 4x4 grid  (90 s plan (main 40/side 40), 60 s plan (main 30/side 20); algorithms: mean ± std over dqn: 3, ppo: 3, ac: 3, ca: 3 training seeds)

| Policy | Waiting time (veh*s, per-step mean) | Wait per vehicle (s) | Queue length (veh) | Throughput (veh) | Phase changes / junction / episode | Teleports / episode | Improvement vs Fixed-time 60 s: Waiting time (%) | Improvement vs Fixed-time 60 s: Wait per vehicle (%) | Improvement vs Fixed-time 60 s: Queue length (%) |
|---|---|---|---|---|---|---|---|---|---|
| Random | 244.53 (±7.40, eval eps) | 27.89 (±0.33, eval eps) | 26.73 (±0.35, eval eps) | 3770.20 (±3.49, eval eps) | 179.34 | 0.00 (±0.00, eval eps) | 31.0 | 17.8 | 21.4 |
| Fixed 10 s (tuned) | 98.80 (±0.74, eval eps) | 20.52 (±0.12, eval eps) | 18.36 (±0.12, eval eps) | 3781.40 (±3.26, eval eps) | 240.00 | 0.00 (±0.00, eval eps) | 72.1 | 39.5 | 46.0 |
| Fixed-time 60 s | 354.14 (±4.15, eval eps) | 33.91 (±0.26, eval eps) | 34.00 (±0.28, eval eps) | 3754.20 (±3.92, eval eps) | 120.00 | 0.00 (±0.00, eval eps) | 0.0 | -0.0 | 0.0 |
| Max-pressure | 82.25 (±1.41, eval eps) | 18.72 (±0.17, eval eps) | 16.26 (±0.18, eval eps) | 3779.70 (±3.63, eval eps) | 206.61 | 0.00 (±0.00, eval eps) | 76.8 | 44.8 | 52.2 |
| DQN | 104.83 ± 18.10 | 20.62 ± 1.29 | 18.36 ± 1.43 | 3778.20 ± 1.49 | 181.51 ± 11.64 | 0.00 ± 0.00 | 70.4 ± 5.1 | 39.2 ± 3.8 | 46.0 ± 4.2 |
| PPO | 91.80 ± 4.32 | 19.79 ± 0.40 | 17.46 ± 0.47 | 3778.87 ± 0.55 | 192.60 ± 6.11 | 0.00 ± 0.00 | 74.1 ± 1.2 | 41.6 ± 1.2 | 48.7 ± 1.4 |
| AC | 96.98 ± 2.20 | 19.73 ± 0.09 | 17.43 ± 0.11 | 3780.57 ± 0.60 | 207.84 ± 11.04 | 0.00 ± 0.00 | 72.6 ± 0.6 | 41.8 ± 0.3 | 48.7 ± 0.3 |
| CA | 146.81 ± 7.61 | 23.44 ± 0.35 | 21.52 ± 0.43 | 3775.10 ± 0.80 | 160.33 ± 13.34 | 0.00 ± 0.00 | 58.5 ± 2.1 | 30.9 ± 1.0 | 36.7 ± 1.3 |

### 3x3 grid  (90 s plan (main 40/side 40), 60 s plan (main 30/side 20); algorithms: mean ± std over dqn: 3, ppo: 3, ac: 3, ca: 3 training seeds)

| Policy | Waiting time (veh*s, per-step mean) | Wait per vehicle (s) | Queue length (veh) | Throughput (veh) | Phase changes / junction / episode | Teleports / episode | Improvement vs Fixed-time 60 s: Waiting time (%) | Improvement vs Fixed-time 60 s: Wait per vehicle (%) | Improvement vs Fixed-time 60 s: Queue length (%) |
|---|---|---|---|---|---|---|---|---|---|
| Random | 230.80 (±8.43, eval eps) | 24.89 (±0.31, eval eps) | 24.83 (±0.40, eval eps) | 3872.70 (±4.73, eval eps) | 180.14 | 0.00 (±0.00, eval eps) | 34.6 | 17.2 | 19.7 |
| Fixed 10 s (tuned) | 99.01 (±1.24, eval eps) | 18.50 (±0.15, eval eps) | 17.45 (±0.16, eval eps) | 3879.60 (±2.80, eval eps) | 240.00 | 0.00 (±0.00, eval eps) | 72.0 | 38.5 | 43.6 |
| Fixed-time 60 s | 353.14 (±6.46, eval eps) | 30.07 (±0.28, eval eps) | 30.92 (±0.30, eval eps) | 3868.30 (±2.19, eval eps) | 120.00 | 0.00 (±0.00, eval eps) | 0.0 | 0.0 | -0.0 |
| Max-pressure | 89.16 (±1.42, eval eps) | 17.38 (±0.16, eval eps) | 16.10 (±0.19, eval eps) | 3881.90 (±3.21, eval eps) | 223.39 | 0.00 (±0.00, eval eps) | 74.8 | 42.2 | 47.9 |
| DQN | 94.36 ± 7.29 | 17.78 ± 0.55 | 16.59 ± 0.62 | 3882.83 ± 1.89 | 206.50 ± 9.32 | 0.00 ± 0.00 | 73.3 ± 2.1 | 40.9 ± 1.8 | 46.4 ± 2.0 |
| PPO | 94.25 ± 0.67 | 17.81 ± 0.04 | 16.66 ± 0.05 | 3882.93 ± 0.45 | 216.78 ± 3.78 | 0.00 ± 0.00 | 73.3 ± 0.2 | 40.8 ± 0.1 | 46.1 ± 0.2 |
| AC | 96.12 ± 0.76 | 17.82 ± 0.02 | 16.65 ± 0.04 | 3881.50 ± 0.36 | 225.81 ± 5.14 | 0.00 ± 0.00 | 72.8 ± 0.2 | 40.8 ± 0.1 | 46.2 ± 0.1 |
| CA | 111.35 ± 2.41 | 18.98 ± 0.13 | 17.97 ± 0.19 | 3881.03 ± 1.07 | 191.03 ± 7.35 | 0.00 ± 0.00 | 68.5 ± 0.7 | 36.9 ± 0.4 | 41.9 ± 0.6 |

Notes: waiting time is the sum of SUMO per-vehicle waiting time over all incoming lanes (default 100 s memory), averaged over steps; wait per vehicle counts only completed trips; baselines are deterministic given the evaluation seeds.
