# Known limitations and disclosed asymmetries

Setting: 4x4 grid (16 junctions, trained on) and 3x3 grid (9 junctions, never trained on); DQN and PPO, 3 training seeds each, 500 episodes
(360,000 environment steps) per run; evaluation is greedy on SUMO seeds 1000-1009.

## 1. Evaluation protocol

- **Same demand everywhere.** Training, the 4x4 evaluation and the baseline tuning all use one route file (4,027 vehicles, fixed departures and routes).
  The SUMO seed only changes driver imperfection. The 4x4 results are therefore in-distribution, not held out. The 3x3 grid is the topology test; it also
  changes the load (4,067 vehicles on 9 junctions vs 4,027 on 16, about 1.8x per junction). The x1.5 / x2.0 demand files are the only demand test, and only
  if they were evaluated. Error bars over evaluation episodes reflect SUMO noise only; the error bars over training seeds are the meaningful ones (n = 3, no
  significance claims).
- **Baselines.** Fixed-time 90 s and 60 s plans: E-W main road / N-S side road, 4 s yellow (which also acts as all-red), main/side split tuned per network on
  seeds 2000-2002 and evaluated on 1000-1009; all junctions start together with no offsets (no green wave). The real cycle is main + side + 10 s. The earlier tuned
  baseline (a 10 s threshold, which equals "always switch") is a very short cycle (about 30 s); it is kept in summary.json and in the objective-J table only.
  Max-pressure is local, with untuned margin 0.
- **Real signal timing in this environment.** Decisions come every 5 s and the first green after a yellow starts 1 s into a step, so a green is at least 11 s and
  at most 86 s (the environment forces the switch at max_green - decision_interval + 1). The first green of each episode starts at 0 and can reach 90 s.
  A literal 90 s main green is therefore impossible without changing max_green and the training setup.
- **Metrics.** mean_waiting_time is the sum of SUMO per-vehicle waiting time (default 100 s memory) over all incoming lanes, averaged over steps.
  mean_vehicle_wait and max_vehicle_wait count only vehicles that finished their trip (about 6% are still on the road at the end), so a congested policy can look
  better than it is; report throughput (= completed vehicles) next to them. Total episode reward is dominated by the switch penalty (the waiting term telescopes)
  and is not a performance metric.
- **Objective mismatch.** The agents are trained on -(change in waiting time) - 0.3 per phase change, i.e. on J = waiting per junction + 0.3 x (changes per junction per
  step) / (1 - gamma). A policy can lose on waiting time alone and win on J by switching less. Report both (scripts/compute_objective_J.py).
- **Diagnostics.** In the 86+ s bin of the switching-behaviour figure the switch is forced whatever the vote; the bar shows the vote probability.
  The max-pressure margin (0) was not tuned; the probe scripts use seed 1000, which is an evaluation seed, but only as a sanity check.

## 2. Training asymmetries between DQN and PPO (not forced to match)

| | DQN | PPO |
|---|---|---|
| Gradient steps per episode | 180 (one step per 4 env steps, batch 64) | 720 (4 epochs x 180 minibatches of 64) |
| Data | replay buffer, off-policy (400k transitions) | fresh on-policy rollout (11,520 transitions) |
| Activation / optimiser | ReLU, Adam 1e-3, Huber loss | Tanh, Adam 3e-4 |
| Gradient clipping | none | max_grad_norm 0.5 |
| Exploration while training | epsilon-greedy 1.0 -> 0.1 over the first 50 episodes | sampling from the policy, entropy bonus 0.01 |
| Value-loss scaling | n/a | scale_value_loss = True (PPO only) |

Matched on purpose: environment, reward and reward clip (50), gamma 0.99, hidden sizes (128, 128), 5 s decision interval, 500 episodes, action masking, seed scheme.
The reward clip binds on 2.4% of per-junction rewards (probe). Both algorithms bootstrap through the 3600 s time limit (DQN: done = terminated only;
PPO: adds gamma x V(next state) at the cutoff). Neither algorithm was hyper-parameter tuned, and only the final checkpoint is evaluated.
Training curves include each algorithm's own exploration and are not the comparison.

## 3. PPO: loss averaging and value-loss scaling

About half of all (step, junction) rows are ineligible to switch (yellow, or elapsed < min_green). For those rows the action is forced to hold, the switch logit is
masked, and P(hold) = 1 exactly, so they contribute exactly 0 to the policy-gradient and entropy terms. Each loss term is a plain `.mean()` over the minibatch:

    loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

- policy_loss and entropy are scaled by f_e, the eligible fraction of rows; value_loss uses all rows.
- Eligible fraction f_e of hand-coded policies (1200 s episodes, seed 1000): always-switch (10 s) 0.333, fixed 15 s 0.500, fixed 30 s 0.708,
  hold-forever 0.892. The value of the trained PPO policy is measured by scripts/measure_eligible_fraction.py and reported with the results.
- **Without scaling**, the value term is weighted value_coeff / f_e relative to the policy term (about 1.0-1.5 for a policy that switches often, i.e. 2-3x the nominal 0.5).
  The gradient probe on an untrained network (one rollout, value targets with std 28.6) found the value gradient 295.6x the policy gradient, and with max_grad_norm 0.5
  the clip factor 0.014 left a policy gradient of about 0.0017.
- **With scale_value_loss = True** (used for all PPO runs) the value loss is divided by max(std of the rollout returns, 1)^2, which divides its gradient by the same factor
  (about 818 for std 28.6): the effective value weight becomes value_coeff / (f_e x std^2) and, from the probe's printed numbers, the value gradient falls to roughly 0.36x the
  policy gradient, below the clip. The logged value_loss is the scaled loss. Advantage normalisation uses eligible rows only. The surrogate and entropy terms were
  deliberately not re-normalised over eligible rows (the follow-up experiment would be `.sum()/n_eligible`).