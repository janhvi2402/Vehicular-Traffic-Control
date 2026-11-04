# AC and CA: design notes for the methods / limitations section

## What is compared
| key | algorithm | folder | setting |
|---|---|---|---|
| dqn | DQN (Mnih et al. 2015) | dqn/ | discounted, off-policy, replay |
| ppo | PPO (Schulman et al. 2017) | ppo/ | discounted, on-policy, GAE |
| ac  | two-timescale actor-critic, critic fast (Bhatnagar, Borkar & Guin) | ac/ | discounted, online one-step |
| ca  | two-timescale critic-actor, average reward (Panda & Bhatnagar, AAAI-25) | ca/ | average reward, online one-step |
`python -m ac.train --variant ca` is the discounted critic-actor of Bhatnagar-Borkar-Guin, kept only as an ablation
(runs/ca_discounted_seed<N>); it is not part of scripts/*.

## Identical for all four algorithms (enforced in code)
11-dim per-junction input, (128, 128) hidden layers, one parameter-shared network applied to all junctions, decision_interval 5 s,
episode 3600 s, reward = info["agent_rewards"] clipped to +-50, 500 episodes = 360,000 env steps, switch votes masked by
environment.eligibility (the executed action is learned from), 3600 s time limit bootstraps through (only `terminated` cuts it),
training env seed = training_env_seed(train_seed, episode), evaluation greedy (DQN epsilon = 0) on seeds 1000.. with the same
baselines (environment/eval_protocol.py is the evaluate.py body of dqn/ppo), one train_log.csv row per episode.

## Disclosed asymmetries (not forced to match)
* Update schedule: DQN 1 gradient step / 4 env steps (batch 64, replay); PPO 4 epochs x minibatches of 64 per 720-step rollout;
  AC/CA one gradient step per env step on the 16 current junction transitions, no replay, no GAE, no advantage normalisation.
* Optimiser/step size: DQN Adam 1e-3 constant, PPO Adam 3e-4 constant, AC/CA Adam with the papers' decaying step sizes
  (base 3e-4, decay every 0.8 % of training).
* Activation: DQN ReLU; PPO, AC, CA tanh.
* PPO divides the value loss by the return std (scale_value_loss); AC/CA use the raw TD error (clipped rewards only).
* CA has no discount factor (average reward) and keeps a separate running average reward L per junction; its default exponents
  (nu 0.5, sigma 0.51) give almost no timescale separation at this training length (step sizes differ by ~5 % at the end).
* Entropy bonus 0.01 (not in the AC/CA papers) is added for AC and CA, as in PPO; averaged over all junction rows (ineligible rows contribute 0).
