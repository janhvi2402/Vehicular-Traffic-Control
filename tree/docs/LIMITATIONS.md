# Known limitations and disclosed asymmetries

## PPO: loss averaging over rows that carry no gradient

About half of all (step, junction) rows are ineligible to switch (yellow, or elapsed < min_green). For those rows
the action is forced to hold, the switch logit is masked, and P(hold) = 1 exactly, so they contribute exactly 0 to
the policy-gradient term and exactly 0 to the entropy term. In `ppo/train.py` the three loss terms are each a plain
`.mean()` over the minibatch:

    loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

- `policy_loss` and `entropy` are therefore scaled by f_e, the eligible fraction of rows (ineligible rows add zeros
  to the sum but still count in the denominator).
- `value_loss` uses all rows (the value head is trained on every state), so it is not scaled.

Hence, relative to the policy term, the value term is weighted `value_coeff / f_e`, not `value_coeff`:

| policy (1200 s episode, seed 1000)            | f_e   | effective value_coeff / f_e (nominal 0.5) |
|-----------------------------------------------|-------|--------------------------------------------|
| always-switch (10 s cycle)                    | 0.333 | 1.50 |
| PPO smoke checkpoint, greedy                  | 0.333 | 1.50 |
| PPO smoke checkpoint, stochastic (rollout)    | 0.440 | 1.14 |
| fixed 15 s cycle                              | 0.500 | 1.00 |
| fixed 30 s cycle                              | 0.708 | 0.71 |
| hold forever (forced switches only)           | 0.892 | 0.56 |

(`scripts/measure_eligible_fraction.py`. A previous DQN checkpoint's greedy episode had 5839 eligible rows of
720 x 16 = 11,520, f_e = 0.507.) For a policy that switches often, which is what training pushes towards here, the
effective ratio is about 1.0 to 1.5, i.e. 2x to 3x the nominal 0.5. The entropy bonus is scaled by the same f_e as
the policy term, so entropy-relative-to-policy is unchanged; entropy and policy are both down-weighted relative to
the value term by 1/f_e.

Advantage normalisation uses eligible rows only (mean/std over rows that actually have a policy gradient). This
keeps the advantages of those rows at unit scale; it does not change the value/policy weighting above, which comes
from the `.mean()` over all rows. The paper's Atari setting (c1 = 1, c2 = 0.01, quoted from memory of Schulman et
al. 2017, Table 5 -- not re-checked here) is a different network and loss-sharing configuration; the point here is only
that the nominal `value_coeff = 0.5` is not the weight the policy term actually sees. The surrogate and entropy
terms were deliberately NOT re-normalised over eligible rows (it would change PPO's effective learning signal in a
way DQN has no analogue for); if a reviewer asks, the follow-up experiment is `.sum()/n_eligible` for those two terms.

(Step 4b appends the remaining disclosed asymmetries here: epochs vs train_frequency, ReLU vs Tanh, learning rate.)
