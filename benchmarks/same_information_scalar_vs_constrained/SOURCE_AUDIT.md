# Source audit

## Lineage and immutable reference

- Preparation base: `8fb14a9861203713484f25a1c79e1643e9eb70ff`
- Completed collector replication: `38a8c7cc6789ae22d44de2b6f6cd3c84352c4b62`
- FSRL version/revision: `0.1.0` / `e056fc9498d5d037869533da7cf976acf462f918`
- Tianshou: `0.5.1`

Pinned installed source SHA-256 values:

- `fsrl/policy/sac_lag.py`: `816829e005851c26701a4c28b5e05b444ac8a2cb07cfe2f3894274130fe59da1`
- `fsrl/policy/lagrangian_base.py`: `3567ff5cd5dbb500927f5a64779d1bf76a81dfd7a9ceadaf9e2276b8c31f043c`
- `fsrl/policy/base_policy.py`: `a3635c1156d42382429bf002bdab10da65cc0038e4f75c8b8587ae7a2298d7f0`

All T11/T29/T47 scientific checkpoints at 50k, 100k, 150k, 200k, 250k,
and 300k exist and match their frozen SHA-256 metadata. Each final payload records
exactly 300,000 real transitions and 30,000 RL updates. The retained learning curves
are therefore reusable; matched Constrained retraining is not required.

Final checkpoint SHA-256 values:

- C11: `e2047a6e9bf217d020e708d32104f24e35d8e34dd7d4900cc87d29cdc069ff1d`
- C29: `d5284426d802163fe8f3045789293e721e81f462771988fe648cf909e8f575db`
- C47: `e828dc0babb42e20b9f5f33bdedbc300bafe21ccdedee101b8ba84b715919e35`

## Exact constrained actor equation

For sampled action `a ~ pi(.|s)`, pinned FSRL forms the minimum of each double-Q
system and minimizes

```text
L_C(pi) = E[
  (alpha * log pi(a|s)
   - Q_reward(s,a)
   + lambda_speed * Q_speed(s,a)
   + lambda_deadline * Q_deadline(s,a))
  / (1 + lambda_speed + lambda_deadline)
]
```

The denominator is exactly `1 / (sum(lambdas) + 1)` when `rescaling=True`. Thus
the proposed coefficients are the exact fixed normalized reward/cost coefficients
for the actor's Q combination. The actor entropy term is also divided by the same
normalizer in Constrained V2.

## Exact entropy and critic semantics

Pinned FSRL applies the same soft bootstrap to every critic `i`, including each cost
critic:

```text
B_i(s') = min_j Q_i,target,j(s', a') - alpha * log pi(a'|s')

G_i,t^(2) = m_i,t + gamma * m_i,t+1
             + gamma^2 * B_i(s_t+2)             (unless terminated)
```

Here `m_0 = r_E`, `m_1 = c_speed`, and `m_2 = c_deadline`. Consequently the cost
critics are entropy-augmented; they are not pure discounted physical-cost values.

The one-system Scalar policy uses standard SAC in the same pinned policy class:

```text
r_scalar = w_reward*r_E - w_speed*c_speed - w_deadline*c_deadline

B_scalar(s') = min_j Q_scalar,target,j(s', a')
               - alpha * log pi(a'|s')

L_S(pi) = E[alpha * log pi(a|s) - min_j Q_scalar,j(s,a)]
```

Therefore immediate information and coefficients match, but entropy semantics do not
collapse to the constrained combination. Combining the three constrained targets
would weight their three identical entropy bootstraps with signs
`(+1, -lambda_speed, -lambda_deadline) / Z`, whereas standard scalar SAC has one
unmodified entropy bootstrap. Likewise, Constrained scales its actor entropy by
`1/Z`; Scalar does not. This difference is allowed and is part of the algorithmic
structure under test. Standard SAC was not modified to manufacture equivalence.

## Implementation choice

The Scalar arm uses the pinned FSRL/Tianshou stack rather than the older custom or
SB3 scalar implementations. It thereby matches actor network, double-Q architecture,
optimizers, alpha, n-step handling, replay, action scaling, collector, and evaluation
semantics. Runtime invariants require one double-Q critic system, zero cost critics,
zero PID optimizers, and `use_lagrangian=False`.

Pinned FSRL couples initialization to the number of networks allocated before its
orthogonal initialization pass. The compatibility factory reproduces the immutable
historical three-critic constructor, retains the actor and first reward critic, and
discards both temporary cost critics before constructing Scalar SAC. Retained actor
and reward-critic hashes match exactly for all three seeds; no cost critic or PID state
exists in the Scalar learner or checkpoint.
