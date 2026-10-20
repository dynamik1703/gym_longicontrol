# Exact pinned-equation audit

## Lineage and source identity

- Preparation base: `8fb14a9861203713484f25a1c79e1643e9eb70ff`
- Completed collector replication: `38a8c7cc6789ae22d44de2b6f6cd3c84352c4b62`
- FSRL: `0.1.0` at `e056fc9498d5d037869533da7cf976acf462f918`
- Tianshou: `0.5.1`

Pinned installed source SHA-256 values:

- `fsrl/policy/sac_lag.py`: `816829e005851c26701a4c28b5e05b444ac8a2cb07cfe2f3894274130fe59da1`
- `fsrl/policy/lagrangian_base.py`: `3567ff5cd5dbb500927f5a64779d1bf76a81dfd7a9ceadaf9e2276b8c31f043c`
- `fsrl/policy/base_policy.py`: `a3635c1156d42382429bf002bdab10da65cc0038e4f75c8b8587ae7a2298d7f0`

The hashes were rechecked against the cached wheel built from the pinned revision.
The LongiControl vector-cost adapter splits FSRL's retained `info.cost` matrix into
one metric column per cost critic; it does not alter any Bellman equation.

## Exact Scalar SAC equations

Let `a' ~ pi(.|s')`, let `qS^- = min_j Q^-_S,j(s',a')`, and let
`ell' = log pi(a'|s')`. Pinned FSRL returns the following bootstrap for the single
Scalar double-Q system:

```text
B_S(s') = qS^- - alpha * ell'
```

For the ordinary nonterminal two-step case (`n_step=2`, `gamma=0.99`), its target is:

```text
Y_S,t = r_scalar,t + gamma*r_scalar,t+1 + gamma^2*B_S(s_t+2)
```

Pinned `nstep_return` shortens the accumulated metrics at a replay `done` boundary.
Its bootstrap value mask is false for physical `terminated` and remains true for a
pure time-limit `truncated` endpoint. Each of the two Q heads minimizes an unhalved
mean squared error against the same `Y_S`; the two losses are summed.

For a fresh sampled actor action `a ~ pi(.|s)` and
`qS = min_j Q_S,j(s,a)`, the exact minimized actor loss is:

```text
L_actor_S = mean(alpha * log pi(a|s) - qS)
```

With `log_alpha` as the optimized parameter, the exact automatic-alpha loss is:

```text
L_alpha_S = -mean(
    log_alpha * (stop_gradient(log pi(a|s)) + target_entropy)
)

alpha = exp(stop_gradient(log_alpha))
target_entropy = -1
```

Alpha is updated after the actor step. There is no reward-scale term in this loss.

## Exact Constrained V2 equations

Define separate target double-Q minima `qr^-`, `qs^-`, and `qd^-` for reward,
speed cost, and deadline cost. The pinned implementation subtracts the same entropy
term in **every** critic bootstrap:

```text
B_r(s') = qr^- - alpha * ell'
B_s(s') = qs^- - alpha * ell'
B_d(s') = qd^- - alpha * ell'
```

Thus the ordinary nonterminal two-step targets are exactly:

```text
Y_r,t = r_E,t        + gamma*r_E,t+1        + gamma^2*B_r(s_t+2)
Y_s,t = c_speed,t    + gamma*c_speed,t+1    + gamma^2*B_s(s_t+2)
Y_d,t = c_deadline,t + gamma*c_deadline,t+1 + gamma^2*B_d(s_t+2)
```

They use the same boundary handling and summed two-head MSE as Scalar SAC. The cost
critics are therefore entropy-augmented soft critics, not pure discounted physical
cost functions.

For current-policy action `a`, pinned SACLag first forms:

```text
L_reward = mean(alpha*log pi(a|s) - min_j Q_r,j(s,a))
L_safety = mean(lambda_s*min_j Q_s,j(s,a))
         + mean(lambda_d*min_j Q_d,j(s,a))
rho       = 1 / (1 + lambda_s + lambda_d)    # rescaling=True

L_actor_C = rho * (L_reward + L_safety)
```

Equivalently, its per-sample integrand is:

```text
(alpha*log pi - Q_r + lambda_s*Q_s + lambda_d*Q_d)
/ (1 + lambda_s + lambda_d)
```

The rescaling applies only to the total actor loss. It does not rescale critic losses
or the automatic-alpha loss. Constrained V2 uses the same exact alpha loss shown
above. Its PID multipliers are updated once per collected episode from the adapter's
separate **undiscounted** episodic cost sums versus zero limits; the actor and critics
use the discounted equations above.

## Why the critics do not collapse to one standard soft critic

Let `U = r_E - lambda_s*c_speed - lambda_d*c_deadline`. Combining the three
Constrained two-step targets gives a bootstrap proportional to:

```text
Q_r^- - lambda_s*Q_s^- - lambda_d*Q_d^-
+ (lambda_s + lambda_d - 1) * alpha*ell'
```

Standard SAC trained directly on `U` instead bootstraps with:

```text
Q_U^- - alpha*ell'
```

The entropy coefficients differ because FSRL inserted one identical entropy term in
each of three separately signed critic targets. Moreover, separately optimized
function approximators need not satisfy
`Q_U = Q_r - lambda_s*Q_s - lambda_d*Q_d`. Therefore neither Candidate N nor U can
make one standard soft critic identical to the Constrained critic combination.
This is an unavoidable algorithmic distinction in the study. Scalar SAC remains a
single standard double-Q system; no three-critic emulation was introduced.

## Reference and initialization preservation

All 18 retained C11/C29/C47 scientific checkpoints remain hash-verified. Each final
contains exactly 300,000 real transitions and 30,000 updates, so no Constrained
retraining is required. The Scalar compatibility constructor still retains the exact
historically initialized actor and first reward critic. Their bytes remain identical
to the corresponding Constrained initialization for seeds 11, 29, and 47; the Scalar
runtime contains no cost critic, PID optimizer, or Lagrange state.
