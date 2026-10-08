# GCSL V1 design

## Scientific question

Can direct goal-conditioned supervised learning from self-generated hindsight
trajectories acquire the canonical LongiControl task more reliably than
reward-based or contrastive learning, without demonstrations or a hand-designed
dense task reward?

The comparison is mechanistic: HER converts a future to a sparse reward/value
target, CRL converts it to a contrastive association, and GCSL converts it to a
direct action label. Projected-goal CRL results were unknown when this design
was frozen.

Prior LongiControl context is Binary success SAC 0/27, goal-conditioned SAC
without HER 0/27, and goal-conditioned SAC with HER 0/27. These are contextual
results rather than controlled learner comparisons. The projected-goal CRL
study was running and its outcomes were deliberately not inspected.

## State, goal and success

The policy receives the original eight sensor-limited observations followed by
normalized current position, previous position, absolute elapsed time and
cumulative maximum speed violation. The 12-vector exposes neither track
identity nor a future speed profile, oracle action, `T_min`, or deadline slack.

The projected future outcome is computed from raw Float64 physical values:

```text
progress = min(position_m / 1000, 1)
timely   = 1[elapsed_time_s <= 140]
safe     = 1[cumulative_max_speed_violation_m_s <= 0]
g        = [progress, timely, safe]
```

Only a real first arrival that is timely and exactly compliant projects to the
canonical `[1,1,1]`. Late and unsafe futures remain late and unsafe. The mapping
is imported directly from the frozen CRL semantic module.

## Supervised objective

For recorded source state `s_t`, executed action `a_t`, and a sampled strict
future `t+k` in the same uninterrupted episode:

```text
g_tk = project(outcome_t+k)
z = atanh(clip(a_t, -1+epsilon, 1-epsilon))
mu, log_sigma = policy(s_t, g_tk)
```

For one action dimension, the transformed density and minimized loss are:

```text
pi(a|s,g) = Normal(z; mu, sigma) / (1 - tanh(z)^2)
NLL = 0.5*((z-mu)/sigma)^2 + log(sigma) + 0.5*log(2*pi)
      + log(1 - tanh(z)^2)
L = mean_batch(NLL)
```

`epsilon=1e-6` gives exact endpoint records a finite inverse-tanh
representation. At collection, `z = mu + sigma*noise` and `a=tanh(z)`, so every
sample lies in `[-1,1]`. Evaluation uses `tanh(mu)`. MSE is only a descriptive
diagnostic and never the optimized loss.

Historical rewards are retained in replay for provenance but are absent from
`GCSLBatch`, the policy and the loss. There is no Q-function, contrastive
critic, Bellman target, reference goal, or energy signal.

## Horizon decision

The primary policy is Markovian and receives no future-lag encoding. This
matches the released principal configuration (`max_horizon=None`); horizon
conditioning was an ablation. Replay still records `k` and reports `0.1*k`
seconds. Absolute elapsed time is part of state, future lag is replay metadata,
and the 140-second deadline is a task predicate. They are tested as distinct
quantities.

## Sampler and goal distributions

One source transition is sampled uniformly with replacement. A replay row is
`(state_t, action_t, outcome_t+1)`, so its own post-action outcome is the
strict one-step future; later rows provide longer futures, including the true
terminal outcome for the final pre-terminal action. Conditional on the source,
a future in its uninterrupted episode is drawn with probability

```text
P(k | t, episode) = 0.99^k / sum_j 0.99^j .
```

This is the frozen CRL-matched discounted-future adaptation, not the official
two-uniform-index sampler. It permits all physical lags from one step onward,
never crosses a reset and never creates a post-terminal state/action row.

- Real collection goal: `[1,1,1]`.
- Training goal: only the sampled actually achieved projected future.
- Deterministic Development/Validation goal: `[1,1,1]`.

Initially no canonical target may exist. The policy is still queryable at the
canonical command, but behavior may require unsupported extrapolation. No
success is inserted, seeded, or relabeled upward.

## Architecture

The actor matches the projected-goal CRL depth-4 actor trunk: concatenate the
12-state and 3-goal vectors, apply Dense(256)-LayerNorm-SiLU, then four
Dense(256)-LayerNorm-SiLU operations with one residual addition. Separate
linear heads produce pre-tanh mean and bounded log standard deviation in
`[-5,2]`. There is no critic. The measured trainable parameter count is recorded
in `RESOURCE_REPORT.md`.

## Interpretation and limitations

A hindsight tuple only says that `a_t` occurred on a trajectory that later
reached `g`. It does not show that the action was optimal, uniquely correct,
fastest, safe for another future, or energy efficient. GCSL imitates its own
past behavior. The projected goal collapses many physical outcomes, potentially
making conditional actions multimodal; a unimodal Gaussian can average
incompatible labels. Unsafe outcomes are honestly represented and may induce
unsafe self-imitation for unsafe goals. Low replay NLL need not prevent
closed-loop compounding error.

The diagnostics distinguish no exploration, local hindsight learning without
canonical support, action averaging, unsafe self-imitation, distribution shift,
and successful iterative bootstrap. They are descriptive and cannot select
hyperparameters.
