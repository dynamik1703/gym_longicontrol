# Canonical GCSL design

## Scientific question

Can direct goal-conditioned supervised learning from self-generated hindsight
trajectories acquire the canonical LongiControl task without demonstrations,
reward learning, value learning, or hand-authored successful behavior?

The mechanism is frozen before policy outcomes: GCSL converts an achieved
future into a direct action label. No policy-performance pilot selected any
semantic, architectural, sampler, or optimizer choice.

Training uses fixed canonical training streams with disclosed
historical-content overlap. Current-study evaluation integrity is specified
separately in `PROTOCOL.md`.

## State, goal, and success

The policy receives the original eight sensor-limited observations plus:

```text
position_m / 1000
previous_position_m / 1000
elapsed_time_s / 180
cumulative_max_speed_violation_m_s / 37
```

The resulting state is exactly 12D. It contains no route identity, future speed
profile, oracle action, `T_min`, deadline slack, or future horizon.

The 3D future goal is projected from Float64 physical outcomes before network
conversion:

```text
progress = min(position_m / 1000, 1)
timely   = 1[elapsed_time_s <= 140]
safe     = 1[cumulative_max_speed_violation_m_s <= 0]
goal     = [progress, timely, safe]
```

The canonical command is `[1,1,1]`. Late and unsafe futures remain late and
unsafe. The projection is the frozen projected-goal CRL mapping.

## Strict-future supervision

For source state `s_t`, executed action `a_t`, and a sampled future in the same
uninterrupted episode, replay supplies `project(outcome_t+k)`. A replay row
stores a pre-action source and its post-action outcome, so its own outcome is
the strict physical lag-one future. Later retained rows provide longer futures,
including the terminal outcome for the final pre-terminal action.

Sources are uniform with replacement. Conditional on a source, eligible lags
have probability proportional to `0.99**k`. Sampling never crosses a reset and
absolute sequence identifiers make ring overwrite safe. This CRL-matched
distribution is a registered adaptation from the official two-index sampler.

Collection and deterministic future evaluation query `[1,1,1]`; supervised
updates use only achieved projected futures. No demonstrations, successful
seeds, curricula, heuristic actions, synthetic successes, or Constrained data
are inserted.

## Continuous-action likelihood

The independent continuous adaptation is a diagonal Gaussian in pre-tanh
space. For one action dimension:

```text
z = atanh(clip(a, -1+1e-6, 1-1e-6))
mu, log_sigma = policy(state, goal)
NLL = 0.5*((z-mu)/sigma)^2 + log(sigma) + 0.5*log(2*pi)
      + log(1 - tanh(z)^2)
```

The Jacobian term is part of the exact transformed density. Stochastic
collection uses `tanh(mu + sigma*noise)` and deterministic evaluation uses
`tanh(mu)`. Endpoint action labels remain finite through inverse-tanh clipping.

Historical reward is replay provenance only. `GCSLBatch` contains state,
action, goal, and diagnostic lag. There is no Q-function, critic, Bellman
target, reward signal, deadline cost, speed cost, or energy signal.

## Policy

State and goal are concatenated, followed by Dense(256)-LayerNorm-SiLU and a
four-operation Dense(256)-LayerNorm-SiLU residual block. Linear mean and
log-standard-deviation heads parameterize the scalar action distribution;
log-standard deviation is bounded in `[-5,2]`. The exact trainable parameter
count is 270,338. Horizon conditioning is off.

## Optimizer-budget decision

The decision was frozen before any GCSL policy outcome.

Budget A provides 7,250 batch-256 updates (1,856,000 sampled tuples) after a
10,000-transition prefill, one update per 40 later transitions. It exactly
matches the prior CRL update-cycle schedule, isolates learner mechanism better,
and is computationally light. Its risk is supervised undertraining relative to
the official GCSL intensity and the canonical Constrained baseline.

Budget B would provide 30,000 batch-256 updates (7,680,000 tuples), matching
the canonical Constrained optimizer-step count more closely and reducing the
undertraining risk. However, optimizer steps are not comparable across SAC and
supervised maximum likelihood, B would depart fourfold from the directly
matched CRL schedule, and upstream GCSL's roughly one-update-per-transition
intensity would still be far higher than B.

Budget A is selected. The mechanistic CRL comparison is the closer controlled
comparison; 1.856 million labeled tuples is substantial, and the study forbids
a later update-count sweep. Severe undertraining remains a preregistered
interpretive possibility, assessed through learning/support diagnostics rather
than a performance pilot.

## Limitations

A hindsight tuple shows that an action occurred on a trajectory later reaching
the projected goal. It does not establish action optimality, uniqueness,
safety for another goal, or energy efficiency. Projection can collapse
physically different futures into a multimodal action condition, while the
policy distribution is unimodal. Low replay NLL need not prevent closed-loop
compounding error. These limitations are descriptive and cannot select new
hyperparameters.
