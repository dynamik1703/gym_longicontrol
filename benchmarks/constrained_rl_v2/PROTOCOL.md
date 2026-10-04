# Constrained RL V2 benchmark protocol

Status: preregistered before implementation smoke tests and before all main V2
training runs. Date: 2026-10-02. Repository base revision:
`bc56a741c33cfb7a6b6681c18753a23dfd1bf5ba`.

Canonical configuration SHA-256:
`89d7142cc84ba60d98075313220b331b60c2c41997b6c65b11adeb1b08712060`.

Constrained V1 is a frozen research artifact. V2 changes only the representation of
the completion/deadline constraint. It does not change the simulator, observation,
action, energy objective, speed constraint, external evaluator, algorithm family,
optimizer settings, data splits, or physical interaction budget.

## Question and V1 diagnosis

V1 learned standstill for all three seeds: validation RSR and completion were 0/27,
while speed compliance was 27/27. Its task cost was a single binary failure value at
episode termination. The registered diagnosis is that this signal arrives too late to
give the actor useful completion/deadline credit. V2 asks whether a dense physical
deadline signal removes that failure while retaining explicit constraints.

The causal claim remains narrow: V2 compares one cost representation with frozen V1.
It cannot establish that all constrained-RL or all dense costs behave similarly.

## FSRL return and constraint semantics

The installed implementation is FSRL 0.1.0 at revision
`e056fc9498d5d037869533da7cf976acf462f918`, with Tianshou 0.5.1. Direct source
inspection established the following behavior before selecting the V2 cost:

1. `FastCollector` reports raw episodic reward by summing step rewards without
   discounting. The LongiControl adapter separately reconstructs each raw episodic
   cost sum because the pinned collector otherwise collapses cost columns.
2. `SACLagrangian.process_fn` calls `compute_nstep_returns` for reward and every cost
   critic. All critic targets use the same `gamma = 0.99` and `n_step = 2`.
3. For metric `m`, the two-step target is of the form
   `m_t + gamma*m_(t+1) + gamma^2*Q_target(s_(t+2), a_(t+2))`, shortened at episode
   boundaries. The pinned SACLag target subtracts the entropy term from every target
   critic, including cost critics.
4. The PID-Lagrange update does not use a discounted critic return. Once per collected
   episode, it compares the adapter's exact undiscounted episodic cost sums with the
   corresponding cost limits. The actor nevertheless receives constraint gradients
   through the discounted cost critics.

Thus FSRL has mixed but explicit semantics here: discounted soft critic estimates and
undiscounted episodic PID constraint measurements.

## Rejected raw elapsed-time candidate

For `dt = 0.1 s` and `gamma = 0.99`, a constant time cost has discounted value

```text
C(T) = 0.1 * (1 - 0.99^(T / 0.1)) / (1 - 0.99).
```

The analytically computed values are:

| Physical duration | Discounted cost |
|---:|---:|
| 100 s | 9.999568287526 |
| 120 s | 9.999942159303 |
| 140 s | 9.999992250522 |
| 160 s | 9.999998961727 |
| 180 s | 9.999999860893 |

The 140-to-160 s gap is only `6.71120525455e-6`; the 160-to-180 s gap is
`8.99165097910e-7`. A naive critic limit of 140 would also be dimensionally wrong,
because the critic asymptote is 10. Although the PID could compare the undiscounted
episode sum to 140 s, the actor's cost critic would see an almost action-independent,
saturated signal. The stop gate is therefore triggered: raw `time_cost_t = dt` is not
the V2 formulation.

## Frozen V2 CMDP formulation

### Objective: unchanged from V1

At every native simulator transition:

```text
objective_t = -signed_step_energy_kwh / 0.25 kWh
```

Regeneration remains signed. There is no progress reward, time penalty, speed penalty,
completion reward, or terminal reward in the objective.

### Constraint 1: unchanged speed integral

Using right-endpoint physical values:

```text
speed_integral_t = max(0, velocity_m_s - speed_limit_m_s) * dt_s
episodic limit = 0 m
```

### Constraint 2: dense normalized deadline-deficit integral

Let the track have piecewise-constant speed limits `v_i` on position intervals
`[x_i, x_(i+1))`, with the final interval ending at route length `L`. Define the
optimistic speed-limit-respecting remaining travel time from position `x`:

```text
T_min(x) = sum_i remaining_length_i(x) / v_i
```

This is an optimistic lower bound: it knows the fixed route speed-limit profile but
ignores acceleration, braking, comfort, and energy. It therefore accommodates varying
speed limits without imposing a linear-position schedule or an unrealistic early
target trajectory.

After each transition, define physical deadline slack and deficit:

```text
slack_s   = 140 s - elapsed_time_s - T_min(position_m)
deficit_s = max(0, -slack_s)

deadline_deficit_integral_t =
    dt_s * deficit_s / 140 s
```

The step cost has units of seconds: it is physical time multiplied by a dimensionless
fractional deadline deficit. Its undiscounted episode sum is the area under normalized
negative deadline slack. Division by the task's own 140 s deadline is fixed
analytically, not tuned. The episodic cost limit is exactly `0 s`.

All Development and Validation tracks have optimistic start-state remaining times below
140 s (range 51.785714--117.000000 s), so the constraint is not impossible at reset.
The signal becomes positive as soon as the current state falls outside this optimistic
deadline envelope. A stopped agent necessarily accumulates positive cost well before
the 180 s truncation. An incomplete episode and a completion after 140 s also incur
positive cost without a binary terminal failure term. No terminal failure cost is used
in V2.

Because the cost is nonnegative and its limit is zero, discounting changes learning
credit and magnitude but not whether a zero-cost trajectory exists. The internal proxy
is conservative: a trajectory that temporarily leaves the optimistic envelope and later
recovers retains positive episode cost. It is not claimed to be equivalent to the hard
deadline. External `EpisodeMetrics` and `is_feasible` remain the ground truth.

The cost order is frozen as:

```text
0: speed_integral_m                 limit 0.0 m
1: deadline_deficit_integral_s      limit 0.0 s
```

Separate critics and separate PID multipliers are retained.

## Frozen implementation and training protocol

- Environment: `StochasticTrack-v1`, `max_episode_steps = 1800`.
- Native integration and action period: 0.1 s; `action_repeat = 1`.
- Algorithm: FSRL SACLag, unchanged from V1.
- Training seeds: 11, 29, 47.
- Development tracks: 2000--2008.
- Validation tracks: 3000--3008.
- Historical 1000--1008 tracks are comparison-only and are not used for V2 tuning.
- Sealed paper tracks 4000--4017 must not be generated, evaluated, or inspected.
- Each seed receives exactly 300,000 native simulator transitions.
- Checkpoints target 50k, 100k, 150k, 200k, 250k, and 300k simulator transitions,
  using the same completed-episode overshoot convention as V1 except that the final
  episode is truncated exactly at 300k.
- Deterministic external evaluation uses unchanged `TaskSpecification`,
  `EpisodeMetrics`, `is_feasible`, and RSR.

The unchanged SACLag settings are: actor LR `5e-4`, critic LR `1e-3`, two 128-unit
hidden layers, automatic entropy tuning with effective initial alpha 1 and alpha LR
`3e-4`, `tau=0.05`, `n_step=2`, `gamma=0.99`, replay capacity 100,000, batch size 256,
update-per-step 0.1, PID `(0.05, 0.0005, 0.1)`, Lagrange rescaling enabled, and both
initial multipliers zero.

One 2,048-transition Development-only implementation smoke run is allowed to verify
finite values, exact accounting, ordering, persistence, and numerical behavior. It may
not select among alternate cost formulas or tune hyperparameters. Any protocol-changing
bug requires documentation and fresh affected main runs.

## Logging and external output

Every training episode records native steps, objective return, physical energy, both
undiscounted cost returns, deadline deficit at episode end, both multipliers, alpha,
actor loss, and all critic losses. Every checkpoint stores model and PID state.

Every external episode stores training seed, checkpoint, track seed, completion,
feasibility, final position, travel time, energy, violation count, maximum speed excess,
and integrated speed violation through the shared evaluator.

Final episode behavior is classified exclusively and reproducibly in this order:

1. `fully_feasible`: externally feasible;
2. `completed_with_speed_violation`: completed but not speed compliant;
3. `completed_too_slowly`: completed, speed compliant, but after 140 s;
4. `standstill`: incomplete with final position at most 1 m;
5. `partial_progress_or_crawling`: all other incomplete episodes.

The category is analysis-only and never enters training.

## Preregistered acceptance criteria

V2 materially improves on V1 only if, at 300k on Validation:

- at least two of three training seeds have both completion greater than zero and RSR
  greater than zero;
- pooled RSR is at least 6/27 (a gain of at least 6/27 over V1's 0/27); and
- standstill is at most 9/27 pooled episodes.

A credible V2 baseline additionally requires pooled speed compliance of at least 24/27
and no late RSR collapse larger than 2/9 within any training seed.

The previously frozen strong-baseline bar is retained separately: each seed needs at
least 8/9 RSR, completion, and speed compliance; final seed RSR range at most 1/9;
peak-to-final RSR loss at most 2/9; and feasible mean energy at most 1.25 times the
fast compliant reference on paired tracks. Energy is ranked only after feasibility.

Systematic optimizer instability is predefined as at least two seeds having a
non-finite required diagnostic, any absolute critic loss above `1e6`, or either
multiplier above `1000`.

## Decision gates

Apply gates in this order:

1. **E -- unstable:** the systematic optimizer-instability definition is met.
2. **A -- robust feasibility:** the material-improvement and credible-baseline criteria
   are met. Dense task credit solves standstill sufficiently for a credible constrained
   baseline; the stronger benchmark bar is reported independently.
3. **B -- interacting constraints:** at least two seeds complete, pooled completion is
   at least 6/27, standstill is at most 9/27, but pooled speed compliance is below 24/27.
4. **D -- persistent standstill:** more than half of final Validation episodes are
   classified as standstill.
5. **C -- movement without robust feasibility:** all remaining outcomes.

V1 frozen results and the frozen SB3-SAC result of 5/27 are comparisons only. V1 is not
retrained. The main causal comparison is V1 binary terminal task cost versus this one V2
dense physical deadline cost.

After analysis and reporting, stop. Do not implement requirement-conditioned RL, goal-
conditioned RL, HER, CRL, another algorithm, or another cost sweep automatically.
