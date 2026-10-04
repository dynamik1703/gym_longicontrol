# Requirement-Conditioned RL V1 protocol

Status: **preregistered before smoke testing or policy training on 2026-10-03**.

## Research question

Can one policy observe an absolute operational deadline and adapt its trajectory
and energy use while completing the route, respecting the strict speed limit,
and meeting that episode-specific deadline? Can it interpolate zero-shot to
unseen intermediate deadlines?

This is not an optimizer comparison. FSRL SACLag and every learning setting are
copied from frozen Constrained V2. The only conceptual change is that `T_max`
varies by episode and the policy observes it.

## Physics-only requirement selection

Requirement design uses Development tracks 2000--2008 only. The optimistic
speed-limit integral `T_min(start)` spans 55.686--117.000 s. The existing
privileged speed-compliant reference (0.5 m/s limit margin, 0.75 m/s² braking
model) needs at most 19.914 s beyond that optimistic bound; this captures
acceleration and braking omitted by `T_min`.

The deterministic rule is:

```text
tight margin = ceil(max Development reference gap / 10 s) * 10 s
training anchors = tight + [0, 20, 40] s
interpolation = tight + [10, 30] s
```

Therefore:

```text
seen training margins:       20, 40, 60 s
unseen interpolation margins: 30, 50 s
T_max = T_min(start) + margin
```

The largest Development deadline is 177 s, below the existing 180-s episode
horizon. Learned-policy outcomes cannot change these values. Tracks 1000--1008
and sealed tracks 4000--4017 play no role in selection.

## Observation and Markov state

The public eight-dimensional observation is unchanged. A benchmark wrapper
appends:

```text
[T_max / 180 s, elapsed_time / 180 s]
```

The policy observation is therefore ten-dimensional. It does not contain
`T_min`, slack, deficit, an oracle action, or an optimal trajectory.

Elapsed time is required because identical position/velocity/track observations
with equal `T_max` but different elapsed time have different remaining budgets
and can require different actions. The augmented state exposes both operands
needed to derive remaining time without exposing the engineered solution.

## Requirement sampling

Training uses a dedicated RNG per training seed. Each shuffled block contains
20, 40, and 60 s exactly once; blocks are independently permuted. Exposure is
balanced to within one completed episode and independent of the environment's
track RNG. Every episode logs margin, `T_min(start)`, and absolute `T_max`.

FSRL's `FastCollector` performs two internal resets after a collected episode.
The runner therefore draws exactly one margin in its outer completed-episode
loop and explicitly passes that same margin to the collector's pre-collection
and internal resets. Analysis rejects a run if completed-episode exposure counts
differ by more than one.

## Frozen learning formulation

```text
objective = -signed_step_energy_kwh / 0.25
speed cost = max(0, velocity - speed_limit) * dt
slack = T_max - elapsed_time - T_min(position)
deficit = max(0, -slack)
deadline cost = dt * deficit / T_max
cost limits = [0, 0]
```

Algorithm: FSRL SACLag at commit
`e056fc9498d5d037869533da7cf976acf462f918`, Tianshou 0.5.1. Actor and critic
layers are `[128, 128]`; actor LR 0.0005, critic LR 0.001, gamma 0.99, n-step 2,
tau 0.05, replay 100,000, batch 256, update ratio 0.1, automatic entropy tuning,
PID `(0.05, 0.0005, 0.1)`, zero initial multipliers, and native 10-Hz actions.

Training seeds are 11, 29, and 47. Each receives exactly 300,000 native
simulator transitions, with evaluations near 50k increments. No budget extension
is permitted if curves are still improving.

## Splits and evaluation matrix

- Development: 2000--2008.
- Validation: 3000--3008.
- Historical 1000--1008: unused.
- Final paper test 4000--4017: sealed and untouched.

At each checkpoint, every policy is evaluated deterministically on all nine
Validation tracks under all five margins: 45 episodes per policy and 135 pooled
episodes. Results are reported separately for seen 20/40/60 and unseen 30/50 s.
Every absolute `T_max` is stored. Physics-only feasibility is checked before
policy evaluation.

At 300k the same policies receive a separate absolute `T_max = 140 s` evaluation
on Validation for comparison with frozen Constrained V2 (21/27). V2 is not
retrained.

## External metrics

Episode feasibility remains:

```text
completed
and travel_time_s <= episode T_max
and max_speed_violation_m_s <= 0
```

`TaskSpecification`, `EpisodeMetrics`, public environments, dynamics, historical
rewards, termination, and external zero-tolerance speed semantics are unchanged.
Energy is compared only among feasible paired episodes.

## Controllability metrics

For each training-seed/track group, all ordered tight-to-loose pairs are used.

- Travel-time monotonic pair: `time_loose >= time_tight - 0.1 s`. The tolerance
  equals one simulator interval.
- Energy monotonic pair, only if both episodes are feasible:
  `energy_loose <= energy_tight + 0.001 kWh`; 1 Wh is an operational rather than
  floating-point tolerance.
- Spearman correlations use average ranks within each track for deadline versus
  achieved time and energy.
- Median paired time and energy differences are reported.

A track/seed group is requirement-sensitive if at least one holds across the
five primary requirements:

```text
travel-time range >= 1.0 s (ten simulator steps)
feasible-energy range >= 0.005 kWh (5 Wh)
mean absolute tightest-vs-loosest action-profile difference >= 0.05
```

The action profile is deterministically interpolated onto 101 longitudinal
positions. These thresholds are frozen before training and prevent an
always-fast, numerically jittering policy from being called controllable.

Interpolation directional consistency checks each 30/50-s result against its
two neighboring seen anchors with the same time tolerance. Energy direction is
reported where all involved episodes are feasible.

## Success criteria

All are fixed before training:

1. pooled primary RSR at least 70%;
2. every one of the five requirements at least 50% RSR;
3. every training seed at least 50% pooled RSR;
4. travel-time pairwise monotonicity at least 70%;
5. feasible-energy pairwise monotonicity at least 60%;
6. at least two thirds of track/seed groups requirement-sensitive;
7. unseen interpolation RSR no more than 15 percentage points below seen RSR;
8. interpolation travel-direction consistency at least 65%.

Requirement satisfaction and sensitivity are co-primary: high RSR with
insensitive trajectories fails the conditioned-control claim.

## Decision gates

- **F**: all criteria pass, including interpolation; freeze Requirement-
  Conditioned V1 as strong controllable zero-shot interpolation.
- **A**: core RSR and controllability pass, but interpolation is useful without
  meeting the stronger F criteria.
- **B**: RSR criteria pass but sensitivity fails; requirements are ignored.
- **C**: seen criteria pass but interpolation RSR/direction fails.
- **D**: the tight 20-s level is below 50% while looser levels work.
- **E**: pooled RSR is below 70% with broad multi-requirement degradation.

No requirement-blind control is included: it would double the main training
scope and is not needed to answer V1's within-policy controllability question.
No HER, CRL, relabeling, reward shaping, speed margin, anticipatory braking, or
new algorithm is permitted. Stop after reporting this study.

## Sampling implementation amendment

The first execution on 2026-10-03 exposed a collector-integration defect before
the study was finalized: internal collector resets advanced the requirement
sampler even though those reset episodes were never driven. Completed-episode
counts were consequently not balanced. That execution is invalidated and is
not used in `results.json`, plots, trajectory selection, or conclusions.

The correction above pins the already-preregistered outer-loop draw across all
collector resets. No deadline, split, metric, threshold, optimizer setting, or
training budget was changed, and the corrected run started from new policies.
This amendment documents implementation fidelity; it is not a result-driven
protocol change.
