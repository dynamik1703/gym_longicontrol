# Goal-conditioned LongiControl design

## Research boundary

This stage asks whether hindsight relabeling changes learning when the physical
task, goal information, sparse reward, optimizer and data budget are otherwise
identical. It prepares two future runs but performs no policy training:

1. goal-conditioned SB3 SAC with zero virtual replay goals;
2. the same learner with position-only future HER.

The canonical evaluation task remains route completion by 140 seconds with
exactly zero measured speed excess. `TaskSpecification`, `EpisodeMetrics` and
`is_feasible` remain authoritative. Energy is recorded externally among
feasible episodes and never enters the training reward.

## Audit of the physical task

The public v1 observation contains velocity, acceleration, the current and two
visible future limits, their distances and `energy_factor`. It omits route
position, elapsed time and cumulative speed-violation history. The base
environment terminates when position reaches 1,000 m. Gymnasium's outer
`TimeLimit` truncates after 1,800 native 0.1-second steps. `info` provides
position, elapsed time and cumulative `max_speed_violation_m_s` on every step.

Goal success therefore needs four stored transition quantities:

- current route position;
- previous route position, to identify first arrival;
- elapsed time;
- maximum speed excess since reset, not current excess.

Adding those quantities does not make the stochastic-track problem fully
observable. Speed limits beyond the existing 150 m sensor horizon remain
hidden, as do the full generated track and future transitions. No track ID,
full-track description, oracle trajectory, optimal action or deadline-slack
helper is exposed.

## Exact goal representation

All goal vectors have four normalized entries in `[0, 1]`:

| Index | `achieved_goal` | `desired_goal` | Scale |
|---:|---|---|---:|
| 0 | current position | target position | 1,000 m |
| 1 | previous position | target position, duplicated for crossing | 1,000 m |
| 2 | elapsed time | deadline | 180 s |
| 3 | cumulative maximum speed excess | allowed maximum excess | 37 m/s |

The duplicated desired position makes first arrival a pure transition test:

```text
previous_position < target_position <= current_position
and elapsed_time <= deadline
and cumulative_max_speed_excess <= tolerance
```

There is no hidden epsilon. The canonical desired goal is
`[1, 1, 140/180, 0]`. On valid environment transitions, canonical success is
checked against `is_feasible` whenever an episode ends. Every real rollout in
both conditions uses this one canonical goal.

The sparse transition reward is `1.0` exactly on a valid first-arrival
transition and `0.0` otherwise. The first-arrival task terminates at that
transition. A later state at or beyond an already reached target is not another
success. The original route naturally terminates on its first crossing, so the
canonical real-rollout reward remains a terminal success bit.

## What hindsight changes

HER introduces a broader training-only family of counterfactual tasks: a target
may be any strictly future position on the same original route that had not
already been reached at the sampled transition. This is additional task
engineering and not a relaxation of canonical evaluation.

Only both copies of target position are relabeled. The 140-second deadline and
zero speed tolerance remain those from the original desired goal. Elapsed time
and cumulative maximum speed excess come from the stored achieved goal and are
never relabeled or reset. Consequently, slowing down cannot erase an earlier
overspeed event, and a prefix reached only after 140 seconds cannot become a
success. A stationary prefix with no strictly future position falls back to the
canonical goal rather than constructing a post-terminal virtual transition.

The counterfactual episode would have ended at first arrival. Its virtual replay
sample therefore uses:

```text
virtual_done = original_done OR relabeled_first_arrival_success
```

Original route termination and the finite 180-second horizon remain terminal.
Later transitions are never rewarded repeatedly for the same arrival because
success requires a strict previous-position crossing.

## Installed SB3 HER audit

Stable-Baselines3 2.9.0 `HerReplayBuffer` was inspected locally. It:

- samples only transitions from episodes whose end has been recorded;
- uses inclusive future sampling, so the current transition may provide its
  next achieved goal;
- replaces the complete desired-goal vector with a future achieved-goal vector;
- calls `env.env_method("compute_reward", next_achieved_goal, new_goal, infos)`;
- retains the original `done` for virtual samples;
- defaults to masking `TimeLimit.truncated` as a nonterminal timeout.

Those stock semantics are not sufficient here. Full-vector replacement would
change the deadline and tolerance, unchanged `done` would bootstrap through a
counterfactual first-arrival terminal, and the default timeout handling would
treat the finite failed task horizon as continuing. Calling the live environment
also makes it easier to accidentally depend on obsolete state.

`GoalReplayBuffer` is the minimal adapter. It retains SB3's episode
bookkeeping and real samples but selects only admissible future positions,
preserves the original deadline/tolerance, computes reward directly from stored
arrays, recomputes virtual terminal masks and freezes
`handle_timeout_termination=False`. It neither changes SAC nor introduces a new
learning algorithm.

Both arms use this episode bookkeeping. Otherwise ordinary Dict replay could
sample the unfinished current episode while HER could not, adding a second
difference. The no-HER arm sets `n_sampled_goal=0` and returns only real
samples; the HER arm sets it to 4. Sample eligibility is therefore identical.

HER cannot sample until one completed episode exists. The shared warmup is
therefore 1,800 interactions, sufficient for a maximum-length episode. This
replaces the historical SAC warmup of 100 for both arms and remains inside each
300,000-transition budget.

## Difference from Binary Reward V1

Binary V1 exposes the unchanged eight-feature observation and emits its `1`
only when the original episode outcome is known. This study adds explicit
position, clock, cumulative violation history and desired requirements through
the Goal API. Its shared reward is still exactly sparse 0/1, but is defined on
the first-arrival transition so a relabeled intermediate target has a coherent
counterfactual terminal. No progress, distance, energy, speed, time, completion
or failure shaping term is added.

The no-HER arm is therefore not historical Binary V1. It controls for all new
goal representation, MultiInput policy, first-arrival reward, completed-episode
sample eligibility and shared 1,800-step warmup. Comparing it with the HER arm
isolates replay relabeling rather than those additional task inputs.

## Remaining limitations

Position-only hindsight changes the training task distribution even though
canonical evaluation is fixed. Some future targets reached late or after a
speed violation correctly remain zero-reward virtual tasks. This may yield
fewer positive relabels than conventional unconstrained HER; it is intentional,
because resetting history or relaxing requirements would answer a different
question.

The study will provide three policy-training replicates per condition, not 54
independent policies. Track-level outcomes are paired descriptively, while seed
results remain visible. No claim about contrastive RL, optimality or energy
efficiency is part of this stage.
