# Requirement-aware projected-goal CRL adaptation

## Design revision

Preparation commit `6eb0009` correctly rejected the point goal
`(1000 m, 140 s, 0 m/s)`: equality with time 140 is not the deadline
inequality, and an exact-outcome goal-set query would require a justified
reference measure and set integral. That rationale remains valid.

This revision deliberately chooses a different benchmark-specific abstraction:
project each actually observed physical outcome into progress and two exact
requirement predicates. It does not integrate logits over hand-sampled physical
goals and does not claim to reproduce the paper's unchanged state-goal
representation.

## Raw state and raw outcomes

Replay/provenance retains the original eight sensor-limited features plus
current position, previous position, absolute episode time, and cumulative
maximum speed violation. Raw values are never clipped, rounded, relabeled, or
restarted at a sampled subtrajectory. The policy state uses the same lossless
linear scaling already audited; it preserves continuous position/time rather
than replacing them with requirement bits. No track ID, full future profile,
optimal action, `T_min`, or deadline-slack helper is exposed, so the task
remains partially observable.

Every recorded future outcome is the physical tuple

```text
(current_position_m,
 previous_position_m,
 absolute_elapsed_time_s,
 cumulative_max_speed_violation_m_s)
```

## Fixed canonical goal projection

For the fixed task `(route=1000 m, deadline=140 s, tolerance=0 m/s)`:

```text
progress_goal = min(current_position_m / 1000, 1)
within_deadline = 1 if absolute_elapsed_time_s <= 140 else 0
compliant_so_far = 1 if cumulative_max_speed_violation_m_s <= 0 else 0
projected_goal = [progress_goal, within_deadline, compliant_so_far]
canonical_command = [1, 1, 1]
```

Predicates are evaluated from float64 raw physical values before conversion to
network inputs. Position saturation is a task abstraction only; overshoot stays
unchanged in raw replay. The API rejects any noncanonical task specification.
Energy remains evaluation-only.

## Canonical equivalence and domain

For valid original-environment transitions, the following assumptions hold:

1. the source state is before original route termination, so
   `previous_position < 1000`;
2. longitudinal position is monotone;
3. the first transition with `current_position >= 1000` terminates the physical
   episode;
4. no post-terminal state is introduced as a fresh source.

The adapter checks these conditions: previous position at/after the route is
rejected, backwards movement is rejected, and `terminated` must equal the raw
route-crossing predicate. Under this domain:

```text
projected_goal == [1,1,1]
iff
previous_position < 1000 <= current_position
and elapsed_time <= 140
and cumulative_max_speed_violation <= 0
```

The forward implication obtains first crossing from the validated transition
domain, not from the three projected coordinates alone. The reverse implication
follows directly from saturation and the two exact predicates. This equivalence
does not hold for arbitrary fabricated post-terminal rows, which are therefore
excluded. `TaskSpecification`, `EpisodeMetrics`, and `is_feasible` remain the
authoritative external evaluator.

## Goal distributions

- **Real collection:** every rollout is commanded `[1,1,1]`, even before replay
  contains a canonical success.
- **Critic update:** a source state/action is paired with a strict future raw
  outcome actually recorded in the same uninterrupted episode; that stored
  outcome is projected and used as its diagonal positive.
- **Actor update:** condition on the same sampled projected future outcomes,
  matching the pinned scaling implementation's future-goal actor update.
- **Evaluation:** command `[1,1,1]`.

Collection and actor-training goals therefore differ. An unobserved command is
allowed; an observed success is never fabricated. A late outcome remains, for
example, `[progress,0,1]`, and an unsafe outcome remains `[progress,1,0]`.
Failed trajectories and unsafe/late goals remain in training. A safe prefix is
safe at its own time, while cumulative history keeps all later post-violation
outcomes unsafe even after braking.

At cold start, replay may contain no canonical success. The collection actor is
still queried with `[1,1,1]`; its response at that unsupported goal relies on
function approximation. Positive associations to intermediate, late, or unsafe
goals supply conditional training tasks but do not relax the canonical
evaluation command and do not guarantee transfer to it.

There are no HER terminal masks, demonstrations, success seeding, handcrafted
controllers, curricula, absorbing endpoints, invented post-terminal
transitions, or repeated terminal positives. A real terminal arrival may be a
future for an earlier source exactly once.

## Statistical interpretation and duplicates

Projection maps multiple physical outcomes to one goal. Its reference
distribution is the projected-outcome distribution induced by real replay and
the pinned discounted-future sampler. Reference columns are samples from that
distribution, not declarations that an outcome is physically impossible.

No duplicate masking or multi-positive loss is introduced. Identical projected
columns remain false negatives under diagonal InfoNCE; an all-identical batch
has classification loss `log(batch_size)`. In the deliberately stationary
500-transition Development check, 500 unique raw outcomes projected to one
unique goal: raw pair collision rate 0, projected rate 1. This exposes a real
learning property, not a general-traffic estimate.

The score remains an uncalibrated density-ratio/association surrogate, not a
first-hit probability or safety certificate. Finite episodes, truncated future
sampling, partial observability, behavior-policy mixtures, duplicates, and
unsupported command queries limit stronger claims. Positive intermediate
associations do not guarantee transfer to `[1,1,1]`, and no theorem of improved
RSR is asserted. This method is explicitly a **requirement-aware projected-goal
CRL adaptation**, not task-free learning; the deadline and compliance bits are
task engineering.
