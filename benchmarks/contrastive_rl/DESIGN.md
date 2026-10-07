# LongiControl mapping decision

## Fixed external semantics

The benchmark evaluator remains authoritative and unchanged. Success is first
arrival across 1,000 m at absolute episode time no later than 140 s with a
cumulative maximum speed excess no greater than 0 m/s. Energy is descriptive
and compared only among feasible episodes.

The policy state considered here is the original eight sensor-limited values
plus current position, previous position, absolute elapsed time, and cumulative
maximum speed violation. This does not expose track identity, a full future
profile, optimal actions, deadline slack, or `T_min`, and it does not make the
problem fully observable.

An achieved outcome is the exact physical tuple

```text
(current_position_m,
 previous_position_m,
 absolute_elapsed_time_s,
 cumulative_max_speed_violation_m_s)
```

Normalization is linear and never clipped. Exact set membership is always
computed in physical units. A failed trajectory remains a valid positive for
its exact achieved future outcome; it is never relabeled as timely or safe.

## Option 1 — point-goal approximation (rejected)

Command `(1000 m, 140 s, 0 m/s)` and train ordinary future-state CRL. This
would mean reaching a time coordinate equal to 140, rather than arriving at any
time `<=140`. It also lacks an exact first-crossing predicate unless previous
position is retained. Dense distance to the point introduces preferences
inside and outside the feasible set that the task does not specify. This option
is simple but semantically changes the benchmark, so it is rejected.

## Option 2 — exact-outcome critic plus canonical goal-set query (selected direction, not ready)

Train the source-faithful critic on exact achieved future outcomes from the same
episode. The canonical command is then the set of all outcomes satisfying the
three inequalities, not a fabricated successful example. This preserves late,
unsafe, and non-arrival outcomes honestly and keeps absolute episode time.

The unresolved part is the actor query. If a contrastive critic estimates a
density ratio for point outcomes under a reference goal distribution, obtaining
mass over a set generally requires integration with the corresponding reference
measure. An arbitrary `logsumexp` over hand-sampled feasible points is not a
calibrated success probability and changes when the goal sampler changes.
Moreover, early replay may contain no safe arrival outcomes; fabricating them is
forbidden, while discarding failed trajectories would bias the task.

Therefore this direction is selected for further derivation, but the adapter is
**NOT READY** for real data collection or learning. Before execution, one must
preregister and justify:

1. the reference measure over exact outcomes and its support;
2. a set-integral/aggregation estimator and sampling weights;
3. the actor-goal distribution, including cold start with no observed feasible
   outcomes;
4. whether the resulting surrogate still answers canonical first-arrival RSR;
5. duplicate/equivalent-outcome treatment applied identically at both depths.

## Occupancy, terminals, and time

The upstream objective associates actions with discounted future occupancy.
LongiControl terminates on route completion, so there is no reward for remaining
near the endpoint and no absorbing endpoint is fabricated. The real terminal
first-arrival outcome may be a future positive for an earlier source, but it is
never repeated or used as a post-terminal source. A timeout is a final failed
outcome, not a canonical positive. A source transition can pair only with a
strictly later state in the same uninterrupted physical episode.
Absolute elapsed time is stored; subtrajectory sampling never restarts the
deadline clock. Cumulative violation history is monotonically retained, so a
later slowdown cannot erase prior unsafe behavior and a safe prefix is not
retroactively invalidated by a future violation.

In-batch equal outcomes are false negatives under unmodified diagonal InfoNCE:
with `k` indistinguishable columns the classification term cannot distinguish
their row identities (all-identical batch loss is `log(batch_size)`). The tests
make this behavior visible. No track ID or masking is added merely to make the
classification easier.
