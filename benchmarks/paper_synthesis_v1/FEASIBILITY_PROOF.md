# Discrete Physical-Feasibility Proof

## Production semantics independently checked

The proof is tied to the source frozen at `9802f926…`:

- `domain/dynamics.py` advances position by
  `v·dt + 0.5·a·dt²`, clips acceleration so the next velocity stays within the
  vehicle velocity interval, and uses `dt = 0.1 s`.
- `domain/track.py` implements right-continuous limits with
  `searchsorted(..., side="right") - 1`; a point exactly at 600 m therefore
  uses the new segment's limit.
- `envs/longicontrol.py` integrates first, senses the limit at the new position,
  updates the metric at that right endpoint, and terminates inclusively when
  `position_m >= 1000`.
- `domain/metrics.py` records positive endpoint excess with no tolerance hidden
  in the metric.
- `domain/task.py` requires completion, `travel_time_s <= max_time_s`, and
  `max_speed_violation_m_s <= max_speed_violation_m_s`, all inclusively and
  without an epsilon.

The serialized task also fixes the initial position and velocity at zero,
vehicle velocity range `[0, 37] m/s`, acceleration range `[-3, 3] m/s²`, native
timestep 0.1 s, zero speed tolerance, and 140 s deadline.

## Lemma

For a segment of length `L` and limit `u`, classify a native step as a boundary
step for that segment if its traversed position interval touches the segment's
entry or exit. Position is monotone because both endpoint velocities are
nonnegative under the clipped constant-acceleration update. Thus there are at
most two boundary steps for any segment (the same step may serve both roles).

The exact source expression is bounded by

```text
D = v_max·dt + 0.5·a_max·dt² + 1e-9 m
  = 3.715000001... m.
```

The 1 nm margin is deliberately larger than the binary64 roundoff of the few
operations and the position addition at this scale. Boundary steps can therefore
cover at most `2D` of a segment.

Every remaining step lies wholly in one segment. Its start velocity is either
the zero initial velocity or the preceding checked endpoint velocity; its end
velocity is checked against the same right-continuous limit. Constant
acceleration makes its exact displacement the endpoint-speed average times
`dt`, hence at most `u·dt`. The executable analysis weakens this further to an
effective `u + 1e-8 m/s` to cover numerical roundoff. Interior-step sets for
different segments are disjoint, so the lower bounds on their durations add:

```text
T >= Σ_i max(0, L_i - 2D) / (u_i + 1e-8 m/s).
```

This is a lower bound on total episode time, not an MPC estimate. It deliberately
discards acceleration-from-rest time, every boundary-step duration, braking
requirements, and any time after overshooting the finish, making it optimistic
for feasibility.

## Track 10200029

The existing canonical serialization hashes independently to:

- route: `75339f9ec7e50d5514afdbdd4ed4d8e720beb4229ba525596c9707bc6d1f8c60`
- full task: `6f0d03873d9039e863cb40cecfb024aac0341f95e878272a86b1f1805801b883`

It contains 600 m at the stored binary64 value of 30 km/h and 400 m at
20 km/h. The continuous speed-limit-only value is 144 s, but that value alone
is not used as the proof. Applying the conservative discrete lemma gives about
141.771 s. Because this remains strictly above the inclusive 140 s deadline,
no legal zero-excess trajectory can satisfy the task. The classification is
therefore `PROVEN_INFEASIBLE`.

No controller was run for this analysis.

## Scope

The analyzer returns `UNRESOLVED` for missing serializations or unsupported
semantics. A failed episode cannot change that result. Positive classification
requires a completed, exact task-linked original-environment record within both
bounds. The proof is not a global safety theorem, energy bound, or claim that
all generated tasks are infeasible.
