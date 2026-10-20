# Baseline and Treatment Comparison

## Three-arm experiment

| Arm | Strategy authority | Safety/viability layer | Learning | Online information |
| --- | --- | --- | --- | --- |
| Control 1 | Fixed 60 km/h causal reference | Its frozen endpoint/braking logic | None | Ordinary 8D observation + frozen causal memory |
| Control 2 | Deterministic time-aware target speed | Identical Hybrid shield | None | Same causal information as treatment |
| Treatment | SAC-learned target speed | Identical Hybrid shield | Three seeds | Same causal information as Control 2 |

The full-route Oracle is reported only as a privileged contextual energy
reference. It is neither information-matched nor claimed globally optimal.

## Control 1 — frozen causal controller

No source or setting changes are allowed. The cap is 60 km/h; velocity and
braking-position margins remain `1e-6 m/s` and `0.5 m`. Disabled Hybrid
integration must continue to reproduce its action, state, completion, deadline,
speed and energy traces exactly. A parity failure blocks training.

## Control 2 — non-learning energy-aware strategy

Every ten native steps, using only controller-propagated state, calculate:

```text
usable_time = 140 s - 2 s reserve - elapsed_time
required_average_speed = remaining_distance / usable_time
target = ceil(required_average_speed in km/h + 5 km/h)
target = clip(target, 20 km/h, 60 km/h)
```

Hold that target for the next ten steps and pass every proposal through the same
endpoint, braking, unknown-preview and deadline gates as Hybrid RL. Constants,
rounding and cadence are frozen in `baseline.py`; Validation cannot tune them.

This arm answers whether an apparent Hybrid improvement over fixed 60 km/h is
merely the result of an obvious deterministic time-aware speed adjustment.

## Fair evaluation

Control 1 and Control 2 each run once on every evaluation track. Each of the
three fixed 300k Hybrid policies runs once per track. All execution is
deterministic and uses the original environment, metrics and `is_feasible`.

Report RSR, completion, deadline and speed for every arm. Energy comparisons
use only paired feasible tracks:

- Hybrid versus frozen causal, per seed;
- Hybrid versus non-learning control, per seed;
- non-learning control versus frozen causal;
- optional Oracle contextual gap, clearly labeled privileged.

Track-level differences are the statistical unit. The three learned policies
are seed replications, not 108 independent track samples. Report each seed and
a pooled descriptive summary; do not hide a failed seed in the pool.

## Decision hierarchy

1. Any Hybrid seed below 35/36 RSR: `FEASIBILITY_LOSS`.
2. Feasibility passes but the frozen influence gate fails:
   `SHIELD_DOMINATES_POLICY` / ineffective learning.
3. Feasibility passes but mean paired energy reduction versus causal is below
   10%: `FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN`.
4. The 10% causal target passes but mean paired reduction versus Control 2 is
   below 3%: `ENERGY_GAIN_EXPLAINED_BY_NONLEARNING_CONTROL`.
5. Both energy targets pass for every seed with feasibility retained:
   `INCREMENTAL_HYBRID_ENERGY_GAIN`.

Exact observed values are reported regardless of labels. Constraint failures
cannot be offset by energy, and infeasible trajectory energy is never compared
as an optimization success.
