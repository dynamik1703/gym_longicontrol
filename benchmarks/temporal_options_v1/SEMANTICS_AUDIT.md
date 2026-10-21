# Semantics Audit

The frozen planner computes distance-triggered phases relative to each
plan's current origin. The frozen controller replans at one-second
boundaries. In deterministic constant-limit fixtures, repeated selection
therefore postponed the 40%, 50% and 82% thresholds when less than that
fraction of the newly anchored 100 m horizon was executed per block.

This is a verified execution behavior, not by itself an implementation
bug or evidence of episode-level energy loss. Velocity-triggered pulse
behavior still entered coasting after reaching its speed threshold.

| Profile | Restart phases | Persistent phases |
| --- | --- | --- |
| `accel_coast_40_0.5` | `{'traction': 90}` | `{'coast': 46, 'traction': 44}` |
| `accel_coast_60_0.5` | `{'traction': 90}` | `{'coast': 30, 'traction': 33}` |
| `accel_coast_brake_50_-0.5` | `{'traction': 90}` | `{'brake': 15, 'coast': 30, 'traction': 30}` |
| `accel_coast_brake_50_-1.0` | `{'traction': 90}` | `{'brake': 17, 'coast': 30, 'traction': 30}` |
| `coast_partial_35` | `{'coast': 90}` | `{'coast': 26, 'partial_traction': 64}` |
| `coast_partial_50` | `{'coast': 90}` | `{'coast': 26, 'partial_traction': 52}` |

## Native action 0

At 15 m/s, action 0 produced acceleration -0.211721726 m/s² and model power 2.172829603 kW. It is the production vehicle's
coasting action, not an added controller or a guaranteed exactly-zero
numerical power value.

## Boundaries and remaining uncertainty

Speed-limit clipping, known-transition braking, unknown-preview checks
and deadline recoverability can legitimately change or interrupt phases.
The final-route option uses its actual shortened length. No claim that
all coasting was historically suppressed is supported.

Fixture native steps: 1103 (below 30,000).
