# Stateful Option Contract

An option is a feedback law plus persistent physical execution state, never a
cached actuator sequence. An instance stores its unique ID, profile/family,
start position/time, original waypoint and length, actual progress and phase,
native duration, last executed action, status and exact termination reason.

At `START(profile)`, the anchor is the current causally propagated position and
the waypoint is `min(anchor + 100 m, 1000 m)`. At every continuation step,
progress is `(actual_position - anchor) / original_length`, clipped to `[0,1]`.
Observation updates, new visible signs, planning and `CONTINUE` do not change the
anchor, waypoint or length. A same-profile `START` creates a new ID and anchor.

Every 0.1 s the executor ingests the ordinary observation through the inherited
fallback, recomputes the profile action from current physical state and persistent
progress, applies inherited speed-cap generation, then applies unchanged endpoint,
known-transition braking, unknown-preview and deadline checks. An override executes
the inherited fallback immediately, records the reason, discards the option and
holds fallback until the next one-second boundary.

Normal termination is waypoint completion, episode end or the registered
400-native-step lifetime. A waypoint reached between strategic boundaries also
holds fallback until the next boundary. Strategic switching, safety interruption,
lifetime timeout, episode termination and infrastructure failure remain distinct.

Action 0 retains its production meaning: the BMW-i3 model's coasting/zero-power
action curve. No pulse hysteresis, alternate coast model or new acceleration
controller is introduced. The Shield remains empirical protection, not a global
safety theorem.

