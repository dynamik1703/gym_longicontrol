# Goal/HER execution incident: 2026-10-06

The first main-study attempt stopped during the first terminal transition of
`sac-no-her`, seed 11, before any checkpoint, Development evaluation or
Validation evaluation. The preserved ignored run root is
`runs/goal-conditioned-her-v1`.

Its manifest records `INTERRUPTED`, zero completed policy transitions, zero
Development/Validation transitions and sealed Validation/Paper splits. Because
the failure preceded the first atomic checkpoint, the exact number of physical
transitions already executed was not durably recorded. No result from this
attempt is part of the main-study budget or analysis.

The failure was in an execution-only diagnostic assertion. `achieved_goal`
stores route position normalized and clipped to `[0, 1]`, while
`info["position_m"]` retains the physical overshoot beyond 1,000 m on the
terminal integration step. The assertion decoded the clipped value as exactly
1,000 m and incorrectly required equality with the overshooting physical value.
Reward, success, goal encoding, replay sampling and terminal masks were not
changed.

The correction compares the stored normalized terminal position with the same
clipped normalization used by the frozen encoder. It also persists exact model
timestep/update counters if a later policy is interrupted. Regression tests
cover a valid terminal overshoot and reject an actually inconsistent terminal
observation.

After the interruption was reported, the user explicitly authorized starting
the six main runs again. The original manifest and lock remain untouched. The
runner requires `--authorized-restart-from` and records hashes of both preserved
files in the new attempt's pre-training provenance; it refuses to treat an
uninterrupted or Validation-opened directory as a restart source.

Preserved attempt hashes at authorization time:

- `manifest.json`: `55a88a291d82be5ea0ba329d0ab4d26456e899401c52af58eb561776a261eb30`
- `ACTIVE.lock`: `b90cc0672a84f58751a5a4d1c490d20b0fcc12c3b2f74854bc94d70ca5f9f6f9`
