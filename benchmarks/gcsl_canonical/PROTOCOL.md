# Frozen execution protocol

## Training matrix and schedule

Policies G11, G29, and G47 each receive exactly 300,000 native simulator
transitions. Their task/reset streams are the exact canonical C11/C29/C47
streams generated from `SeedSequence([seed, 0x43414E4F])`. Maximum reachable
exposure is recomputed as `floor(300000/347)+1 = 865` tasks per policy.
These are fixed canonical training streams with disclosed historical-content
overlap; they are not claimed to be globally route-clean.

Replay prefill is 10,000. The first supervised update is at transition 10,040
and the last at 300,000, with one batch-256 update every 40 transitions: 7,250
updates and 1,856,000 hindsight tuples per policy. There is no update-count
sweep.

## Split scopes

`CURRENT_GCSL_EVALUATION_RESERVED` contains `GCSL_DIAGNOSTIC_V1` and
`GCSL_VALIDATION_V1`. Every route and full task in both is excluded from every
reachable GCSL training exposure. The sets are internally unique and mutually
route/full-task disjoint.

`HISTORICALLY_OPENED_CONTENT` includes legitimate prior exploratory,
Development, Validation, diagnostic, collector, and Same-Information content.
It is excluded from new GCSL evaluation selection. The fixed canonical
training streams nevertheless contain six disclosed route-and-full-task
matches with this historical scope. Those matches do not fail the current
study gate because GCSL is never evaluated on the matched historical sets.

Paper seeds 4000--4017 are numeric exclusions only. Their content remains
`SEALED_UNINSPECTED`: it is never instantiated, fingerprinted, hashed, or
evaluated.

Diagnostic runs at the already scheduled 50k, 100k, 150k, 200k, 250k, and
300k scientific checkpoints during future authorized training. Validation is
frozen unopened and may run only after all GCSL training completes under a
separate authorization. Progress reporting summarizes those existing events;
it creates no additional checkpoints or evaluation episodes.

## Diagnostics

Existing sampled batches report source coverage, physical future lag, terminal
fraction, goal composition and uniqueness, canonical target support, progress,
goal-to-action ambiguity, NLL, action error, policy standard deviation,
gradient/parameter norms, nonfinite counts, and fixed-state goal queries.
Private diagnostic RNG state cannot advance policy, replay, task-stream, Python,
NumPy, or Torch training RNGs.

## Recovery and attempts

Technical checkpoints occur every 10k transitions and overwrite one recovery
slot. Scientific checkpoints occur every 50k and are retained atomically. An
immutable Attempt ID binds policy, optimizer, replay and eligibility, episode
boundaries, environment, current observation/outcome/task, stream cursor,
policy/replay/Python/NumPy/Torch RNGs, counters, diagnostics, hashes, and runtime
versions.

Same-Attempt exact resume is permitted. Duplicate execution and silent fresh
restart are blocked. Tests cover mid-episode, pending update, termination/reset,
and replay-overwrite recovery. Detached launchd configuration writes heartbeat
and status files; reading those files adds no scientific interaction.

## Contextual Constrained reference

Frozen canonical TRANSITIONAL policies C11/C29/C47 are hash-verified and will
only be evaluated on `GCSL_VALIDATION_V1` after GCSL training and explicit
Validation authorization. This is contextual, not a perfectly controlled
comparison: GCSL uses 12D state, Constrained uses 8D state, objectives differ,
and optimizer budgets are 7,250 versus 30,000 updates.

## Authorization

Preparation ends with readiness true but authorization false. Main training,
Validation, Constrained evaluation, and Paper interaction remain prohibited.
No preparation command changes these flags.
