# Protocol

## Construction and static gates

1. Freeze the first 18 acceptable tasks from numeric namespace 9200000 as
   `COLLECTOR_DIAGNOSTIC_V1`.
2. Freeze the first 18 acceptable tasks from namespace 9300000 as
   `COLLECTOR_VALIDATION_V1`, enforcing uniqueness across both new sets.
3. Do not construct Paper content. Paper exclusion is numeric only.
4. For each nominal seed, draw candidates from exactly
   `default_rng(SeedSequence([seed, 0x43414E4F]))`.
5. Reject a candidate whose numeric seed or route/full-task hash is reserved.
   Do not reject natural repetitions among accepted training tasks.
6. Freeze the first 865 accepted tasks and their ordered SHA-256 for each seed.
7. Recheck both evaluation sets against all three complete streams. Any
   collision stops preparation before training.

## Initialization and execution gates

Within a seed pair, hashes of actor, all critics and targets, optimizer states,
alpha, PID, and empty replay must be identical. Different seeds must initialize
differently. Collector ordering, deterministic evaluation non-interference,
atomic recovery round-trip, and the measured 18-hour/15-GB resource gate must
all pass.

The immutable serial order is E11, T11, E29, T29, E47, T47. Recovery checkpoints
are atomically replaced every 10,000 transitions; scientific checkpoints are
retained at 50k increments. Attempt IDs are immutable. An interrupted run may
resume only from its own verified complete checkpoint. If that is impossible,
the state is `FRESH_RESTART_REQUIRED` and execution stops.

Diagnostic evaluations occur at 50k, 100k, 150k, 200k, 250k, and 300k. They
are deterministic actor-only evaluations and cannot affect learner RNG. The
300k policy is always final; there is no checkpoint selection.

Validation remains unopened until every final checkpoint exists and its hash
verifies. It then runs once for all six policies (108 episodes). No training,
tuning, or repeat follows Validation.

## Frozen descriptive interpretation

`EPISODIC_ROBUST_ADVANTAGE` requires pooled episodic RSR to exceed transitional
by at least 6/54, episodic to be no worse in every seed pair, and episodic to
lead by at least 2/18 in at least two seeds. The transitional criterion is
symmetric.

`NO_MATERIAL_DIFFERENCE` requires absolute pooled difference below 6/54 and no
strong contradictory pattern, defined here as at least one seed effect ≥ +2 and
another ≤ -2. Every remaining pattern is `SEED_HETEROGENEOUS`.

Robust advantage selects its collector. No material difference selects
TRANSITIONAL because its standard online cadence avoids accumulated episode-end
update backlogs and is simpler to account for reproducibly. Seed heterogeneity
freezes no collector and recommends only a future 2×2 mechanism study separating
within-episode actor updates from PID timing; that study is not authorized here.
