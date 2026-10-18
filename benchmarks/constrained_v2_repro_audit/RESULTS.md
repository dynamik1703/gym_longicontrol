# Audit results

## Observed outcomes

| Seed | Historical RSR | Current RSR | Historical completion/deadline/speed | Current completion/deadline/speed |
|---:|---:|---:|---|---|
| 11 | 9/9 | 0/9 | 9 / 9 / 9 | 2 / 0 / 9 |
| 29 | 5/9 | 4/9 | 9 / 9 / 5 | 5 / 4 / 9 |
| 47 | 7/9 | 6/9 | 9 / 9 / 7 | 9 / 6 / 9 |
| Pooled | 21/27 | 10/27 | 27 / 27 / 21 | 16 / 10 / 27 |

The qualitative shift—historical policies always finish but sometimes violate speed,
while current policies always respect speed but often fail the deadline—is real in the
retained artifacts. It does not by itself identify a PID defect.

## Primary classification

**C — RNG/trajectory mismatch (high confidence).**

The same nominal seed does not imply the same complete stochastic trajectory. The
environment seed construction differs, the first training track differs at episode 0
for every seed, and the historical collector discards an extra reset-generated track
after every episode. The current first 100 generated reset seeds match its persisted
checkpoint record exactly.

This plausibly explains seed 11 changing from 9/9 to 0/9 because off-policy deep RL
can follow a different optimization path after its first data differ. Confidence that
RNG mismatch alone quantitatively causes the entire 11-episode gap is medium, not
high, because runner cadence and runtime also differ and no controlled 300k ablation
was authorized.

## A–F causal assessment

| Category | Evidence | Counterevidence / limitation | Confidence | Plausibly explains seed 11? |
|---|---|---|---|---|
| A — scientific configuration mismatch | None found | Exact V2 hash, effective defaults, task/env/evaluator, and shared source match | High confidence rejected | No |
| B — end-to-end implementation mismatch | Whole-episode/bulk versus transition/interleaved collection; reset and PID timing; final update totals differ | Replay storage and n-step formula match conditionally | High, secondary/co-contributor | Yes |
| C — RNG/trajectory mismatch | Different RNG construction; first track differs; extra historical reset consumption | Historical raw tracks/runtime were not persisted | High, primary | Yes |
| D — runtime/dependency drift | 2 versus 4 Torch threads; OS patch differs; historical NumPy/Torch/Gymnasium unknown | Same CPU, Python, FSRL revision, Tianshou version; no specific runtime-only effect isolated | Medium, possible secondary | Possible |
| E — historical provenance insufficient | Exact run commit, three dependency versions, initial/RNG/replay state absent | Raw configs, all checkpoints/results, diagnostics, FSRL/Tianshou pins, and archived source remain | High as a limitation, not the leading mechanism | Prevents exact proof |
| F — no material mismatch | Conditional replay/n-step parity and shared scientific source | Direct deterministic RNG and schedule mismatches found | High confidence rejected | No |

## Integrity judgment

The historical 21/27 result is internally consistent with its three hashed final
checkpoints and raw Validation artifacts. The current 10/27 result is internally
consistent with its verified final checkpoints, one-shot Validation artifacts, exact
counters, and unchanged actors during evaluation. No protocol/integrity defect was
found that invalidates either observation.

The scientifically precise conclusion is:

> The current Matched Real-Only arm is a valid control within the MVE runner, but it
> is not an exact seed-paired end-to-end reproduction of historical Constrained V2.

## Earliest deterministic divergence

1. Network constructors match under the same current runtime.
2. The first generated training track differs.
3. The first reset observation consequently differs for the bounded audit seed.
4. Action, transition, replay, target, gradient, episode, and PID ladder levels were
   not run because the earlier divergence makes downstream bit parity impossible.

Audit LongiControl transitions consumed: **0 of the allowed 10,000**.

## Is a clean reproduction necessary?

A clean reproduction is not necessary to accept either existing final-policy
evaluation. It is scientifically necessary if the claim to resolve is causal—whether
RNG stream, runner schedule, or runtime accounts for the historical/current gap—or if
future work wants to call the MVE Real-Only arm a reproduction rather than a matched
within-runner control.

The smallest robust follow-up is a seed-11, 2×2, 300k reproduction after freezing an
exact dependency/container lock:

1. historical continuous/double-reset track stream + historical episode-bulk runner;
2. current explicit-seed track stream + historical episode-bulk runner;
3. historical track stream + current transition-interleaved runner;
4. current track stream + current runner.

All four should share one recorded initial policy state and record per-update replay
indices/RNG hashes. This isolates track-stream and schedule main effects plus their
interaction. Only expand to seeds 29 and 47 if seed 11 remains ambiguous. This audit
does not authorize or start that experiment.

## Limitations

- Historical run-producing commit/branch: `UNKNOWN`.
- Historical NumPy, Torch, and Gymnasium versions: `UNKNOWN`.
- Historical initial policy, RNG, replay, and environment states were not saved.
- Historical first-100 track hashes are current-runtime source reconstructions.
- No full training or performance experiment was run, so causal effect sizes remain
  unmeasured.
