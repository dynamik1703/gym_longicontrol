# Collision findings

## Critical 500-task streams

All three known Development-2006 findings reproduce at both fingerprint levels:

| Stream | Episode | Reset seed | RNG draw | Development seed | Route match | Full-task match |
|---|---:|---:|---:|---:|---:|---:|
| T_HIST | 158 | unavailable (`reset(None)`) | 316 | 2006 | yes | yes |
| T_CURRENT | 163 | 1418256283 | 163 | 2006 | yes | yes |
| T_CURRENT | 242 | 274583042 | 242 | 2006 | yes | yes |

The shared fingerprints are:

- `route_sha256 = 19067fb1a1ab19fad62d1ed1c4d49249a5c9c1f95424bfb5a606d09aa01ac324`
- `full_task_sha256 = a1af0095c4fa5c954fe7040b176171e98d90b9d1bfda7404d617b1362b272963`

There are no other Development collisions and no route or full-task collision with
Validation `3000..3008` in either 500-task prefix.

## Internal duplicates

| Stream | Tasks | Unique routes | Duplicate rate | Max multiplicity | Multiplicity distribution |
|---|---:|---:|---:|---:|---|
| T_HIST | 500 | 482 | 3.6% | 3 | 467×1, 12×2, 3×3 |
| T_CURRENT | 500 | 490 | 2.0% | 4 | 482×1, 7×2, 1×4 |

The V2 task fields are fixed, so full-task statistics equal route statistics here.
Duplicate rate means `(episodes - unique content) / episodes`, not the fraction of
unique hashes that have multiplicity greater than one.

## Empirical split-match rates

| Stream | Split | Route probability | Full-task probability | Expected matches at 100 / 200 / 300 / 500 |
|---|---|---:|---:|---|
| T_HIST | Development | 0.002 | 0.002 | 0.2 / 0.4 / 0.6 / 1.0 |
| T_CURRENT | Development | 0.004 | 0.004 | 0.4 / 0.8 / 1.2 / 2.0 |
| T_HIST | Validation | 0.000 observed | 0.000 observed | 0 / 0 / 0 / 0 observed |
| T_CURRENT | Validation | 0.000 observed | 0.000 observed | 0 / 0 / 0 / 0 observed |

These are plug-in estimates from one 500-task stream, not exact generator
probabilities. Zero observed Validation matches is evidence for these finite prefixes,
not proof that future draws have zero collision probability.

## Completed-study audit

Counts below are task occurrences across completed policy runs. Repeated content in two
policies is counted twice because each policy was exposed separately.

| Study | Reconstruction | Dev route/full | Validation route/full | Dev integrity | Validation integrity |
|---|---|---:|---:|---|---|
| Historical Constrained V2 | PARTIAL | 1 / 1* | 0 / 0* | UNKNOWN | UNKNOWN |
| Current MVE matched Real-Only | EXACT | 2 / 2 | 0 / 0 | FULL_TASK_OVERLAP | CLEAN |
| H=1 Physics MBRL | EXACT | 0 / 0 | 0 / 0 | CLEAN | CLEAN |
| Requirement-Conditioned RL | PARTIAL | 2 / 2* | 1 / 1* | UNKNOWN | UNKNOWN |
| Matched scalar SAC | PARTIAL | 0 / 0* | 0 / 0* | UNKNOWN | UNKNOWN |
| HER | EXACT | 0 / 0 | 0 / 0 | CLEAN | CLEAN |
| CRL | EXACT | 2 / 2 | 0 / 0 | FULL_TASK_OVERLAP | CLEAN |
| Binary Success | PARTIAL | 1 / 1* | 0 / 0* | UNKNOWN | UNKNOWN |
| LLM Reward final policies | EXACT | 1 / 1 | 0 / 0 | FULL_TASK_OVERLAP | CLEAN |

An asterisk is current-runtime reconstruction evidence, not an assertion about exact
historical task bytes. Partial studies remain `UNKNOWN` even when the indicative prefix
has zero collisions.

Exact Development collision details:

- Current MVE Real-Only seed 11 episode 163 (`1418256283`) = Development 2006.
- Current MVE Real-Only seed 29 episode 210 (`2144948820`) = Development 2001.
- CRL depth 4 seed 11 episode 63 (`2449317112`) = Development 2001.
- CRL depth 16 seed 11 episode 63 (`2449317112`) = Development 2001.
- LLM Reward seed 11 episode 203 (`reset(None)`) = Development 2006.

The second T_CURRENT/2006 collision at episode 242 is outside the completed seed-11
Real-Only run's 223 tasks, but inside the required conservative 500-task critical
prefix.

The partial Requirement-Conditioned reconstruction contains a possible Validation
3006 full-task collision at seed 47 episode 245 (RNG draw 736). Because its historical
NumPy/Gymnasium runtime and realized tasks were not preserved, this is an audit warning,
not a definitive leakage claim; its integrity classification is `UNKNOWN`.

Development is a repeatedly observed calibration/model-selection split, so overlap can
make Development curves optimistic but does not have the same evidentiary status as
Validation leakage. Validation is the relevant final-generalization check. No historical
score was changed, deleted, or relabeled by this audit.

## Why repeats occur

The seed namespace is not the content space. The generator chooses among nine possible
100 m transition sites, uses a discrete transition count, and samples limits on the
20..100 km/h grid with at most a 40 km/h adjacent change and no equal neighbors.

Exact enumeration of the implemented support gives **265,949,468 route contents**.
Fixed V2 has the same number of full-task contents. That count is much smaller than an
unbounded nominal `SeedSequence` namespace and, more importantly, is highly
non-uniform: the generator assigns 20% probability to the one-transition stratum, which
contains only **351** routes. The remaining transition counts 2..9 each receive 10%.
The observed repeats therefore arise from discrete templates and coarse limits plus a
skewed count distribution—not hash canonicalization and not a deliberate finite
catalogue keyed one-to-one by seed.
