# Same-Information Scalar SAC Main-Training Report

Status: **main training and SAMEINFO_DIAGNOSTIC_V1 evaluation complete**.
SAMEINFO_VALIDATION_V1 remains frozen and unopened; Paper remains sealed and
uninspected.

This report is development-style diagnostic evidence only. It does not apply
the frozen final Validation classification and does not select a winner or a
best checkpoint.

## Execution provenance

| Seed | Attempt ID | Resume | Transitions | Updates | Final SHA-256 | Walltime |
| --- | --- | --- | ---: | ---: | --- | ---: |
| S11 | `7b3d8c81-1539-4cdf-8ab4-0e1b21ebb8fd` | NOT_RESUMED | 300,000 | 30,000 | `97abc187734f1664907b6ad4a6512715014a7c453687377f06ef4b443b22d249` | 26.67 min |
| S29 | `9fbcec36-3df3-4a1f-922e-a360d36537c6` | NOT_RESUMED | 300,000 | 30,000 | `6a17a80c2af7412ff7eee2448ce7f98bcff4c316b9f88ce167ae82e4c0228fba` | 35.35 min |
| S47 | `70c3d224-78ba-4d23-9c8b-9e56b0b9bc06` | NOT_RESUMED | 300,000 | 30,000 | `b28151f6f25e8510258119aa5f24a06d088980d9e536e044f529241785de53d1` | 24.42 min |

Actor and reward-critic initialization hashes were byte-identical to the
paired C11/C29/C47 references for every seed. Each run used its frozen
canonical ordered task/reset stream; the verified fingerprints are stored
in `main_training_results.json`.

## Diagnostic curves

### Requirement-satisfaction rate (RSR)

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | --- | --- | --- | --- | --- | --- |
| C11 | 27.8% (5/18) | 22.2% (4/18) | 50.0% (9/18) | 77.8% (14/18) | 55.6% (10/18) | 55.6% (10/18) |
| S11 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| C29 | 0.0% (0/18) | 38.9% (7/18) | 55.6% (10/18) | 33.3% (6/18) | 66.7% (12/18) | 33.3% (6/18) |
| S29 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| C47 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 88.9% (16/18) | 61.1% (11/18) | 66.7% (12/18) |
| S47 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |

### Completion rate

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | --- | --- | --- | --- | --- | --- |
| C11 | 100.0% (18/18) | 50.0% (9/18) | 72.2% (13/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) |
| S11 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| C29 | 0.0% (0/18) | 61.1% (11/18) | 100.0% (18/18) | 61.1% (11/18) | 100.0% (18/18) | 61.1% (11/18) |
| S29 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| C47 | 0.0% (0/18) | 0.0% (0/18) | 27.8% (5/18) | 100.0% (18/18) | 100.0% (18/18) | 77.8% (14/18) |
| S47 | 0.0% (0/18) | 0.0% (0/18) | 5.6% (1/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |

### Deadline-compliance rate

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | --- | --- | --- | --- | --- | --- |
| C11 | 88.9% (16/18) | 27.8% (5/18) | 55.6% (10/18) | 77.8% (14/18) | 55.6% (10/18) | 55.6% (10/18) |
| S11 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| C29 | 0.0% (0/18) | 44.4% (8/18) | 100.0% (18/18) | 33.3% (6/18) | 100.0% (18/18) | 38.9% (7/18) |
| S29 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| C47 | 0.0% (0/18) | 0.0% (0/18) | 11.1% (2/18) | 94.4% (17/18) | 100.0% (18/18) | 66.7% (12/18) |
| S47 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |

### Speed-compliance rate

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | --- | --- | --- | --- | --- | --- |
| C11 | 27.8% (5/18) | 83.3% (15/18) | 88.9% (16/18) | 100.0% (18/18) | 94.4% (17/18) | 100.0% (18/18) |
| S11 | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 77.8% (14/18) | 100.0% (18/18) | 100.0% (18/18) |
| C29 | 100.0% (18/18) | 88.9% (16/18) | 55.6% (10/18) | 100.0% (18/18) | 66.7% (12/18) | 94.4% (17/18) |
| S29 | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 88.9% (16/18) | 94.4% (17/18) | 100.0% (18/18) |
| C47 | 100.0% (18/18) | 100.0% (18/18) | 22.2% (4/18) | 94.4% (17/18) | 61.1% (11/18) | 100.0% (18/18) |
| S47 | 100.0% (18/18) | 100.0% (18/18) | 50.0% (9/18) | 100.0% (18/18) | 100.0% (18/18) | 94.4% (17/18) |

## Final 300k diagnostic comparison

| Pair | C RSR | S RSR | Δ count (C − S) |
| --- | ---: | ---: | ---: |
| C11 vs S11 | 55.6% (10/18) | 0.0% (0/18) | +10/18 |
| C29 vs S29 | 33.3% (6/18) | 0.0% (0/18) | +6/18 |
| C47 vs S47 | 66.7% (12/18) | 0.0% (0/18) | +12/18 |

These are diagnostic paired differences only. No frozen Validation
classification was applied.

### Final failure modes and feasible-only energy

| Policy | Failure modes | Behavior states | Feasible energy mean (kWh) |
| --- | --- | --- | ---: |
| C11 | `{"COMPLETED_DEADLINE_VIOLATION": 8, "FEASIBLE": 10}` | `{"crawling": 0, "speed_violation": 0, "standstill": 0, "timely_completion": 10}` | 0.172692 |
| S11 | `{"INCOMPLETE_SAFE": 18}` | `{"crawling": 18, "speed_violation": 0, "standstill": 0, "timely_completion": 0}` | n/a |
| C29 | `{"COMPLETED_DEADLINE_VIOLATION": 4, "COMPLETED_SPEED_VIOLATION": 1, "FEASIBLE": 6, "INCOMPLETE_SAFE": 7}` | `{"crawling": 7, "speed_violation": 1, "standstill": 0, "timely_completion": 7}` | 0.199959 |
| S29 | `{"INCOMPLETE_SAFE": 18}` | `{"crawling": 18, "speed_violation": 0, "standstill": 0, "timely_completion": 0}` | n/a |
| C47 | `{"COMPLETED_DEADLINE_VIOLATION": 2, "FEASIBLE": 12, "INCOMPLETE_SAFE": 4}` | `{"crawling": 4, "speed_violation": 0, "standstill": 0, "timely_completion": 12}` | 0.171725 |
| S47 | `{"INCOMPLETE_SAFE": 17, "INCOMPLETE_WITH_SPEED_VIOLATION": 1}` | `{"crawling": 18, "speed_violation": 1, "standstill": 0, "timely_completion": 0}` | n/a |

## Scalar training dynamics

| Seed | 50k α | 100k α | 150k α | 200k α | 250k α | 300k α | Final actor loss | Final Q loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S11 | 0.223151 | 0.0502792 | 0.0114111 | 0.00292509 | 0.00137874 | 0.000961806 | 0.467935 | 0.0366956 |
| S29 | 0.223383 | 0.0503793 | 0.0115264 | 0.00271401 | 0.000759427 | 0.000608039 | 0.755946 | 0.034675 |
| S47 | 0.223304 | 0.0503874 | 0.0116081 | 0.00299464 | 0.000831237 | 0.000626068 | 1.42658 | 0.0507533 |

### Retained replay scalar-reward mean

| Seed | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S11 | -0.0353999 | -0.0367259 | -0.0412263 | -0.0399973 | -0.0344843 | -0.0355139 |
| S29 | -0.0411016 | -0.04048 | -0.0397443 | -0.0414968 | -0.0439686 | -0.0382898 |
| S47 | -0.0439145 | -0.0418468 | -0.0387222 | -0.0432268 | -0.0438659 | -0.0377804 |

All retained Scalar network parameters, optimizer state, rewards, and six
checkpoint loss snapshots were finite. Full-training scalar reward
distributions were not retained; the JSON reports clearly scoped final
100k-replay reward statistics instead.

## Descriptive mechanism inspection

- S11: 0 standstill, 18 crawling, 0 timely-completion, and 0 speed-violation episodes; RSR 0/18, speed compliance 18/18.
- S29: 0 standstill, 18 crawling, 0 timely-completion, and 0 speed-violation episodes; RSR 0/18, speed compliance 18/18.
- S47: 0 standstill, 18 crawling, 0 timely-completion, and 1 speed-violation episodes; RSR 0/18, speed compliance 17/18.

Final Scalar diagnostic RSR count range across seeds was 0–0 of 18. These descriptions are observational only and did not alter training.

## Resource accounting

| Quantity | Actual | Frozen projection |
| --- | ---: | ---: |
| Launch preparation → final Scalar | 1.548 h | 0.895 h |
| First Attempt start → final Scalar | 1.441 h | — |
| Retained disk allocation | 1.639 GB | 3.177 GB |
| CPU time | unavailable (not retained by launchd runner) | — |
| Peak RSS | unavailable (not retained by launchd runner) | — |

The rolling 10k recovery checkpoint overwrites its predecessor, so only a
lower bound from retained checkpoint write metadata is available. Per-episode
diagnostic walltimes were likewise not retained; missing values are not
fabricated.

## Integrity and seals

- Scalar budgets: 900,000 real transitions and 90,000 updates exactly.
- Diagnostics: 648 episodes exactly (324 Scalar + 324 immutable Constrained).
- All Scalar final checkpoint hashes and all 18 reference hashes verified.
- No Constrained policy was trained or resumed.
- No best checkpoint was selected; 300k is final for every seed.
- `SAMEINFO_VALIDATION_V1`: `FROZEN_UNOPENED` (0 episodes).
- Paper: `SEALED_UNINSPECTED` (0 episodes).
- `main_training_enabled`: `false` after completion.
