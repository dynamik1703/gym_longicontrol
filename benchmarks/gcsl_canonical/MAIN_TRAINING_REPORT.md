# Canonical GCSL Main-Training Report

Status: **main training and GCSL_DIAGNOSTIC_V1 evaluation complete**.
GCSL_VALIDATION_V1 remains frozen and unopened; the immutable Constrained references were not evaluated; Paper remains sealed and uninspected.

This is development-style diagnostic evidence. It does not apply a final Validation RSR category, select a winner, or select a best checkpoint.

## Execution provenance

| Policy | Attempt ID | Resume | Transitions | Updates | Sampled tuples | Final SHA-256 | Walltime |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| G11 | `89a770a9-81b2-442c-a961-f04824496e46` | NOT_RESUMED | 300,000 | 7,250 | 1,856,000 | `f5002bf96a628ed6301eb21b483fa40e204435bed67cb7c940c8ba823bf07bdc` | 15.99 min |
| G29 | `60740736-81ec-4c44-b20f-05a7abc86f15` | NOT_RESUMED | 300,000 | 7,250 | 1,856,000 | `85c1db15981cb997d07551dd2bef2bcd5f97818fcd7263a4c6f19eb4e8047ad5` | 15.57 min |
| G47 | `fb7e881b-cb82-4a51-a868-a263f7865b5a` | NOT_RESUMED | 300,000 | 7,250 | 1,856,000 | `f70f8dd544e66d9ceeeee4e1d591d3035a0beb74ca0e4ae62b73d883c2d5bbec` | 12.99 min |

Execution commit: `0548163405bb4191d10099785d235015c4d34580`; frozen scientific configuration SHA-256: `c8807fd21cc2335b87bc9d1391970f89b7a2d7af7d385bb1253888185dcfedad`.
All three policies used the fixed canonical stream in exact prefix order; stream hashes and final cursors are recorded in the JSON report.

## Diagnostic curves

### Requirement-satisfaction rate (RSR)

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 11.1% (2/18) | 11.1% (2/18) | 5.6% (1/18) |
| G29 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| G47 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |

### Completion rate

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.0% (0/18) | 0.0% (0/18) | 5.6% (1/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) |
| G29 | 77.8% (14/18) | 94.4% (17/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) |
| G47 | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 83.3% (15/18) | 88.9% (16/18) |

### Deadline-compliance rate

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 94.4% (17/18) | 100.0% (18/18) | 100.0% (18/18) |
| G29 | 77.8% (14/18) | 94.4% (17/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) |
| G47 | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 100.0% (18/18) | 83.3% (15/18) | 88.9% (16/18) |

### Speed-compliance rate

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 100.0% (18/18) | 100.0% (18/18) | 77.8% (14/18) | 11.1% (2/18) | 11.1% (2/18) | 5.6% (1/18) |
| G29 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) |
| G47 | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 0.0% (0/18) | 5.6% (1/18) | 5.6% (1/18) |

### Mean final progress

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.000000 | 0.000000 | 0.267913 | 1.000000 | 1.000000 | 1.000000 |
| G29 | 0.978943 | 0.995978 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| G47 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.916704 | 0.941211 |

## Canonical training successes

| Policy | First transition | First episode | 50k intervals | Total |
| --- | ---: | ---: | --- | ---: |
| G11 | 147,496 | 83 | 0, 0, 2, 1, 0, 1 | 4 |
| G29 | 10,614 | 10 | 3, 3, 0, 2, 5, 0 | 13 |
| G47 | 29,655 | 34 | 1, 1, 0, 1, 1, 1 | 5 |

## Replay support and learning curves

### Cumulative canonical hindsight tuples

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0 | 0 | 0 | 24 | 46 | 69 |
| G29 | 86 | 173 | 256 | 335 | 410 | 476 |
| G47 | 20 | 71 | 96 | 110 | 136 | 158 |

### Maximum replay progress

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.523026 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| G29 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| G47 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### Expected future-lag mean (seconds)

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 9.090825 | 9.090825 | 9.069186 | 9.025919 | 9.012004 | 8.975730 |
| G29 | 8.573734 | 8.445089 | 8.316172 | 8.268146 | 8.207228 | 8.124479 |
| G47 | 8.022228 | 8.105348 | 8.251693 | 8.355800 | 8.418908 | 8.433993 |

### Transformed action NLL

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | -3.68494 | -4.00591 | -4.16806 | -4.8462 | -4.5364 | -4.55974 |
| G29 | -0.230168 | -0.531799 | -0.436669 | -0.304756 | -0.606659 | -0.433832 |
| G47 | -0.647195 | -0.235342 | -0.206919 | -0.303542 | -0.258594 | -0.286829 |

### Action MAE mean

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.698314 | 0.837214 | 0.868602 | 0.881885 | 0.849485 | 0.879718 |
| G29 | 0.728735 | 0.774009 | 0.741736 | 0.70002 | 0.793336 | 0.7408 |
| G47 | 0.565974 | 0.611391 | 0.609383 | 0.659526 | 0.659297 | 0.690429 |

### Action MSE mean

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 0.784547 | 0.910697 | 0.895702 | 0.896137 | 0.814665 | 0.851452 |
| G29 | 0.632983 | 0.71759 | 0.677243 | 0.630005 | 0.763278 | 0.709634 |
| G47 | 0.485451 | 0.527859 | 0.510895 | 0.565602 | 0.577871 | 0.591378 |

### Policy standard-deviation mean

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | 5.09975 | 5.88999 | 6.65009 | 6.69328 | 4.95428 | 5.16012 |
| G29 | 2.52786 | 2.42074 | 2.47206 | 2.39711 | 2.56902 | 2.52647 |
| G47 | 1.83658 | 1.65263 | 1.91395 | 1.93766 | 1.8225 | 1.93213 |

### Sampled policy entropy estimate mean

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G11 | -3.12017 | -3.44432 | -4.07676 | -3.99927 | -2.93762 | -2.97236 |
| G29 | -0.365192 | -0.317323 | -0.469426 | -0.234102 | -0.562023 | -0.48882 |
| G47 | -0.341199 | -0.0945246 | -0.169575 | -0.0809271 | 0.0299199 | -0.130589 |

Lag mean/median/p90 values in steps and seconds, total eligible tuple counts, latest sampled-batch lag summaries, terminal-future fractions, projected-goal distributions, source coverage, action-error distributions, norms, and ambiguity diagnostics are retained checkpoint-by-checkpoint in `main_training_results.json`. Median/p90 are exact expectations under the frozen truncated-geometric sampler at each replay state; the runner did not retain the entire empirical cumulative lag history, so that history is not fabricated.

## Final 300k diagnostic state

| Policy | RSR | Completion | Deadline | Speed | Mean progress | Failure modes | Feasible energy mean (kWh) | Mechanism label |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| G11 | 5.6% (1/18) | 100.0% (18/18) | 100.0% (18/18) | 5.6% (1/18) | 1.000000 | `{"feasible": 1, "speed": 17}` | 0.197523 | `SUCCESSFUL_BOOTSTRAP` |
| G29 | 0.0% (0/18) | 100.0% (18/18) | 100.0% (18/18) | 0.0% (0/18) | 1.000000 | `{"speed": 18}` | n/a | `CANONICAL_SUPPORT_NO_TRANSFER` |
| G47 | 0.0% (0/18) | 88.9% (16/18) | 88.9% (16/18) | 5.6% (1/18) | 0.941211 | `{"incomplete+deadline": 1, "incomplete+deadline+speed": 1, "speed": 16}` | n/a | `CANONICAL_SUPPORT_NO_TRANSFER` |

These mechanism labels are descriptive and preregistered. No final GCSL_ROBUST/PARTIAL/WEAK/ZERO category was applied, and no winner was chosen.

## Resource accounting

| Quantity | Actual | Frozen projection |
| --- | ---: | ---: |
| Matrix walltime | 0.7438 h | 0.2013 h |
| Retained disk | 0.831 GB | ≤ 12 GB gate |
| Scientific-checkpoint writes | 3.664 s | — |
| CPU time | unavailable (not retained by launchd runner) | — |
| Peak RSS | unavailable (not retained by launchd runner) | — |

## Integrity and seals

- Exactly 900,000 real transitions, 21,750 updates, and 5,568,000 sampled hindsight tuples were recorded.
- All 18 scientific checkpoint hashes verified against both manifest and metadata; 300k is final for every seed and no best checkpoint was selected.
- All replay sequence, episode-boundary, and same-episode future-endpoint checks passed; no nonfinite learning value was recorded.
- Diagnostics: exactly 324 episodes (3 policies × 6 checkpoints × 18 tracks), all on GCSL_DIAGNOSTIC_V1.
- No historical evaluation set was used.
- GCSL_VALIDATION_V1: `FROZEN_UNOPENED` (0 episodes).
- Canonical Constrained references: immutable and unevaluated on GCSL Validation.
- Paper: `SEALED_UNINSPECTED` (0 episodes; 0 content fingerprints).
- `main_training_enabled`: `false` after completion; Validation and Paper authorization remain `false`.

Full exact checkpoint metrics and integrity evidence are in `main_training_results.json`.
