# V3 results

## Outcome

`COMPLETED`

All four seed-11 policies reached exactly 300,000 real transitions and 30,000 RL
updates. Only the frozen 18-task route-clean Diagnostic set was evaluated. Paper
remained sealed; legacy Development and Validation received zero policy episodes.

## Primary 300k endpoint

Counts are RSR / completion / deadline / speed, each out of 18.

| Condition | Counts |
|---|---:|
| A | 9 / 18 / 9 / 18 |
| B | 7 / 16 / 7 / 18 |
| C | 10 / 18 / 10 / 18 |
| D | 3 / 4 / 3 / 18 |

## Diagnostic RSR curves

| Condition | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 5 | 11 | 16 | 2 | 9 |
| B | 0 | 2 | 1 | 3 | 14 | 7 |
| C | 1 | 9 | 11 | 15 | 13 | 10 |
| D | 0 | 10 | 6 | 5 | 0 | 3 |

## Interpretation

- Factorial classification: `INTERACTION_DOMINANT`.
- Canonical baseline recommendation:
  `BOTH_STREAM_GENERATION_AND_COLLECTOR_MUST_BE_CANONICALIZED`.
- D reproduction: `EXACT_CURRENT_REGIME_REPRODUCTION`.
- A is a reconstructed historical-direction regime, not an exact historical run.
- These are descriptive single-seed contrasts; no significance claim is made.
- Future evaluation must follow `Route-clean Policy R`.

## Execution integrity

- Attempts: 4.
- Exact resumes: 1.
- Real transitions: 1200000.
- RL gradient updates: 120000.
- Diagnostic episodes: 432.
- Legacy Development / Validation / Paper episodes: 0 / 0 / 0.

Machine-readable curves, endpoint episodes and failure modes, feasible-only energy,
episode-level PID/lambda/alpha trajectories, exposure statistics, normalized factorial
contrasts, reproduction anchors, and integrity evidence are in `results.json`.
