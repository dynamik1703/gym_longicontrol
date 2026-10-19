# Constrained V2 collector replication results

## Outcome

`COMPLETED`

All six policies reached exactly 300,000 real transitions and 30,000 RL
updates. The frozen Validation set was opened once only after all six final
checkpoint hashes verified. Paper remained sealed.

## Primary paired Validation outcome

RSR columns are EPISODIC / TRANSITIONAL / difference, each seed out of 18.
Other columns are paired EPISODIC-minus-TRANSITIONAL differences.

| Seed | RSR E / T / Δ | Completion Δ | Deadline Δ | Speed Δ |
|---:|---:|---:|---:|---:|
| 11 | 13 / 10 / +3 | +0 | +0 | +3 |
| 29 | 5 / 9 / -4 | +7 | -3 | -5 |
| 47 | 2 / 9 / -7 | -7 | -6 | -3 |

Pooled RSR: **20 / 54
EPISODIC** versus **28 / 54
TRANSITIONAL** (Δ
-8).

## Interpretation and decision

- Classification: `SEED_HETEROGENEOUS`.
- Decision: `DO_NOT_FREEZE_COLLECTOR`.
- Selected collector: `None`.
- This is the frozen descriptive rule, not a significance test.

## Integrity

- Real transitions / RL updates: 1800000 /
  180000.
- Diagnostic / Validation / Paper episodes: 648 /
  108 / 0.
- Mechanism diagnostics, learning curves, replay-at-update summaries, paired
  completion/deadline/speed comparisons, and raw Validation episodes are in
  `results.json`.

## Post-processing recovery

The frozen finalizer exited after Validation because it compared the
alphabetically serialized `runs` object order with the preregistered condition
order. The recovery wrapper verified the authoritative `condition_order` and
run set, then aggregated the already-sealed artifacts. It repeated no training
or evaluation and modified no sealed artifact. Full recovery provenance is in
`results.json`.
