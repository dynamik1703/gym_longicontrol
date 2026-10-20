# Preparation resource report

The final selected Budget A implementation was benchmarked on CPU with PyTorch
2.14.1. Two calibration passes consumed 2,000 total preparation-only simulator
transitions on dedicated seed 98,000,000. Those transitions were discarded,
were not used for learning, and were not interpreted as policy performance.
The final recorded pass used 30 synthetic updates and a 5,000-row synthetic
replay. No GCSL diagnostic, GCSL Validation, Constrained Validation, or Paper
episode ran.

Measured final-pass values:

- Collection: 2,806.11 transitions/s.
- Deterministic actor inference: 0.1155 ms/action.
- Hindsight batch sampling: 0.3228 ms/batch.
- Batch-256 supervised update: 5.9163 ms/update.
- 5,000-row checkpoint: 4.043 MB, 56.08 ms write, 26.55 ms restore.
- Projected 300,000-row checkpoint: 48.311 MB and 0.670 s/write.

Diagnostic time was projected from measured collection throughput at the
conservative maximum of 1,800 transitions for each of 18 episodes; no policy
evaluation was performed. The resulting projections are:

- One 300k policy excluding diagnostics: 172.25 s.
- Three policies excluding diagnostics: 516.74 s.
- All 18 scheduled diagnostic sets: 207.83 s.
- Three-policy main training including scheduled diagnostics: 724.57 s
  (0.2013 h).
- Future 54-episode GCSL Validation: 34.64 s.
- Future 54-episode contextual Constrained Validation: 34.64 s.
- Retained artifacts: 1.551 GB.

Both frozen hard gates pass: projected three-policy main training is below 12
hours and projected retained artifacts are below 12 GB. Raw measurements are
in `resource_measurements.json`; the reproducible bounded probe is
`resource_check.py`.
