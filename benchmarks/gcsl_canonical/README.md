# Canonical Goal-Conditioned Supervised Learning

This directory contains the completed LongiControl canonical GCSL
main-training study. It directly supervises an executed action at a source
state under an actually achieved strict-future projected goal. It consumes no
task reward or cost and learns no critic.

The training inputs are fixed canonical training streams with disclosed
historical-content overlap (G11/G29/G47). The scientific integrity condition is
`current_study_route_clean_verified`: `GCSL_DIAGNOSTIC_V1` and
`GCSL_VALIDATION_V1` are route- and full-task-disjoint from every reachable
training task. See `SPLIT_SCOPE_GATE.md`, `historical_overlap_ledger.json`, and
`SPLIT_LEDGER.md`.

Supporting records:

- `SOURCE_AUDIT.md`: primary-source and immutable old-port audit.
- `DESIGN.md`: exact state, goal, sampler, likelihood, and budget decision.
- `PROTOCOL.md`: split, recovery, evaluation, and authorization rules.
- `RESOURCE_REPORT.md`: bounded preparation benchmark and hard gates.
- `preparation_status.json`: machine-readable readiness state.
- `MAIN_TRAINING_REPORT.md`: development-style main-training diagnostics.
- `main_training_results.json`: machine-readable main-training results.

Safe status commands:

```bash
python -m benchmarks.gcsl_canonical.cli preflight
python -m benchmarks.gcsl_canonical.cli status
```

The authorized G11/G29/G47 matrix is complete and `main_training_enabled` is
false, so `matrix` and `resume` fail closed. Validation is separately gated and
remains frozen. There is no Paper-opening command.

The retained heartbeat/status artifacts can be read with `status`; this does
not add simulator work, checkpoints, or evaluation events.
