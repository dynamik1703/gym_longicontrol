# Constrained V2 — Three-Seed Collector Replication

This is the final collector-selection study for the canonical Constrained-V2
baseline. It compares the verified V3 episode-wise collector (`EPISODIC`) with
the current transition-wise collector (`TRANSITIONAL`) on identical, newly
canonicalized route-clean streams for nominal seeds 11, 29, and 47.

The implementation descends from completed V3 commit
`7d4b63aa021a8694bf51cca2ed319d55045bb7f8`. Completed studies are inputs only;
none is modified.

The frozen serial order is `E11`, `T11`, `E29`, `T29`, `E47`, `T47`. Every
policy receives 300,000 native transitions and 30,000 RL updates. Diagnostic
curves use `COLLECTOR_DIAGNOSTIC_V1`; the primary outcome is the single-pass
post-freeze evaluation on `COLLECTOR_VALIDATION_V1`. Paper seeds 4000–4017 are
never instantiated, fingerprinted, or evaluated.

Run preparation:

```bash
python -m benchmarks.constrained_v2_collector_replication.preflight --materialize
```

After every gate passes, prepare the detached job:

```bash
python -m benchmarks.constrained_v2_collector_replication.runner prepare-launchd
```

The matrix command performs exact same-Attempt recovery, completes all six
policies, opens Validation only after all final hashes verify, and materializes
the final results:

```bash
python -m benchmarks.constrained_v2_collector_replication.runner matrix
```

See `DESIGN.md`, `PROTOCOL.md`, `SPLIT_LEDGER.md`, `RESOURCE_REPORT.md`, and
`RESULTS.md`. Machine-readable evidence is frozen in `freeze.json`,
`split_ledger.json`, `stream_ledger.json`, `resource_measurements.json`, and
`results.json`.
