# Computational feasibility report

Measured 2026-10-13 on macOS 27.0.1 arm64, Python 3.11.13, Torch 2.14.1,
NumPy 2.4.6, FSRL 0.1.0 at the pinned revision, and four Torch CPU threads.
The workload used the actual pinned actor/critics, batch size 256, vectorized exact
physics, `M=60`, real replay, target construction, optimizer steps, collection, and a
verified atomic checkpoint. Initialization was warmed before five timed target/update
repetitions. Raw values are frozen in `resource_measurements.json`.

| Measurement | Result |
|---|---:|
| Exact batched physics | 706,946.06 state-steps/s |
| Target construction, batch 256 and M=60 | 0.047491 s/update |
| Complete real-only SACLag update | 0.007758 s/update |
| Complete MVE SACLag update | 0.063449 s/update |
| Actor/environment collection | 1,695.72 transitions/s |
| Verified 2,048-row checkpoint round trip | 0.126055 s |
| Checkpoint artifact | 3,819,597 bytes |

The full-protocol arithmetic projection includes 300,000 collection transitions,
30,000 updates, six conservative nine-track Development evaluations (allowing the full
1,800-step episode limit), and 30 checkpoints per policy. Checkpoint time is
conservatively scaled linearly from 2,048 to the 100,000-row replay capacity.

| Projection | Hours |
|---|---:|
| Collection per policy | 0.0491 |
| Development per policy | 0.0159 |
| Checkpointing per policy | 0.0513 |
| Matched real-only policy | 0.1810 |
| MVE policy | 0.6451 |
| All six policies, serial | 2.4783 |
| CPU-hours at four threads | 9.9134 |

These are local arithmetic extrapolations from short warmed measurements, not a policy
pilot or a runtime guarantee; filesystem pressure and long-running host contention can
increase walltime. Even so, the margins are large relative to the preregistered gates:

```text
MVE policy: 0.645 h <= 12 h                         PASS
six-policy serial: 2.478 h <= 36 h                 PASS
computational_feasibility_verified = true
```

The final benchmark consumed 512 simulator transitions on explicit seed 0 solely to
measure collection throughput. An earlier provenance-check measurement consumed 512
more unreserved training-stream transitions, so cumulative preparation use is 1,024.
Neither run used a reserved Development, Validation, or paper-final seed or measured a
policy outcome. This remains within the 5,000-transition preparation limit.
