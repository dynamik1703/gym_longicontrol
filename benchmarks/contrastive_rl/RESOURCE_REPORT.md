# Preparation resource report

Measured on 2026-10-07 on an Apple arm64 host (12 logical CPUs, 32 GiB RAM),
Python 3.10.14, JAX/JAXlib 0.4.23, Flax 0.7.4, Optax 0.1.7, CPU device
`TFRT_CPU_0`. The isolated environment used upstream's SciPy 1.12.0 pin after
the unpinned resolver initially selected an incompatible macOS SciPy 1.15.3
binary. No GPU, cloud service, W&B, or telemetry was used.

These are synthetic engineering measurements, not control performance.
`resource_measurements.json` is the machine-readable record.

| Measurement (batch 256) | Depth 4 | Depth 16 |
|---|---:|---:|
| Residual blocks per network | 1 | 4 |
| Actor parameters | 270,594 | 1,066,242 |
| State-action encoder parameters | 285,760 | 1,081,408 |
| Goal encoder parameters | 283,456 | 1,079,104 |
| Total trainable parameters (including alpha) | 839,811 | 3,226,755 |
| Actor compile + first batched call | 0.183 s | 0.555 s |
| Steady actor call, batch 256 | 1.023 ms | 4.435 ms |
| Full update compile + first call | 6.653 s | 39.649 s |
| Steady actor+alpha+critic update | 25.50 ms | 68.81 ms |
| Observed process RSS increase | 344 MB | 744 MB |

The RSS figures are before/after process observations, not allocator-level peak
memory. The depth-16 measurement started after depth 4 in the same process, so
its baseline includes reusable JAX state and compiled artifacts. Host-to-JAX-CPU
placement of one batch measured 0.073 ms and 0.065 ms respectively; there was
no CPU-to-accelerator path to measure. Actor latency is a batched CPU result,
not single-step simulator latency and not a GPU extrapolation.

Each depth executed one compile/first update plus 20 steady synthetic updates:
21 per depth, 42 total. All final synthetic metrics were finite. Depth 64 was
not profiled because it is not a main condition and the large depth-16 compile
overhead already answers first-tier CPU feasibility without spending more
preparation budget.

The unchanged Gymnasium simulator executed exactly 1,000 transitions using
fixed zero action on Development seed 2000, with no reset and no learning. It
measured 17,527 transitions/s over 0.057 s. This stationary-action microbenchmark
is an optimistic adapter/physics throughput check, not a training-speed claim.
No physical trajectory was placed in replay.

## Feasibility interpretation

Both planned depths fit comfortably in available host memory at batch 256 and
run finite CPU updates. Depth 16 has 3.84× the trainable parameters and 2.70×
the measured steady update time of depth 4; fixed width is not equal-compute.
Compilation, especially 39.6 s for depth 16, must be separated from steady
timing in any later run.

At the draft one-update-per-40-transitions schedule, a 300k policy would perform
about 7,250 updates after a 10k prefill (`(300k-10k)/40`), not counting any
still-unresolved semantic change. Multiplying measured isolated update times
gives roughly 3.1 minutes (depth 4) or 8.3 minutes (depth 16) of pure steady
JAX update compute. These are arithmetic extrapolations only: replay sampling,
Python orchestration, checkpointing, evaluation, and serial environment time
are excluded. No policy run was used to choose this schedule.

Resource readiness is positive; semantic readiness is not. Hardware is not the
current blocker.
