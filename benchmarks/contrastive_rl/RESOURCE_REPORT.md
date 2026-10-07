# Preparation resource report

Measured on 2026-10-07 on an Apple arm64 host (12 logical CPUs, 32 GiB RAM),
Python 3.10.14, JAX/JAXlib 0.4.23, Flax 0.7.4, Optax 0.1.7, CPU device
`TFRT_CPU_0`. No GPU, cloud service, W&B, or telemetry was used.

These are synthetic engineering measurements, not policy performance. The
initial four-dimensional exact-outcome preparation remains recorded in
`resource_measurements.json`; the three-dimensional projected-goal revision is
in `resource_measurements_projected.json`.

## Projected-goal network measurements

| Measurement (batch 256) | Depth 4 | Depth 16 |
|---|---:|---:|
| Residual blocks per network | 1 | 4 |
| Actor parameters | 270,338 | 1,065,986 |
| State-action encoder parameters | 285,760 | 1,081,408 |
| Goal encoder parameters | 283,200 | 1,078,848 |
| Total trainable parameters (including alpha) | 839,299 | 3,226,243 |
| Actor compile + first batched call | 0.168 s | 0.582 s |
| Steady actor call, batch 256 | 1.266 ms | 4.599 ms |
| Full update compile + first call | 6.599 s | 46.765 s |
| Steady actor+alpha+critic update | 25.58 ms | 73.58 ms |
| Observed process RSS increase | 336 MB | 537 MB |

The three-dimensional goal reduces Actor and Goal-Encoder input projections by
256 parameters each relative to the initial preparation. State-action encoder
size is unchanged. The RSS figures are process before/after observations, not
allocator-level peak memory; depth 16 ran after depth 4 in the same process.
There was no CPU-to-accelerator transfer path.

The revision used one compile/first update plus 19 steady updates: exactly 20
synthetic full cycles per depth, 40 total, all finite. These are separate from
the previous 21 cycles per depth. No simulator data entered either check.

## Projection collision measurement

The revision consumed exactly 500 additional native simulator transitions with
fixed zero action on Development seed 2001. They were not placed in replay and
were not used for learning. The stationary script produced:

| Quantity | Value |
|---|---:|
| Unique raw physical outcomes | 500 |
| Unique projected goals | 1 |
| Raw equivalent-pair rate | 0.0 |
| Projected equivalent-pair rate | 1.0 |
| Canonical projected outcomes | 0 |

Time makes every raw row unique while stationary progress and requirement bits
are identical. This is a deliberate collision stress case, not a claim about a
future behavior-policy distribution. Diagonal InfoNCE remains unchanged and
therefore treats the identical columns as competing references.

## Cumulative preparation resources

- Initial preparation: 1,000 fixed-action Development transitions and 21
  synthetic cycles per depth.
- Projection revision: 500 fixed-action Development transitions and 20
  synthetic cycles per depth.
- Execution infrastructure: 3 fixed-action Development transitions on seed
  2002 to verify bit-identical environment continuation after serialization.
  Two small-core synthetic update comparisons verified checkpoint continuation;
  neither used a depth-4/depth-16 physical policy.
- Cumulative physical transitions: 1,503, all preparation-only.
- Policy-training transitions: 0.
- Validation and paper-test transitions: 0.

## Planned-budget interpretation

The frozen schedule would execute 7,250 cycles per 300k policy. At the revised
steady CPU timings, pure update arithmetic is about 3.1 minutes for depth 4 and
8.9 minutes for depth 16. These are extrapolations excluding replay sampling,
Python orchestration, checkpoints, evaluation, and environment time. No policy
result selected any setting.

Both depths remain technically feasible on the measured host. Execution and
checkpoint infrastructure is now tested; the only remaining blocker is a
separate explicit authorization, not network memory or update throughput.
