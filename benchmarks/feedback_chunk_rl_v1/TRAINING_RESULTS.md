# Training Results

All six preregistered runs completed their exact budgets in one immutable
attempt. No run resumed, restarted, or changed its Attempt ID. These are
training-process results, not held-out policy-performance estimates.

## Consumed budget and runtime

- Native transitions: 1,800,000 (300,000 per run).
- Gradient updates: 180,000 (30,000 per run).
- Scientific checkpoints: 36 total, at 50k, 100k, 150k, 200k, 250k, and
  300k for every run.
- Measured process time: 20,528.509 s (5.702 h).
- Measured active wall time: 20,843.116 s (5.790 h).
- Host-suspension time derived from UTC attempt spans minus active wall time:
  17.691 h, of which 16.740 h occurred during CHUNK3 seed 29 and 0.951 h
  during RESIDUAL1S seed 47. It is excluded from active compute.
- Retained run directory at reporting time: 2,853,736 KiB (2.72 GiB,
  2.92 GB), below the 12 GB limit.

The realized training process time is below the preregistered 24 h active
compute limit. The incomplete Diagnostic does not permit a realized total
study-compute comparison against the projection.

## Run status and immutable final artifacts

The actor SHA-256 is a reporting hash over actor tensors in lexicographic key
order: parameter name, NumPy dtype, JSON shape, and contiguous C-order bytes,
with NUL separators. The checkpoint SHA-256 is the primary immutable artifact
hash and was independently recomputed after training.

| Method | Seed | Status | Native | Macro | Updates | Process s | Active wall s | Actor SHA-256 | Checkpoint SHA-256 |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| CHUNK3 | 11 | COMPLETED | 300000 | 10119 | 30000 | 3253.481 | 3275.150 | `2bbcf78213215e266318357d1f33646c0fb52c78cd8f466e3b81dfd5fe3042c1` | `de59e6435e93f81b08931f414b8615226c424e025c44090e31d107d2e61770e0` |
| CHUNK3 | 29 | COMPLETED | 300000 | 10136 | 30000 | 3471.573 | 3585.711 | `80a20c356f33867d7880ae89d73621d05d3e66ebe1e3df51ab4c591de935488b` | `7b71c9c2206e0e3a5e96486812a8a85925b4ceef73bc7bdfce7a00482410939d` |
| CHUNK3 | 47 | COMPLETED | 300000 | 10129 | 30000 | 3689.924 | 3793.996 | `389af4d7081d6f620338d6776d67f7e4c680ed4c3d948c1255d0df18d63e59ba` | `207a5cc0feae49bab0a2027fcf1ab8432a073c7776a468f7d1af5f2d73ed813f` |
| RESIDUAL1S | 11 | COMPLETED | 300000 | 30105 | 30000 | 3283.143 | 3308.993 | `152d0c78036a4153d130b91397a148d21de1379b50147e959ec61f742b606f09` | `90a832a9839a6a7f22298c91e45d61d8e2d3f47a508dcd6f4a1475cd9a056591` |
| RESIDUAL1S | 29 | COMPLETED | 300000 | 30108 | 30000 | 3293.071 | 3321.171 | `1cf280b3b296287d95e9393a1676553438c38e4c6e34a4279d60c5d1d077b090` | `46e5aa9ce633a88dbca012db352f1d923daf68de7ced29c2c0a2aa1d8167bead` |
| RESIDUAL1S | 47 | COMPLETED | 300000 | 30105 | 30000 | 3537.317 | 3558.096 | `33cda1d183c387b95091972b1bf3479c371b446fdee928150993b9e6569153c8` | `282544146f0ece5a6b37b9032907149fe808100d5955bb933129538ecd363424` |

## Training behavior and action influence

`Influenced` is the fraction of all native proposals whose executed action
was changed by the accepted residual. `Accepted` is conditional on a residual
attempt. Full/cancelled fractions use recorded macro chunks. Mean macro length
is in native 0.1 s transitions.

| Method | Seed | Evolving training episodes feasible | Influenced | Accepted | Full chunks | Cancelled chunks | Mean macro length |
|---|---:|---:|---:|---:|---:|---:|---:|
| CHUNK3 | 11 | 248/257 | 50.468% | 97.666% | 45.143% | 52.614% | 29.647 |
| CHUNK3 | 29 | 253/260 | 48.431% | 97.428% | 43.054% | 54.607% | 29.597 |
| CHUNK3 | 47 | 248/256 | 47.940% | 97.480% | 42.176% | 55.583% | 29.618 |
| RESIDUAL1S | 11 | 235/244 | 58.199% | 94.956% | 59.060% | 40.259% | 9.965 |
| RESIDUAL1S | 29 | 239/245 | 58.596% | 94.950% | 59.097% | 40.189% | 9.964 |
| RESIDUAL1S | 47 | 235/243 | 57.252% | 95.084% | 58.681% | 40.674% | 9.965 |

The CHUNK3 training actions did not collapse to zero under the stochastic
training policy. Mean absolute actor components `(d0,d1,d2)` were
`(0.454,0.451,0.472)`, `(0.395,0.368,0.386)`, and
`(0.385,0.386,0.448)` for seeds 11, 29, and 47. Pairwise component
correlations `(d0-d1,d0-d2,d1-d2)` were `(0.241,0.172,0.295)`,
`(0.152,-0.007,0.197)`, and `(0.263,0.230,0.227)`. These values include
exploration and cannot substitute for deterministic Diagnostic mechanism
analysis.

## Critic and constraint diagnostics

All numeric values in the 301 sampled update-diagnostic records per run were
finite. Final sampled values were:

| Method | Seed | Reward critic loss | Cost critic loss | Alpha | Speed multiplier | Terminal multiplier |
|---|---:|---:|---:|---:|---:|---:|
| CHUNK3 | 11 | 1.5447 | 2.5057 | 0.03739 | 0.0000 | 5.4213 |
| CHUNK3 | 29 | 1.9093 | 4.5999 | 0.03854 | 1.6421 | 20.4146 |
| CHUNK3 | 47 | 2.2519 | 7.4900 | 0.04562 | 0.0000 | 17.1479 |
| RESIDUAL1S | 11 | 1.6776 | 1.3046 | 0.01960 | 0.3965 | 27.2532 |
| RESIDUAL1S | 29 | 1.0011 | 0.7396 | 0.02094 | 14.5678 | 11.8390 |
| RESIDUAL1S | 47 | 1.1904 | 0.6607 | 0.01785 | 22.2701 | 26.6903 |

Finite training diagnostics establish numerical completion, not policy
feasibility. The multipliers also show substantial seed variation, which would
have required the scheduled held-out Diagnostic to interpret.
