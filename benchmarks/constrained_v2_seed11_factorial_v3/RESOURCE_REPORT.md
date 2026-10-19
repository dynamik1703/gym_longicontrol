# Resource preflight

The final measured gate passed.

| Measurement | Value | Limit |
|---|---:|---:|
| Projected total serial walltime | 1.181 h | 24 h |
| Projected retained checkpoint disk | 4.236 GB | 15 GB |
| `C_HIST` 2,048-transition pilot | 4.005 s | — |
| `C_CURRENT` 2,048-transition pilot | 3.482 s | — |
| Three-episode fixture evaluation | 1.135 s | — |

Each collector pilot executed its real frozen scheduling path, completed 2,048
transitions and 204 updates, and round-tripped a checkpoint. Evaluation used only
unreserved fixture seeds 0, 1, and 2; its performance was discarded. It changed
neither the actor nor global training RNG state and touched no V3 diagnostic task.

Tianshou replay storage is capacity-preallocated on first use. Disk and checkpoint
write projections therefore bound the live V3 pilot with larger verified full-100k
checkpoints from the completed Multi-Step MBRL study rather than incorrectly scaling
the already-capacity-sized pilot by `100000 / 2048`. Full measurements and the
conservative projection inputs are in `resource_measurements.json`.
