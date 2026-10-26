# Observed computation and limits

Certificate-only microbenchmarks use exact production `advance`, the BMW-i3
power model and the audited runtime; no environment reset or step is performed.
Each state has ten repeated measurements. No surrogate physics or unchecked
cache is used. Values measure this host only.

| Initial speed m/s | Full 21 mask ms | Physics predictions/mask | Admitted points | Backup cert ms | Final cert ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000000 | 1.284 | 21 | 21 | 0.067 | 0.064 |
| 5.555556 | 1.546 | 32 | 21 | 0.061 | 0.112 |
| 10.000000 | 9.356 | 342 | 21 | 0.416 | 0.464 |
| 16.666667 | 24.767 | 859 | 21 | 1.216 | 1.241 |
| 20.000000 | 34.409 | 1188 | 21 | 1.465 | 1.516 |
| 25.000000 | 49.327 | 1760 | 21 | 2.242 | 2.462 |

The full per-action physics counts, finite-prefix backup counts, memory/mask
serialized sizes and final verification predictions are in
resource_measurements.json. Thus both the 21-action query and its individual
certificates are accounted for. More permissive high-speed states cost more
because their entire braking prefixes are simulated with the original power
model.

The retained constant-20 fixture measured mean mask time
1.396 ms, mean 21.000 physics
predictions per mask, and mean final check time
0.075 ms. The later independent disk replay
took 0.955 s. Supplemental benchmark CPU
1.309 s; process peak resident memory
113.62 MiB. Retained study artifacts measured
4675998 bytes at report generation, far below 3 GiB.

The original preflight aborted before finalizing its total process CPU record
or full-panel compute/storage projection. Those fields remain explicitly
unknown; the study does not claim the six-hour full-panel gate passed. The
configured limits remain 21600 active seconds, 3 GiB, 54 Development episodes,
zero RL transitions and zero gradients. Development was never opened, and no
scientific panel was reduced or substituted to fit resources.

Runtime: Python 3.11.13, NumPy 2.4.6, Gymnasium
1.4.0. Exact executable and imported local production path are
retained in resource_measurements.json. The existing virtual environment was
used read-only; no historical package, model or checkpoint was updated.
