# Observed resource tail attribution

The V3 native-event maxima are matched to the same event indices in component timing arrays. Journal append includes canonical row/hash serialization, write, file fsync and directory fsync. Its wall-minus-process-CPU time establishes a non-CPU wait inside this evidence-I/O operation; it does not identify the exact kernel/filesystem/scheduler cause. This is separate from Banker calculation or actor inference. V2 lacks this event-matched instrumentation, so the exact cause of its historical stall remains unidentified.

| Arm | Complete wall s | Process CPU s | Journal append wall s | Journal append CPU s | Actor wall s |

|---|---:|---:|---:|---:|---:|

| BANKER_POST_SAC | 0.937337 | 0.012428 | 0.920800 | 0.001240 | 0.000361 |

| BANKER_PRE_SAC | 0.310411 | 0.012174 | 0.294519 | 0.001367 | 0.000357 |

| RAW_DISCRETE_SAC | 1.973641 | 0.005117 | 1.962230 | 0.001592 | 0.000318 |

All events remain in the sealed raw data and block admission rates. The separate repeated-individual-maxima wall stress forecast is 161.4145 hours; the preregistered block-rate admission is 11.491876 hours. Neither number is an observed training duration. Hard runtime CPU/storage monitors remain mandatory.
