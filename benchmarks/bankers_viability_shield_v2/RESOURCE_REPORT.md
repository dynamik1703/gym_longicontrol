# Resource gate — BANKERS_V2_RESOURCE_STOP
The predeclared projection charges all 72*1800=129,600 native decisions the
maximum measured online+independent-replay+evidence-I/O cost, then adds 25%
overhead and measured preflight CPU. It was fixed before any Development task
was constructed, and was not changed after observing measurements.

Worst charged cost: 0.195653083 s/step.
Projected Development: 31695.80 s.
Measured preflight including final test CPU: 485.23 s.
Frozen total projection: **32181.03 s = 8.939 h**,
exceeding **21,600 s = 6 h**. Compute planning gate FAILED. No Development episode ran.

This conservative uniform worst-workload projection deliberately charges even
slow/stopped states and the cheaper reference arm the expensive workload. It is
not a proof that the actual 18-task sample would require that long, and no actual
six-hour overrun occurred. The negative outcome is a failure of this frozen
planning bound; it does not establish an intrinsic lower bound on BankerShield
cost. No certificate optimization, evidence reduction, panel reduction or revised
projection was used to turn this stopped study into a pass.

Projected retained V2 artifacts: 886766880 bytes
(0.826 GiB), below 3 GiB. Measured pre-report retained artifacts:
35845680 bytes. Peak RSS: 784465920 bytes
(748.1 MiB). Full evidence is approximately
46–47 kB canonical JSON per step, compressed at about 2.8–5.3 kB per step in the
resource probes; all fields and all 21 certificates are retained.

| Starting speed m/s | Mean mask ms | Max mask ms | Mean physics calls/mask | Replay ms/step |
|---:|---:|---:|---:|---:|
| 0.000000 | 1.407 | 1.496 | 21.0 | 13.064 |
| 5.555556 | 2.735 | 3.843 | 74.0 | 9.762 |
| 10.000000 | 14.673 | 24.452 | 384.0 | 24.964 |
| 16.666667 | 25.978 | 26.969 | 912.0 | 50.491 |
| 20.000000 | 34.255 | 35.100 | 1237.0 | 66.771 |
| 25.000000 | 50.471 | 52.086 | 1802.4 | 93.083 |
| 25.500000 | 50.714 | 51.819 | 1849.2 | 96.526 |

Fixture-weighted mean mask latency 6.653 ms, mean physics calls
193.59; observed worst latency 163.247 ms and maximum calls
1862. The worst observed query exceeds a 100 ms real-time period,
so these measurements do not establish a hard real-time 10-Hz guarantee.
Full fixture online/replay/I/O/CPU timings and exact evidence sizes are retained
per episode in fixture_results.json. resource_measurements.json includes final
native recertification in the online measurement; final certification at high
speed costs about 2.55 ms in the probe.

For one future three-seed masked arm (900k native steps), worst measured mean
mask+final cost projects 13.32 h. A scenario requiring a
second full mask per step projects 26.63 h, already
above 24 h without network updates, environment/controller overhead or evaluation.
Stored causal next-state masks might avoid duplicate computations; this was not
implemented or benchmarked as a learner. Mask-only one-query cost fits 24 h;
full training feasibility remains UNESTABLISHED. The complete proposed three-arm
comparison is nine runs / 2.7 million training steps, not 900k total. No training
was executed. See RL_READINESS.md for the prospective identical-algorithm design.

Final readback used 36.38 CPU s; additional resource disk replay
2.46 CPU s; preliminary checks 27.31 CPU s. With an explicit
300 s allowance for other preparation/reporting, accounted CPU is 851.38 s
(14.19 min), comfortably within the actual active-compute limit. These
additional sealing costs do not change or clear the failed projection gate.
