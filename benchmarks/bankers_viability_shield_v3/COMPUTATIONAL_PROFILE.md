# Computational cost audit and matched latency
The deterministic workload was fixed before fast implementation: eight synthetic
constant100km/h states, five repeats each, all 21 actions and selected recertification.
The source-only initial probe at 100km/h had no certified action; the raw-grid
profiling harness was corrected before any profile artifact was sealed. This was
an engineering probe, not a historical scientific rerun. baseline_profile.json
retains workload, all timings, call counts and cProfile entries. Timing values
vary with the host; workload/source identities are deterministic.

| Component | Instrumented baseline profile |
|---|---|
| advance | 38135 calls; self 0.122567s; cumulative 1.251427s |
| acceleration_from_action | 38135 calls; self 0.096613s; cumulative 0.375266s |
| power_kw | 38135 calls; self 0.026654s; cumulative 0.707622s; 38135 calls; self 0.061147s; cumulative 0.680968s |
| predict | 38135 calls; self 0.483446s; cumulative 0.590981s |
| certify | 880 calls; self 0.092020s; cumulative 3.873415s |
| asdict | 77985 calls; self 0.048457s; cumulative 2.082733s |
| digest | 1715 calls; self 0.004301s; cumulative 0.173495s |

Cumulative timings overlap and must not be summed. In particular the metered
power_kw wrapper includes the underlying power_kw. Candidate masks made 36165
complete production calls in this 40-mask workload; selected verification adds 1970.
Each production call includes one power MLP. Mask candidates retain the same
number of exact kinematic predictions but avoid all their power calls in V3.
The profile shows substantial asdict/transcript-digest cost as well as neural
inference; the speedup cannot be attributed only to the neural network.

Causal envelope ingest, independent all-grid replay, serialization, hashing,
compression and optional inherited+1 checks are separately timed per speed in
baseline_profile.json. V2 evaluation also computes a shadow mask and separate
final certificate for CAUSAL60; V3 removes both from that reference arm. Legacy
actual proposal checks run once, with full inherited-grid diagnostics only on
Banker steps divisible by100. The declared scientific scheme never ran on
Development after the resource stop. Selected Banker full-production verification
remains, and independent all-grid replay remains intentionally costly.

The fair matched comparison is latency_results.json, without cProfile,280 queries
uniform over seven stated speeds (40 per state). Allocation probes use tracemalloc
separately and are excluded from latency summaries. They measure traced Python
allocations/peak bytes, not all native allocator activity. The original and fast
certificates have different storage content; no allocation reduction is assumed.

| Initial speed m/s | Frozen mean mask ms | V3 mean mask ms | Kinematic predictions/mask | V3 mask power calls |
|---:|---:|---:|---:|---:|
| 0.000000 | 1.933 | 0.272 | 21 | 0 |
| 5.555556 | 2.199 | 0.321 | 32 | 0 |
| 10.000000 | 10.132 | 1.804 | 342 | 0 |
| 16.666667 | 23.535 | 4.200 | 859 | 0 |
| 20.000000 | 31.707 | 5.851 | 1188 | 0 |
| 25.000000 | 46.172 | 8.546 | 1760 | 0 |
| 25.500000 | 48.292 | 8.612 | 1821 | 0 |

Overall frozen mean mask 23.424163ms;
V3 4.229461ms (5.538x).
V3 mask p95 8.666361ms,
p99 9.009747ms,
observed max 15.581542ms.
Mask+full final observed max 20.387959ms<100ms.
This finite synthetic sample does not establish hard real-time 10 Hz operation.
Mean/median/p95/p99/max for full selected certificates, backup witnesses and total
safety overhead, raw samples and complete state/speed distribution are retained.
