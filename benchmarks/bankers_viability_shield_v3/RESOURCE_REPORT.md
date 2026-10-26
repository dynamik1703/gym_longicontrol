# Frozen V3 resource result — BANKERS_V3_RESOURCE_STOP
Compute projection 14043.366385383s = 3.900935107h <= 6 h: PASS.
Retained-artifact projection 3499446104.000bytes = 3.259113155GiB > 3 GiB: FAIL.
The failed storage planning bound stops Development BEFORE namespace/task creation.
No actual compute or disk overrun occurred. No formula, panel size, representation
or verification requirement was changed after observing this failure.

The source/protocol was committed at 4e2b096a024ef8d12b1567655e5496e31f86b9c0 before preflight.
RESOURCE_PROTOCOL.md and canonical.json fix the method-specific formula and 1.5
factor. Each method pays its own worst seven-band measured online+replay+I/O cost
at all 32400 planned decisions. The largest cross-method cost is
0.126225967s/step. All28 probe workloads and
actual speed distributions are in resource_measurements.json. Independent full
production replay of all applicable certificates dominates the high-speed cost.

| Method | Worst charged ms/step | Max compressed bytes/step |
|---|---:|---:|
| CAUSAL60 | 6.620 | 352.8 |
| LEGACY_NATIVE_POST_GREEDY | 6.801 | 457.4 |
| BANKERS_GREEDY | 126.226 | 34074.6 |
| BANKERS_MASKED_CAUSAL60 | 125.735 | 33065.6 |

Compute components: 1.5*32400*sum(method costs), preflight CPU
198.585237s, prior charged CPU 47.261302s,
and900s preparation/source/reporting allowance. The prior term includes quality
phase CPU and an additional separately charged differential CPU term, deliberately
conservatively double charging that component. This does not change the frozen
formula. Final readback/sealing/report CPU is accounted separately in
resource_accounting.json and covered by the900s allowance.

Storage components: pre-report retained bytes 55206808,
1.5*32400*sum(method maximum compressed bytes/step),125072640bytes for one live
native journal, and16MiB reports. High-speed Banker evidence is33-34kB/step:
explicit binary64 transcripts for every candidate remain much larger than V2's
hash-only backup transcript evidence. This V3 representation is versioned and
independently replayable but fails its conservative storage projection. Its
execution-safe evidence choice therefore creates a valid negative engineering
outcome, despite reduced online mask cost. No post-result compression redesign.

Peak preflight RSS 378355712bytes. Actual package size and
accounted CPU appear in resource_accounting.json; they remain below hard limits.
V2's immutable8.94h projection is not replaced. Comparing V2 and V3 projection
numbers combines different preregistered accounting models and evidence contracts;
the isolated kernel speedup is established only by matched measurements.
