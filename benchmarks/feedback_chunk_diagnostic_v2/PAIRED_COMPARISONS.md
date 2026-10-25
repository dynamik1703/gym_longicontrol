# Paired comparisons

All method-level results group the three training seeds within each route. The
35 jointly feasible routes are the independent units; the 108 learned outcomes
are never treated as 108 independent routes. Positive saving favors the
left-hand method.

| Comparison | Mean saving | Median | Mean absolute difference (kWh) | Route bootstrap 95% |
|---|---:|---:|---:|---:|
| CHUNK3 vs RESIDUAL1S | +0.303% | +0.517% | +0.000548 | [-0.156%, +0.756%] |
| CHUNK3 vs AdaptivePrice | -1.104% | -0.931% | -0.001820 | [-1.960%, -0.252%] |
| CHUNK3 vs FixedPrice | -0.343% | -0.708% | -0.000534 | [-1.124%, +0.446%] |
| CHUNK3 vs Hybrid | +3.489% | +3.315% | +0.006147 | [+2.365%, +4.576%] |
| RESIDUAL1S vs AdaptivePrice | -1.430% | -1.046% | -0.002368 | [-2.408%, -0.487%] |
| Hybrid vs TIME_AWARE | +0.937% | +1.309% | +0.001683 | [+0.413%, +1.402%] |

There are no additional CHUNK3 failures in any comparison and no seed-specific
feasibility regression. However, CHUNK3 misses the frozen +1% RESIDUAL1S target
and is worse, not 3% better, than AdaptivePrice. Mean per-route training-seed
energy standard deviation is 0.001035 kWh for CHUNK3, 0.001086 kWh for
RESIDUAL1S, and 0.001049 kWh for Hybrid.

Full per-route values and all 10,000-resample distributions are in
`paired_comparisons.json`.
