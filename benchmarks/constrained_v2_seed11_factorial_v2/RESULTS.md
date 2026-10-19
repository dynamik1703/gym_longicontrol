# Results

## Outcome

`EXECUTION_GATE_FAILED`

The proposed frozen diagnostic set is not route-clean against every task exposure
possible in the exact training streams. Diagnostic seed `9000005` and
`T_CURRENT` episode index 559 (explicit reset seed `1141367263`) have identical
canonical route and full-task hashes:

```text
route_sha256     37421e43f27b4e72f5516f962be36e23c86f15be602f66ecd23151c561b6369c
full_task_sha256 3a04942965fea88d7969485a38756d84bb473f0ee60d1757572557d31f094c6b
```

The exact current stream remains unfiltered, as required. The diagnostic task was
not replaced. Main training was not authorized.

## Integrity evidence

- Exact minimum physical episode length: 347 native transitions.
- Maximum complete episodes in 300,000 transitions: 864.
- Maximum task exposures including a final active episode: 865.
- Reconstructed `T_HIST` tasks: 865; diagnostic collisions: 0.
- Reconstructed `T_CURRENT` tasks: 865; diagnostic collisions: 1.
- Exact historical first-100 stream matches the reproducibility audit: yes.
- Exact current first-100 stream matches the reproducibility audit: yes.
- Frozen diagnostic hashes reproduce the content-split audit: yes.
- LongiControl `env.step` calls: 0.

Machine-readable ordered-stream hashes, all 18 per-task checks, duplicate counts,
runtime provenance, and collision evidence are in `results.json`.

## Execution accounting

| Item | Count |
|---|---:|
| Attempts | 0 |
| Conditions started | 0 |
| Real training transitions | 0 |
| Gradient updates | 0 |
| Diagnostic evaluation episodes | 0 |
| Legacy Development episodes | 0 |
| Legacy Validation episodes | 0 |
| Paper episodes | 0 |

There are no diagnostic curves, endpoint counts, failure modes, energy summaries,
exposure diagnostics from trained policies, PID/lambda/alpha trajectories,
factorial contrasts, mechanistic classification, D reproduction result, A
historical-side result, or canonical-baseline recommendation. Producing any of them
would require violating the hard stop.

Legacy Development was not evaluated. Legacy Validation was not evaluated. Paper
tracks remained sealed and uninspected.
