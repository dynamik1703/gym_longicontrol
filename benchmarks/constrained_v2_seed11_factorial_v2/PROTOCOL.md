# Protocol and execution gate

The study descends from content-split audit commit
`ab6df07b2e28fadc87ab05c3f64f76b7b2bef6cb`. Its canonical intent is frozen in
`canonical.json`.

## Route-clean preflight

Before training, every one of the 18 diagnostic tasks must be numerically clean and
have route and full-task hashes absent from both exact training streams over every
task exposure possible within 300,000 native transitions.

The physical action is bounded by 1. At every state, action 1 selects the greatest
available acceleration; dynamics and termination distance are route independent.
That maximal trajectory completes 1,000 m in 347 native steps. Consequently a
300,000-transition run can complete at most 864 episodes and expose at most 865
tasks including its final active, lifetime-truncated episode.

Canonical reconstruction finds:

- `T_HIST[0:865]`: no diagnostic numeric, route, or full-task collision.
- `T_CURRENT[559]`: reset seed `1141367263` has the route and full-task content of
  diagnostic seed `9000005`.
- The collision is content-based; no diagnostic numeric seed was drawn.

Frozen collision hashes:

```text
route_sha256     37421e43f27b4e72f5516f962be36e23c86f15be602f66ecd23151c561b6369c
full_task_sha256 3a04942965fea88d7969485a38756d84bb473f0ee60d1757572557d31f094c6b
```

## Mandatory stop

The prompt requires an immediate stop if any collision is discovered and forbids
replacement of individual diagnostic tracks. The study therefore enters
`EXECUTION_GATE_FAILED`.

The following downstream operations are not authorized and were not run:

- learner initialization or equality preflight;
- collector implementation/semantic execution tests;
- runtime and disk pilots;
- Attempt creation or recovery/resume tests;
- any A/B/C/D training;
- any diagnostic, legacy Development, or legacy Validation episode;
- any inspection, hashing, instantiation, or evaluation of Paper tracks.
