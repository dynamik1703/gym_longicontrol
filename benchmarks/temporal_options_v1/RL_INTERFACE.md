# Future RL Interface (No Training in This Study)

The discrete action order is:

```text
0: CONTINUE
1..17: START(profile_0)..START(profile_16)
```

`CONTINUE` is masked when no option is active. The safety fallback is outside the
learner's action set and always remains available to the supervisor.

The causal input is 70-dimensional:

| Slice | Width |
| --- | ---: |
| inherited causal strategy observation | 16 |
| active-option flag | 1 |
| active profile one-hot | 17 |
| phase one-hot (`inactive`, regulated, pulse traction, traction, coast, partial traction, brake, fallback) | 8 |
| progress, normalized elapsed lifetime, normalized remaining distance | 3 |
| previous termination/interruption one-hot | 7 |
| valid-action mask | 18 |
| **Total** | **70** |

No route object, hidden tail, task ID, reset seed, Oracle action/trajectory or
future realized state enters these features. Equal first native actions do not
make actions equivalent: anchor, phase, subsequent memory and admissibility must
also match. The implementation performs no action merging.

Each future transition record contains proposed high-level action, actual native
duration `k`, option state before/after, native reward and signed energy sums,
speed cost, episode/bootstrap masks and interventions. For native discount
`gamma_native`, a later learner must bootstrap with `gamma_native ** k` for the
duration actually executed. A one-second switch does not pretend that an option
completed. No optimization or gradient update is implemented here.

