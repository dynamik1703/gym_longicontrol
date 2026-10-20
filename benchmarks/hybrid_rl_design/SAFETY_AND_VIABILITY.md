# Safety and Deadline Viability

## Action-acceptance rule

At native step `t`, the frozen causal controller first ingests the ordinary 8D
observation and computes its unchanged fallback action. The strategy target is
converted into a native proposal using exact known dynamics. The proposal is
applied only when all four gates pass:

1. **Right endpoint:** exact one-step `advance` gives
   `v[t+1] <= limit(x[t+1]) - 1e-6 m/s`, including a remembered transition
   crossed within the step.
2. **Known transitions:** from the proposed next state, exact maximum-braking
   distance plus the frozen 0.5 m position margin fits before every remembered
   lower limit.
3. **Unknown preview:** exact maximum braking from the proposed next state to
   `20 km/h - 1e-6 m/s` fits inside 150 m minus the frozen position margin. The
   target is independently bounded by the frozen 60 km/h cap.
4. **Deadline recoverability:** the causal pessimistic completion estimate plus
   a 2 s reserve fits before 140 s. Before this can be established, only a
   proposal numerically identical to the frozen fallback is accepted.

Any failed gate executes the unchanged frozen causal action. Rejection reasons
are exclusive and ordered: right endpoint, known-transition braking, unknown
preview braking, deadline recoverability.

## Deadline condition

For proposed next state `s+`, the runtime condition constructs an all-20-km/h
unknown-tail envelope:

```text
if velocity(s+) < 20 km/h - margin:
    integrate exact maximum traction until the floor is reached
completion_bound = resulting_time
                 + remaining_distance / (20 km/h - margin)
accept temporal gate iff completion_bound <= 140 s - 2 s
```

This explicitly prevents an energy-seeking policy from spending time reserve
that the causal envelope cannot recover. At reset the condition is deliberately
false: an all-20-km/h 1,000 m tail does not fit. RL can initially reproduce but
not slow the fallback. As progress accumulates and a conservative deadline
reserve becomes available, lower targets may be accepted.

The condition is falsifiable. Every accepted, certified fixture step is checked
against the eventual original-environment completion time. Current fixtures
show zero falsifications, exact speed compliance and deadline completion.

## What is and is not guaranteed

The gate proves one-step endpoint compliance under exact arithmetic used by the
production dynamics and checks exact braking paths for known transitions. The
60 km/h cap plus exact 60→20 braking check gives a causal safety envelope for a
new minimum-limit sign appearing at the 150 m horizon.

It does **not** establish a formal global theorem for every possible generated
track. In particular:

- the deadline estimate is a conservative runtime condition whose dominance
  over every discrete fallback/transition sequence has fixture evidence, not a
  complete mathematical proof;
- numerical/software faults and model mismatch are outside the simulator-only
  claim;
- a future task family with lower limits, shorter preview, different dynamics
  or different deadlines invalidates the frozen assumptions;
- safety is not feasibility: endpoint/braking gates alone permit crawling, so
  the deadline gate and terminal success evaluation remain mandatory;
- switching to fallback after arbitrary unrestricted RL history is not assumed
  safe. The shield constrains every accepted deviation from the start.

If any bound is falsified, fallback parity changes, or a final episode violates
speed, the architecture fails its gate. The response is investigation and a new
study, not a relaxed tolerance or repaired result.

## Deterministic verification

Six fixtures cover constant high limit, rising limits, one maximum legal
downshift, consecutive downshifts, a short braking window and a speed transition
at the route endpoint. Disabled Hybrid influence exactly matches 4,648 frozen
fallback actions and traces across those fixtures. Active time-aware control
generated 7,609 proposals:

- override fraction: 0.17598;
- executed-action influence fraction: 0.63385;
- all six fixtures influenced;
- zero speed/deadline/feasibility failures;
- zero deadline-bound falsifications;
- `SHIELD_DOMINATES_POLICY = false`.

These are correctness fixtures, not performance estimates or substitutes for
the frozen future Diagnostic and Validation sets.
