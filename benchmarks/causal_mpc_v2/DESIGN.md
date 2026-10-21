# MPC V2 Design

## Primary optimizer

V2 uses a deterministic bounded multi-phase profile optimizer. A frozen
17-profile lattice
of policies is rolled out to the fixed spatial waypoint with the exact
production `advance` function. Profiles include regulated cruise, pulse/coast,
traction-then-coast, traction/coast/regenerative-brake and coast-then-partial-
traction families. The action modes are `+1.0`, `+0.5`, `0.0`, `-0.5` and
`-1.0`; exact speed-cap actions may be fractional. Thus one trajectory can
accelerate early and coast or brake later.

Generation never receives lambda. For every causal state, all prices score the
same feasible candidate tuple. Selection minimizes

`horizon energy + terminal energy + lambda * (horizon time + terminal time)`.

Deterministic ties use quantized objective, shorter total estimated duration,
lower total estimated energy and profile ID. This construction makes selected
duration non-increasing in lambda up to the declared `1e-12 kWh` tie quantum.

For `P` profiles and at most `H` native steps, time is `O(PH)`, memory is
`O(PH)` because executable action sequences are retained. Numerical propagation
is float64 and calls production dynamics directly, so there is no separate
surrogate-integration error; objective ties are quantized at `1e-12 kWh`.
Floating-point round-off is not assigned a formal bound. The 17-profile lattice
has an unbounded structural optimization gap relative to continuous control;
the study makes no global-optimality claim.

## Runtime and causality

At each 1 Hz boundary the selector emits one scalar price, the optimizer plans,
and ten native actions are exposed. Each 0.1 s proposal passes through the
unchanged endpoint, known-transition, unseen-transition and deadline checks.
Newly visible transitions, an exhausted segment or an override invalidates the
plan. Controller state advances only from the executed action.

Online inputs are the ordinary 8D observation, propagated state/time, visible
and remembered transitions, exact physics, task/generator constants and lambda.
There is no Track, reset seed, Oracle decision, future simulator state or panel
outcome in the controller API.
