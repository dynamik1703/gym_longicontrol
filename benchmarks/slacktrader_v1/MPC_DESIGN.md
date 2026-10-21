# Fixed-Distance MPC Design

## Formulation

From the propagated state, the waypoint is
`min(position + 100 m, route end)`. Nine deterministic target-speed candidates
from 20 to 60 km/h in 5 km/h increments are simulated with the exact 0.1 s
production dynamics until they cross that waypoint. This is a spatial horizon;
candidate durations differ.

Each native rollout step respects the causally known current limit, right
endpoint limit and braking feasibility for remembered transitions. A candidate
must make positive progress and reach the waypoint within 1,800 native steps.
The solve cost is O(9H), with H the native steps needed to traverse 100 m.

## Terminal treatment

To prevent a short horizon from winning merely by ending slowly, a candidate
below the common causal terminal reference is charged the exact production-
dynamics positive-traction energy needed to regain
`min(60 km/h, known limit at waypoint)`. This continuation uses no future route
state and contributes energy only; horizon time remains the spatial travel
time priced by lambda. The treatment is deterministic and consistent across
all price arms.

## Numerics

- Float64 throughout planning.
- Objective tie quantization: `1e-12` kWh.
- Tie order: shorter duration, lower horizon-plus-terminal energy, lower target.
- First ten native actions form the 1 s segment.
- New visible transitions, exhausted segments or a rejected proposal cause
  causal replanning at the current state.

The solver is reproducible and practical (~28.6 ms weighted mean per solve on
the capability partition), but its restricted candidate family did not produce
a sufficiently strong energy controller.
